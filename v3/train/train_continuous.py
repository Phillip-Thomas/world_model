"""
Continuous Training Loop (v3)
==============================
Train VQ-VAE, World Model, and Policy together in a unified loop.

Key improvements over v2:
1. Efficient token storage (30-60x memory reduction)
2. Support for continuous VQ-VAE/WM updates
3. Codebook rehearsal to prevent catastrophic forgetting
4. Optional curiosity-driven exploration

Training modes:
- staged: Traditional VQ-VAE -> WM -> Policy (like v2)
- continuous_wm: VQ-VAE frozen, WM + Policy update together
- continuous_full: All three models update (experimental)
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import asdict
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

# Limit OpenCV threads
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# Setup paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V3_DIR = os.path.dirname(_SCRIPT_DIR)
_WORLD_MODEL_DIR = os.path.dirname(_V3_DIR)
_V2_DIR = os.path.join(_WORLD_MODEL_DIR, "v2")

# Add v3 to path for local imports
sys.path.insert(0, _V3_DIR)

# Import from v3 FIRST (before adding v2 which has conflicting module names)
from config.training_config import TrainingConfig, get_default_config
from agents.efficient_buffer import EfficientTokenBuffer, PrioritizedEfficientBuffer, TransitionBatch, ImaginedBuffer
from agents.frame_buffer import FrameBuffer, CodebookRehearsalBuffer

# Remove v3 from path and clear cached modules to avoid conflicts
sys.path.remove(_V3_DIR)
# Clear any cached 'agents' and 'models' modules from v3
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith('agents') or mod_name.startswith('models') or mod_name.startswith('config'):
        del sys.modules[mod_name]

# Now add v2 to path for v2 imports
sys.path.insert(0, _V2_DIR)

# Import from v2
from models.vqvae_hires import VQVAEHiRes
from models.temporal_world_model import TemporalVisualWorldModel
from agents.dqn_agent import DQNAgent, DQNConfig

try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: gymnasium not installed")


# Compute checkpoint root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORLD_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
CHECKPOINTS_ROOT = os.path.join(_WORLD_MODEL_ROOT, "world_model", "checkpoints", "v3")


class VectorizedAtariEnv:
    """
    Vectorized Atari environment wrapper.
    
    Features:
    - Parallel environment execution
    - Frame preprocessing (resize, normalize)
    - Life loss penalty
    """
    
    def __init__(
        self,
        game: str = "ALE/MsPacman-v5",
        n_envs: int = 4,
        frame_skip: int = 4,
        frame_size: Tuple[int, int] = (84, 64),
        max_episode_steps: int = 10000,
    ):
        self.n_envs = n_envs
        self.frame_size = frame_size
        
        # Create environments
        self.envs = []
        for i in range(n_envs):
            env = gym.make(
                game,
                frameskip=1,  # We handle frame skip
                repeat_action_probability=0.0,
                render_mode=None,
            )
            env = gym.wrappers.AtariPreprocessing(
                env,
                noop_max=30,
                frame_skip=frame_skip,
                screen_size=max(frame_size),
                terminal_on_life_loss=False,
                grayscale_obs=False,  # Keep RGB
            )
            self.envs.append(env)
        
        self.n_actions = self.envs[0].action_space.n
        self.prev_lives = [None] * n_envs
        
        # Episode tracking
        self.episode_rewards = [0.0] * n_envs
        self.episode_lengths = [0] * n_envs
        self.completed_episodes = []
    
    def reset(self) -> np.ndarray:
        """Reset all environments."""
        observations = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            obs = self._preprocess(obs)
            observations.append(obs)
            self.prev_lives[i] = info.get('lives', None)
            self.episode_rewards[i] = 0.0
            self.episode_lengths[i] = 0
        
        return np.stack(observations)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            # Life loss penalty
            current_lives = info.get('lives', None)
            if self.prev_lives[i] is not None and current_lives is not None:
                if current_lives < self.prev_lives[i]:
                    reward -= 1.0  # Life loss penalty
            self.prev_lives[i] = current_lives
            
            obs = self._preprocess(obs)
            
            # Track episode stats
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            if done:
                self.completed_episodes.append({
                    'reward': self.episode_rewards[i],
                    'length': self.episode_lengths[i],
                })
                
                # Auto-reset
                obs, info = env.reset()
                obs = self._preprocess(obs)
                self.prev_lives[i] = info.get('lives', None)
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
            
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos
        )
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame: resize and normalize to [-1, 1]."""
        # Resize if needed
        if frame.shape[:2] != self.frame_size:
            frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
        
        # Convert to float and normalize to [-1, 1]
        frame = frame.astype(np.float32) / 127.5 - 1.0
        
        # (H, W, C) -> (C, H, W)
        frame = np.transpose(frame, (2, 0, 1))
        
        return frame
    
    def get_completed_episodes(self) -> List[Dict]:
        """Get and clear completed episode stats."""
        episodes = self.completed_episodes
        self.completed_episodes = []
        return episodes
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class ContinuousTrainer:
    """
    Continuous training loop for World Model-based RL.
    
    Supports three modes:
    1. staged: Train VQ-VAE -> WM -> Policy sequentially
    2. continuous_wm: VQ-VAE frozen, WM + Policy update together
    3. continuous_full: All three models update together
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        vqvae_path: str = None,
        wm_path: str = None,
        policy_path: str = None,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        
        # Initialize environment
        print("\n[1/5] Creating environment...")
        self.env = VectorizedAtariEnv(
            game=config.game,
            n_envs=config.n_envs,
            frame_skip=config.frame_skip,
        )
        
        # Initialize models
        print("[2/5] Loading VQ-VAE...")
        self.vqvae = self._load_vqvae(vqvae_path)
        
        print("[3/5] Loading World Model...")
        self.world_model = self._load_world_model(wm_path)
        
        print("[4/5] Creating Policy...")
        self.policy = self._create_policy(policy_path)
        
        # Initialize buffers
        print("[5/5] Creating buffers...")
        self._create_buffers()
        
        # Optimizers
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config.world_model.finetune_lr,
        )
        
        if config.mode == "continuous_full":
            self.vqvae_optimizer = torch.optim.Adam(
                self.vqvae.parameters(),
                lr=config.vqvae.learning_rate,
            )
        
        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        self.best_eval_reward = float('-inf')
        
        # Logging
        self.train_metrics = []
        self.eval_metrics = []
        self.episode_rewards = []  # Track completed episodes
        self.buffer_sizes = []     # Track buffer growth
        self.recent_episode_reward = 0.0  # For progress bar
        
        # Frame history for each environment
        self.frame_history = [[] for _ in range(config.n_envs)]
        
        # Create run directory
        self.run_dir = config.get_run_dir(
            os.path.join(CHECKPOINTS_ROOT, "mspacman")
        )
        print(f"\nRun directory: {self.run_dir}")
        
        # Save config
        config.save(os.path.join(self.run_dir, "config.json"))
    
    def _load_vqvae(self, path: str = None) -> VQVAEHiRes:
        """Load or create VQ-VAE."""
        vqvae = VQVAEHiRes(
            in_channels=self.config.vqvae.in_channels,
            hidden_channels=self.config.vqvae.hidden_channels,
            latent_channels=self.config.vqvae.latent_channels,
            n_embeddings=self.config.vqvae.n_embeddings,
            input_size=self.config.vqvae.input_size,
        ).to(self.device)
        
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                vqvae.load_state_dict(checkpoint['model_state_dict'])
            else:
                vqvae.load_state_dict(checkpoint)
            print(f"  Loaded VQ-VAE from {path}")
        
        # Freeze if not continuous_full mode
        if self.config.mode != "continuous_full":
            vqvae.freeze_for_world_model()
        
        return vqvae
    
    def _load_world_model(self, path: str = None) -> TemporalVisualWorldModel:
        """Load or create World Model."""
        wm = TemporalVisualWorldModel(
            n_vocab=self.config.world_model.n_vocab,
            d_model=self.config.world_model.d_model,
            n_heads=self.config.world_model.n_heads,
            n_layers=self.config.world_model.n_layers,
            n_actions=self.config.world_model.n_actions,
            token_h=self.config.vqvae.token_h,
            token_w=self.config.vqvae.token_w,
            max_history=self.config.history_len,
        ).to(self.device)
        
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                wm.load_state_dict(checkpoint['model_state_dict'])
            else:
                wm.load_state_dict(checkpoint)
            print(f"  Loaded World Model from {path}")
        
        return wm
    
    def _create_policy(self, path: str = None) -> DQNAgent:
        """Create or load policy."""
        # DQNConfig holds hyperparameters (learning_rate, gamma, etc.)
        policy_config = DQNConfig(
            learning_rate=self.config.policy.learning_rate,
            gamma=self.config.policy.gamma,
            tau=self.config.policy.tau,
            epsilon_start=self.config.policy.epsilon_start,
            epsilon_end=self.config.policy.epsilon_end,
            epsilon_decay_steps=self.config.policy.epsilon_decay_steps,
            grad_clip=self.config.policy.grad_clip,
            token_grid_h=self.config.vqvae.token_h,
            token_grid_w=self.config.vqvae.token_w,
        )
        
        # DQNAgent takes architecture params directly
        policy = DQNAgent(
            n_vocab=self.config.vqvae.n_embeddings,
            n_actions=self.config.policy.n_actions,
            history_len=self.config.history_len,
            n_tokens=self.config.vqvae.n_tokens,
            config=policy_config,
            device=self.device,
        )
        
        if path and os.path.exists(path):
            policy.load(path)
            print(f"  Loaded Policy from {path}")
        
        return policy
    
    def _create_buffers(self):
        """Create replay buffers."""
        # Efficient token buffer
        if self.config.policy.use_prioritized:
            self.token_buffer = PrioritizedEfficientBuffer(
                capacity=self.config.real_buffer_size,
                n_tokens=self.config.vqvae.n_tokens,
                history_len=self.config.history_len,
                vocab_size=self.config.vqvae.n_embeddings,
                alpha=self.config.policy.priority_alpha,
                beta=self.config.policy.priority_beta_start,
            )
        else:
            self.token_buffer = EfficientTokenBuffer(
                capacity=self.config.real_buffer_size,
                n_tokens=self.config.vqvae.n_tokens,
                history_len=self.config.history_len,
                vocab_size=self.config.vqvae.n_embeddings,
            )
        
        # Frame buffer (for continuous_full mode or VQ-VAE analysis)
        if self.config.mode == "continuous_full":
            self.frame_buffer = FrameBuffer(
                capacity=self.config.frame_buffer_size,
                frame_height=self.config.vqvae.input_size[0],
                frame_width=self.config.vqvae.input_size[1],
            )
            
            self.rehearsal_buffer = CodebookRehearsalBuffer(
                n_codes=self.config.vqvae.n_embeddings,
            )
            self.rehearsal_buffer.set_vqvae(self.vqvae, self.device)
        else:
            self.frame_buffer = None
            self.rehearsal_buffer = None
        
        # Imagined buffer for trust-weighted Dyna
        self.imagined_buffer = ImaginedBuffer(
            capacity=self.config.imagined_buffer_size,
            n_tokens=self.config.vqvae.n_tokens,
            history_len=self.config.history_len,
            vocab_size=self.config.vqvae.n_embeddings,
        )
        
        # Track imagination stats
        self.imagination_metrics = []
    
    def _encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """Encode a single frame to tokens."""
        with torch.no_grad():
            frame_t = torch.from_numpy(frame).unsqueeze(0).to(self.device)
            tokens = self.vqvae.encode(frame_t)
            return tokens[0].cpu().numpy().flatten()
    
    def _get_epsilon(self) -> float:
        """Get current exploration epsilon."""
        decay_steps = self.config.policy.epsilon_decay_steps
        eps_start = self.config.policy.epsilon_start
        eps_end = self.config.policy.epsilon_end
        
        progress = min(1.0, self.total_steps / decay_steps)
        return eps_start + (eps_end - eps_start) * progress
    
    def collect_step(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect one step of experience.
        
        Args:
            observations: Current observations (n_envs, C, H, W)
            
        Returns:
            next_observations, actions
        """
        # Encode current frames
        with torch.no_grad():
            obs_t = torch.from_numpy(observations).to(self.device)
            current_tokens = self.vqvae.encode(obs_t).cpu().numpy()
        
        # Build state from history
        states = []
        for i in range(self.config.n_envs):
            # Add current tokens to history
            self.frame_history[i].append(current_tokens[i].flatten())
            
            # Trim history
            if len(self.frame_history[i]) > self.config.history_len:
                self.frame_history[i] = self.frame_history[i][-self.config.history_len:]
            
            # Pad if needed
            while len(self.frame_history[i]) < self.config.history_len:
                self.frame_history[i].insert(0, self.frame_history[i][0].copy())
            
            states.append(np.stack(self.frame_history[i]))
        
        states = np.stack(states)  # (n_envs, history_len, n_tokens)
        
        # Select actions
        epsilon = self._get_epsilon()
        actions = []
        for i in range(self.config.n_envs):
            if np.random.random() < epsilon:
                action = np.random.randint(0, self.env.n_actions)
            else:
                state_t = torch.from_numpy(states[i]).unsqueeze(0).long().to(self.device)
                with torch.no_grad():
                    q_values = self.policy.policy_net(state_t)
                    action = q_values.argmax(dim=1).item()
            actions.append(action)
        
        actions = np.array(actions)
        
        # Step environment
        next_observations, rewards, dones, infos = self.env.step(actions)
        
        # Encode next frames
        with torch.no_grad():
            next_obs_t = torch.from_numpy(next_observations).to(self.device)
            next_tokens = self.vqvae.encode(next_obs_t).cpu().numpy()
        
        # Store transitions
        for i in range(self.config.n_envs):
            # Clip reward (standard Atari DQN)
            clipped_reward = np.sign(rewards[i])
            
            # Store single token frame (efficient!)
            self.token_buffer.add(
                next_tokens[i].flatten().astype(np.uint16),
                actions[i],
                clipped_reward,
                dones[i],
            )
            
            # Store raw frame if continuous_full mode
            if self.frame_buffer is not None:
                # Convert back to uint8 for storage
                frame_uint8 = ((observations[i] + 1) * 127.5).clip(0, 255).astype(np.uint8)
                self.frame_buffer.add(frame_uint8, actions[i], rewards[i], dones[i])
            
            # Reset history on episode end
            if dones[i]:
                self.frame_history[i] = []
                self.total_episodes += 1
        
        self.total_steps += self.config.n_envs
        
        # Track completed episodes from environment
        completed = self.env.get_completed_episodes()
        for ep in completed:
            self.episode_rewards.append({
                'step': self.total_steps,
                'reward': ep['reward'],
                'length': ep['length'],
            })
            self.recent_episode_reward = ep['reward']
        
        return next_observations, actions
    
    def train_policy_step(self) -> Dict:
        """Train policy for one step, mixing real + imagined data."""
        if not self.token_buffer.is_ready(self.config.warmup_steps):
            return {}
        
        total_loss = 0.0
        total_q = 0.0
        imagined_loss = 0.0
        imagined_steps = 0
        
        for _ in range(self.config.gradient_steps):
            # === Real data training ===
            if self.config.policy.use_prioritized:
                batch, weights, indices = self.token_buffer.sample(self.config.policy.batch_size)
                batch = batch.to(self.device)
                weights = weights.to(self.device)
                
                # Train step with prioritized replay
                loss, td_errors = self.policy.train_step_prioritized(batch, weights)
                q_mean = 0.0  # Not returned by prioritized step
                
                # Update priorities
                self.token_buffer.update_priorities(indices, td_errors)
            else:
                batch = self.token_buffer.sample(self.config.policy.batch_size)
                batch = batch.to(self.device)
                loss = self.policy.train_step(batch)
                q_mean = 0.0  # Not returned by train_step
            
            total_loss += loss
            total_q += q_mean
            
            # === Imagined data training (trust-weighted) ===
            if self.imagined_buffer.is_ready(self.config.imagined_batch_size):
                # How many imagined samples? Based on imagined_ratio
                imag_batch_size = int(self.config.policy.batch_size * self.config.imagined_ratio)
                if imag_batch_size > 0:
                    imag_batch, trust_weights = self.imagined_buffer.sample_uniform(imag_batch_size)
                    
                    if imag_batch is not None:
                        imag_batch = imag_batch.to(self.device)
                        trust_weights = trust_weights.to(self.device)
                        
                        # Train with trust-weighted loss
                        imag_loss = self._train_imagined_batch(imag_batch, trust_weights)
                        imagined_loss += imag_loss
                        imagined_steps += 1
        
        result = {
            'policy_loss': total_loss / self.config.gradient_steps,
            'q_mean': total_q / self.config.gradient_steps,
        }
        
        if imagined_steps > 0:
            result['imagined_policy_loss'] = imagined_loss / imagined_steps
        
        return result
    
    def _train_imagined_batch(self, batch: TransitionBatch, trust_weights: torch.Tensor) -> float:
        """
        Train policy on imagined batch with trust-weighted loss.
        
        The trust weights downweight contributions from less reliable WM predictions.
        """
        # Get Q-values for current states
        q_values = self.policy.policy_net(batch.states)
        
        # Get Q-values for next states (from target network)
        with torch.no_grad():
            next_q_values = self.policy.target_net(batch.next_states)
            
            # Double DQN: use online network to select action
            next_actions = self.policy.policy_net(batch.next_states).argmax(dim=1, keepdim=True)
            next_q = next_q_values.gather(1, next_actions).squeeze(1)
            
            # TD target
            target_q = batch.rewards + self.config.policy.gamma * next_q * (1 - batch.dones)
        
        # Current Q for taken actions
        current_q = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        
        # TD error (per sample)
        td_error = target_q - current_q
        
        # Trust-weighted Huber loss
        huber_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        weighted_loss = (trust_weights * huber_loss).mean()
        
        # Backward pass
        self.policy.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.policy_net.parameters(), self.config.policy.grad_clip)
        self.policy.optimizer.step()
        
        # Soft update target network
        self.policy._soft_update()
        
        return weighted_loss.item()
    
    def train_wm_step(self) -> Dict:
        """Train world model for one step."""
        if not self.token_buffer.is_ready(self.config.warmup_steps):
            return {}
        
        self.world_model.train()
        total_loss = 0.0
        
        for _ in range(self.config.world_model.updates_per_step):
            # Sample with recency bias
            batch = self.token_buffer.sample_recent(
                self.config.world_model.batch_size,
                recent_k=self.config.world_model.recent_k,
                recent_frac=self.config.world_model.recent_frac,
            )
            batch = batch.to(self.device)
            
            # Forward pass
            self.wm_optimizer.zero_grad()
            
            # Target: last frame of next_state
            targets = batch.next_states[:, -1, :].long()
            
            # WM forward returns (logits, loss, metrics) when given targets
            logits, loss, _ = self.world_model(
                batch.states.long(),
                batch.actions.long(),
                target_tokens=targets,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(),
                self.config.world_model.grad_clip,
            )
            self.wm_optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'wm_loss': total_loss / self.config.world_model.updates_per_step,
        }
    
    def generate_imagined_rollouts(self) -> Dict:
        """
        Generate trust-weighted imagined rollouts from WM.
        
        This implements trust-gated Dyna:
        1. Sample real transitions as seeds
        2. Compute WM prediction loss on real data to get trust scores
        3. Keep only seeds where WM is accurate (high trust)
        4. Roll out WM for K steps from trusted seeds
        5. Store imagined transitions with trust weights
        """
        if not self.token_buffer.is_ready(self.config.warmup_steps * 2):
            return {}
        
        self.world_model.eval()
        
        with torch.no_grad():
            # 1. Sample real transitions as seeds
            batch = self.token_buffer.sample_recent(
                self.config.imagined_batch_size,
                recent_k=self.config.world_model.recent_k,
                recent_frac=0.8,  # More recent for imagination
            )
            batch = batch.to(self.device)
            
            # 2. Compute per-sample WM loss on real transitions
            targets = batch.next_states[:, -1, :].long()
            _, _, aux = self.world_model(
                batch.states.long(),
                batch.actions.long(),
                target_tokens=targets,
            )
            
            # Get per-sample loss for trust calculation
            per_sample_loss = aux.get('token_loss_per_sample', None)
            if per_sample_loss is None:
                # Fallback: use scalar loss as uniform trust
                per_sample_loss = torch.full(
                    (batch.states.shape[0],),
                    aux['token_loss'].item(),
                    device=self.device
                )
            
            # 3. Compute trust and filter seeds
            trust = torch.exp(-self.config.trust_scale * per_sample_loss)
            keep_mask = trust > self.config.trust_threshold
            
            # Track stats
            n_seeds = batch.states.shape[0]
            n_kept = keep_mask.sum().item()
            self.imagined_buffer.total_generated += n_seeds
            self.imagined_buffer.total_accepted += n_kept
            
            if n_kept == 0:
                return {
                    'imagined_seeds': n_seeds,
                    'imagined_accepted': 0,
                    'imagined_mean_trust': trust.mean().item(),
                }
            
            # Filter to trusted seeds
            trusted_states = batch.next_states[keep_mask]  # Start from next_state
            trusted_trust = trust[keep_mask]
            
            # 4. Roll out from trusted seeds
            current_state = trusted_states.long()  # (n_kept, T, N)
            
            for step in range(self.config.imagined_rollout_len):
                # Select actions using current policy (with some exploration)
                epsilon = max(0.1, self._get_epsilon() * 0.5)  # Some exploration
                
                B = current_state.shape[0]
                actions = []
                for i in range(B):
                    if np.random.random() < epsilon:
                        action = np.random.randint(0, self.env.n_actions)
                    else:
                        q_values = self.policy.policy_net(current_state[i:i+1])
                        action = q_values.argmax(dim=1).item()
                    actions.append(action)
                
                actions_t = torch.tensor(actions, device=self.device, dtype=torch.long)
                
                # WM prediction
                next_tokens, reward_pred, done_pred = self.world_model.forward_with_heads(
                    current_state,
                    actions_t,
                    deterministic=True,
                )
                
                # Decay trust with horizon
                step_trust = trusted_trust * (self.config.trust_horizon_decay ** step)
                
                # Build next_state by shifting and appending new tokens
                next_state = torch.cat([
                    current_state[:, 1:, :],  # Drop oldest frame
                    next_tokens.unsqueeze(1),  # Add predicted frame
                ], dim=1)
                
                # 5. Store imagined transitions
                states_np = current_state.cpu().numpy().astype(np.uint16)
                actions_np = actions_t.cpu().numpy().astype(np.int8)
                rewards_np = reward_pred.cpu().numpy().astype(np.float32)
                next_states_np = next_state.cpu().numpy().astype(np.uint16)
                dones_np = done_pred.cpu().numpy().astype(np.float32)
                trust_np = step_trust.cpu().numpy().astype(np.float32)
                
                self.imagined_buffer.add_batch(
                    states_np, actions_np, rewards_np,
                    next_states_np, dones_np, trust_np
                )
                
                # Update state for next step, filter out "done" episodes
                current_state = next_state
                alive_mask = done_pred < 0.5
                if alive_mask.sum() == 0:
                    break
                current_state = current_state[alive_mask]
                trusted_trust = trusted_trust[alive_mask]
        
        self.world_model.train()
        
        return {
            'imagined_seeds': n_seeds,
            'imagined_accepted': n_kept,
            'imagined_mean_trust': trust.mean().item(),
            'imagined_buffer_size': len(self.imagined_buffer),
        }
    
    def evaluate(self) -> Dict:
        """Run evaluation episodes."""
        self.policy.policy_net.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(self.config.eval_episodes):
            # Create single eval env
            eval_env = gym.make(
                self.config.game,
                frameskip=1,
                repeat_action_probability=0.0,
            )
            eval_env = gym.wrappers.AtariPreprocessing(
                eval_env,
                noop_max=30,
                frame_skip=self.config.frame_skip,
                screen_size=84,
                terminal_on_life_loss=False,
                grayscale_obs=False,
            )
            
            obs, _ = eval_env.reset()
            obs = self._preprocess_eval(obs)
            
            history = []
            total_reward = 0.0
            length = 0
            done = False
            
            while not done and length < 10000:
                # Encode and build state
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    tokens = self.vqvae.encode(obs_t).cpu().numpy().flatten()
                
                history.append(tokens)
                if len(history) > self.config.history_len:
                    history = history[-self.config.history_len:]
                while len(history) < self.config.history_len:
                    history.insert(0, history[0].copy())
                
                state = np.stack(history)
                state_t = torch.from_numpy(state).unsqueeze(0).long().to(self.device)
                
                # Greedy action
                with torch.no_grad():
                    q_values = self.policy.policy_net(state_t)
                    action = q_values.argmax(dim=1).item()
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                obs = self._preprocess_eval(obs)
                
                total_reward += reward
                length += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            eval_env.close()
        
        self.policy.policy_net.train()
        
        return {
            'eval_reward_mean': np.mean(episode_rewards),
            'eval_reward_std': np.std(episode_rewards),
            'eval_length_mean': np.mean(episode_lengths),
            'eval_rewards': episode_rewards,
        }
    
    def _preprocess_eval(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for evaluation."""
        if frame.shape[:2] != self.config.vqvae.input_size:
            frame = cv2.resize(frame, (self.config.vqvae.input_size[1], self.config.vqvae.input_size[0]))
        frame = frame.astype(np.float32) / 127.5 - 1.0
        frame = np.transpose(frame, (2, 0, 1))
        return frame
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save training checkpoint."""
        checkpoint = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_eval_reward': self.best_eval_reward,
            'train_metrics': self.train_metrics,
            'eval_metrics': self.eval_metrics,
            'world_model': self.world_model.state_dict(),
            'policy_net': self.policy.policy_net.state_dict(),
            'target_net': self.policy.target_net.state_dict(),
            'policy_optimizer': self.policy.optimizer.state_dict(),
            'policy_epsilon': self.policy.epsilon,
            'policy_train_steps': self.policy.train_steps,
            'wm_optimizer': self.wm_optimizer.state_dict(),
        }
        
        path = os.path.join(self.run_dir, f"{name}.pt")
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
        
        # Also save policy in v2-compatible format for easy loading
        policy_path = os.path.join(self.run_dir, f"policy_{name}.pt")
        self.policy.save(policy_path)
    
    def _compute_ema(self, values: list, alpha: float = 0.3) -> list:
        """Compute exponential moving average."""
        if not values:
            return []
        ema = [values[0]]
        for v in values[1:]:
            ema.append(alpha * v + (1 - alpha) * ema[-1])
        return ema
    
    def plot_progress(self):
        """Plot comprehensive training progress (6-panel like v2)."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Helper to get metrics safely
        def get_metric(metrics_list, key):
            steps = [m['step'] for m in metrics_list if key in m]
            values = [m[key] for m in metrics_list if key in m]
            return steps, values
        
        # 1. Training Loss (top-left)
        ax = axes[0, 0]
        steps, losses = get_metric(self.train_metrics, 'policy_loss')
        if steps:
            ax.plot(steps, losses, 'b-', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        
        # 2. Episode Rewards (top-middle)
        ax = axes[0, 1]
        # Training rewards from completed episodes
        if hasattr(self, 'episode_rewards') and self.episode_rewards:
            ep_steps = [e['step'] for e in self.episode_rewards]
            ep_rewards = [e['reward'] for e in self.episode_rewards]
            ax.plot(ep_steps, ep_rewards, 'g-', alpha=0.4, label='Train')
        # Eval rewards
        if self.eval_metrics:
            eval_steps = [m['step'] for m in self.eval_metrics]
            eval_rewards = [m['eval_reward_mean'] for m in self.eval_metrics]
            ax.plot(eval_steps, eval_rewards, 'ro', markersize=6, alpha=0.5, label='Eval')
            # EMA of eval rewards
            if len(eval_rewards) >= 2:
                eval_ema = self._compute_ema(eval_rewards, alpha=0.3)
                ax.plot(eval_steps, eval_ema, 'r-', linewidth=2.5, label='Eval EMA')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Exploration Rate (top-right)
        ax = axes[0, 2]
        steps, eps_vals = get_metric(self.train_metrics, 'epsilon')
        if steps:
            ax.plot(steps, eps_vals, 'purple', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate')
        ax.grid(True, alpha=0.3)
        
        # 4. WM Loss (bottom-left) - new for v3!
        ax = axes[1, 0]
        steps, wm_losses = get_metric(self.train_metrics, 'wm_loss')
        if steps:
            ax.plot(steps, wm_losses, 'orange', alpha=0.7)
            ax.set_title('World Model Loss')
        else:
            ax.set_title('World Model Loss (no data)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # 5. Episode Length (bottom-middle)
        ax = axes[1, 1]
        if hasattr(self, 'episode_rewards') and self.episode_rewards:
            ep_steps = [e['step'] for e in self.episode_rewards if 'length' in e]
            ep_lengths = [e['length'] for e in self.episode_rewards if 'length' in e]
            if ep_steps:
                ax.plot(ep_steps, ep_lengths, 'cyan', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Length')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        
        # 6. Buffer Size (bottom-right) - show real + imagined
        ax = axes[1, 2]
        if hasattr(self, 'buffer_sizes') and self.buffer_sizes:
            buf_steps = [b['step'] for b in self.buffer_sizes]
            buf_sizes = [b['size'] for b in self.buffer_sizes]
            ax.plot(buf_steps, buf_sizes, 'b-', alpha=0.7, label='Real')
        # Add imagined buffer size from imagination_metrics
        if hasattr(self, 'imagination_metrics') and self.imagination_metrics:
            imag_steps = [m['step'] for m in self.imagination_metrics if 'imagined_buffer_size' in m]
            imag_sizes = [m['imagined_buffer_size'] for m in self.imagination_metrics if 'imagined_buffer_size' in m]
            if imag_steps:
                ax.plot(imag_steps, imag_sizes, 'r-', alpha=0.7, label='Imagined')
        ax.set_xlabel('Step')
        ax.set_ylabel('Size')
        ax.set_title('Buffer Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'training_progress.png'), dpi=150)
        plt.close()
        
        # Also save text stats
        self._save_training_stats()
    
    def _save_training_stats(self):
        """Save training stats to text file."""
        path = os.path.join(self.run_dir, 'training_stats.txt')
        with open(path, 'w') as f:
            f.write("Continuous Training Statistics (v3)\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            f.write(f"Mode: {self.config.mode}\n")
            f.write(f"Total Steps: {self.total_steps:,}\n")
            f.write(f"Total Episodes: {self.total_episodes:,}\n")
            f.write(f"Best Eval Reward: {self.best_eval_reward:.1f}\n\n")
            
            # Latest eval
            if self.eval_metrics:
                latest = self.eval_metrics[-1]
                f.write(f"Latest Eval Reward: {latest['eval_reward_mean']:.1f} +/- {latest['eval_reward_std']:.1f}\n")
            
            # Latest training metrics
            policy_losses = [m['policy_loss'] for m in self.train_metrics if 'policy_loss' in m]
            if policy_losses:
                f.write(f"Latest Policy Loss: {policy_losses[-1]:.4f}\n")
            
            wm_losses = [m['wm_loss'] for m in self.train_metrics if 'wm_loss' in m]
            if wm_losses:
                f.write(f"Latest WM Loss: {wm_losses[-1]:.4f}\n")
            
            f.write(f"\nBuffer Size: {len(self.token_buffer):,}\n")
            f.write(f"Epsilon: {self._get_epsilon():.4f}\n")
            
            # Imagination stats
            if len(self.imagined_buffer) > 0:
                imag_stats = self.imagined_buffer.get_stats()
                f.write(f"\n--- Imagination (Trust-Weighted Dyna) ---\n")
                f.write(f"Imagined Buffer Size: {imag_stats['size']:,}\n")
                f.write(f"Accept Rate: {imag_stats['accept_rate']*100:.1f}%\n")
                f.write(f"Mean Trust: {imag_stats['mean_trust']:.3f}\n")
                f.write(f"Trust Percentiles (10/50/90): "
                       f"{imag_stats.get('p10_trust', 0):.3f} / "
                       f"{imag_stats.get('p50_trust', 0):.3f} / "
                       f"{imag_stats.get('p90_trust', 0):.3f}\n")
            
            # Eval history
            if self.eval_metrics:
                f.write("\n" + "=" * 50 + "\n")
                f.write("Evaluation History\n")
                f.write("=" * 50 + "\n")
                f.write(f"{'Step':>10} {'Reward':>10} {'Std':>10}\n")
                f.write("-" * 32 + "\n")
                for m in self.eval_metrics:
                    f.write(f"{m['step']:>10,} {m['eval_reward_mean']:>10.1f} {m['eval_reward_std']:>10.1f}\n")
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Continuous Training (v3)")
        print("=" * 60)
        self.config.print_summary()
        
        # Reset environment
        observations = self.env.reset()
        
        # Progress bar
        pbar = tqdm(total=self.config.total_steps, desc="Training")
        pbar.update(self.total_steps)
        
        last_train_step = 0
        last_wm_step = 0
        last_imag_step = 0
        last_eval_step = 0
        last_save_step = 0
        
        while self.total_steps < self.config.total_steps:
            # Collect experience
            observations, actions = self.collect_step(observations)
            
            # Train policy
            if self.total_steps - last_train_step >= self.config.train_freq:
                if self.token_buffer.is_ready(self.config.warmup_steps):
                    metrics = self.train_policy_step()
                    metrics['step'] = self.total_steps
                    metrics['epsilon'] = self._get_epsilon()
                    self.train_metrics.append(metrics)
                last_train_step = self.total_steps
            
            # Train world model
            if self.total_steps - last_wm_step >= self.config.world_model.update_every_n_steps:
                if self.token_buffer.is_ready(self.config.warmup_steps):
                    wm_metrics = self.train_wm_step()
                    if wm_metrics:
                        wm_metrics['step'] = self.total_steps
                        self.train_metrics.append(wm_metrics)
                last_wm_step = self.total_steps
            
            # Generate imagined rollouts (trust-weighted Dyna)
            if self.total_steps - last_imag_step >= self.config.imagined_update_freq:
                if self.token_buffer.is_ready(self.config.warmup_steps * 2):
                    imag_metrics = self.generate_imagined_rollouts()
                    if imag_metrics:
                        imag_metrics['step'] = self.total_steps
                        self.imagination_metrics.append(imag_metrics)
                last_imag_step = self.total_steps
            
            # Evaluate
            if self.total_steps - last_eval_step >= self.config.eval_freq:
                print(f"\n[Step {self.total_steps:,}] Evaluating...")
                eval_metrics = self.evaluate()
                eval_metrics['step'] = self.total_steps
                self.eval_metrics.append(eval_metrics)
                
                # Track buffer size
                self.buffer_sizes.append({
                    'step': self.total_steps,
                    'size': len(self.token_buffer),
                })
                
                # Get recent losses for reporting
                recent_policy_loss = None
                recent_wm_loss = None
                for m in reversed(self.train_metrics[-100:]):
                    if recent_policy_loss is None and 'policy_loss' in m:
                        recent_policy_loss = m['policy_loss']
                    if recent_wm_loss is None and 'wm_loss' in m:
                        recent_wm_loss = m['wm_loss']
                    if recent_policy_loss and recent_wm_loss:
                        break
                
                # Print comprehensive summary
                print(f"  Eval reward: {eval_metrics['eval_reward_mean']:.1f} +/- {eval_metrics['eval_reward_std']:.1f}")
                if recent_policy_loss is not None:
                    print(f"  Policy loss: {recent_policy_loss:.4f}")
                if recent_wm_loss is not None:
                    print(f"  WM loss: {recent_wm_loss:.4f}")
                print(f"  Epsilon: {self._get_epsilon():.3f}")
                print(f"  Buffer: {len(self.token_buffer):,} transitions")
                print(f"  Episodes: {self.total_episodes:,}")
                
                # Imagination stats
                if len(self.imagined_buffer) > 0:
                    imag_stats = self.imagined_buffer.get_stats()
                    print(f"  Imagined: {imag_stats['size']:,} transitions, "
                          f"trust={imag_stats['mean_trust']:.3f}, "
                          f"accept={imag_stats['accept_rate']*100:.1f}%")
                
                # Save best
                if eval_metrics['eval_reward_mean'] > self.best_eval_reward:
                    self.best_eval_reward = eval_metrics['eval_reward_mean']
                    self.save_checkpoint("best")
                    print(f"  -> New best! Saved checkpoint.")
                
                self.plot_progress()
                last_eval_step = self.total_steps
            
            # Save checkpoint
            if self.total_steps - last_save_step >= self.config.save_freq:
                self.save_checkpoint(f"step_{self.total_steps}")
                last_save_step = self.total_steps
            
            # Update progress bar with more info
            pbar.update(self.config.n_envs)
            postfix = {
                'eps': f"{self._get_epsilon():.3f}",
                'rew': f"{self.recent_episode_reward:.0f}",
                'buf': f"{len(self.token_buffer):,}",
            }
            # Add latest loss if available
            if self.train_metrics:
                for m in reversed(self.train_metrics[-10:]):
                    if 'policy_loss' in m:
                        postfix['loss'] = f"{m['policy_loss']:.4f}"
                        break
            # Add imagination buffer size
            if len(self.imagined_buffer) > 0:
                postfix['imag'] = f"{len(self.imagined_buffer):,}"
            pbar.set_postfix(postfix)
        
        pbar.close()
        
        # Final save
        self.save_checkpoint("final")
        self.plot_progress()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best eval reward: {self.best_eval_reward:.1f}")
        print(f"Run directory: {self.run_dir}")
        print("=" * 60)
        
        self.env.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous World Model Training (v3)")
    parser.add_argument("--game", type=str, default="mspacman", help="Game to train on")
    parser.add_argument("--mode", type=str, default="continuous_wm", 
                       choices=["staged", "continuous_wm", "continuous_full"],
                       help="Training mode")
    parser.add_argument("--vqvae", type=str, default=None, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--wm", type=str, default=None, help="Path to World Model checkpoint")
    parser.add_argument("--policy", type=str, default=None, help="Path to Policy checkpoint")
    parser.add_argument("--steps", type=int, default=None, help="Total training steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create config
    config = get_default_config(args.game)
    config.mode = args.mode
    
    if args.steps:
        config.total_steps = args.steps
    
    # Create trainer
    trainer = ContinuousTrainer(
        config=config,
        vqvae_path=args.vqvae,
        wm_path=args.wm,
        policy_path=args.policy,
        device=args.device,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
