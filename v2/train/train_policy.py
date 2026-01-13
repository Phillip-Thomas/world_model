"""
Dyna-Style Policy Training
==========================
Train a policy using a mix of real and imagined (world model) experience.

Key concepts:
1. Collect real experience from environment
2. Generate imagined experience using the world model
3. Train policy (DQN) on mixed real + imagined data
4. Repeat

Training is organized into epochs for:
- Periodic checkpointing
- Training progress plots
- Better monitoring
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
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

# Limit OpenCV threads to prevent CPU thrashing (must be before cv2 import)
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Compute checkpoint root relative to this script (world_model/v2/train/train_policy.py)
# Checkpoints go to: world_model/checkpoints/v2/{game}
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORLD_MODEL_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # world_model/
CHECKPOINTS_ROOT = os.path.join(_WORLD_MODEL_ROOT, "checkpoints", "v2")


def resolve_policy_resume_path(
    from_run: str = None, 
    from_checkpoint: str = None,
    base_dir: str = None,
) -> tuple:
    """
    Resolve policy and world model checkpoint paths for resuming training.
    
    Args:
        from_run: Run directory name (e.g., "20260113_004626") or "latest"
        from_checkpoint: Checkpoint name: "latest", "best", or "epochXX"
        base_dir: Base checkpoint directory (default: CHECKPOINTS_ROOT/mspacman)
        
    Returns:
        (policy_path, wm_path) tuple, or (None, None) if from_run not specified
    """
    import glob
    import re
    
    if not from_run:
        return None, None
    
    if base_dir is None:
        base_dir = os.path.join(CHECKPOINTS_ROOT, "mspacman")
    
    runs_dir = os.path.join(base_dir, "policy_runs")
    
    # Find run directory
    if from_run == "latest":
        if not os.path.exists(runs_dir):
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found in: {runs_dir}")
        run_dirs.sort(reverse=True)  # Timestamp format sorts correctly
        run_name = run_dirs[0]
        print(f"  Using latest run: {run_name}")
    else:
        run_name = from_run
    
    run_path = os.path.join(runs_dir, run_name)
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # Resolve checkpoint name
    checkpoint_name = from_checkpoint or "latest"
    
    if checkpoint_name == "best":
        policy_path = os.path.join(run_path, "policy_best.pt")
        wm_path = os.path.join(run_path, "world_model_finetuned_best.pt")
    elif checkpoint_name == "latest" or checkpoint_name == "final":
        # Try final first, then best
        policy_path = os.path.join(run_path, "policy_final.pt")
        if not os.path.exists(policy_path):
            policy_path = os.path.join(run_path, "policy_best.pt")
        wm_path = os.path.join(run_path, "world_model_final.pt")
        if not os.path.exists(wm_path):
            wm_path = os.path.join(run_path, "world_model_finetuned_best.pt")
    elif checkpoint_name.startswith("epoch"):
        # Extract epoch number: epoch20 -> 20
        epoch_match = re.match(r'epoch(\d+)', checkpoint_name)
        if not epoch_match:
            raise ValueError(f"Invalid checkpoint format: {checkpoint_name}. Use 'best', 'latest', or 'epochXX'")
        epoch_num = epoch_match.group(1)
        policy_path = os.path.join(run_path, f"policy_epoch{epoch_num}.pt")
        wm_path = os.path.join(run_path, f"world_model_epoch{epoch_num}.pt")
    else:
        raise ValueError(f"Invalid checkpoint: {checkpoint_name}. Use 'best', 'latest', or 'epochXX'")
    
    # Validate policy exists
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
    
    # WM is optional (might not have fine-tuned WM)
    if not os.path.exists(wm_path):
        wm_path = None
    
    return policy_path, wm_path


from models.vqvae_hires import VQVAEHiRes
from models.temporal_world_model import TemporalVisualWorldModel
from agents.dqn_agent import DQNAgent, DQNConfig
from agents.replay_buffer import DualReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer

try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: gymnasium not installed. Run: pip install gymnasium[atari]")


@dataclass
class DynaConfig:
    """Configuration for Dyna-style training."""
    # Environment
    game: str = "ALE/MsPacman-v5"
    frame_skip: int = 4
    max_episode_steps: int = 10000
    n_envs: int = 6                   # Reduced for memory (was 8)
    
    # Training - epoch based
    n_epochs: int = 50                # Increased for longer training
    steps_per_epoch: int = 500       # Increased for more steps per epoch
    train_freq: int = 4               # Train every N environment steps
    batch_size: int = 1024             # Reduced for memory (was 256)
    gradient_steps: int = 2           # Single step to reduce memory
    warmup_steps: int = 5000          # 5k steps * 6 envs = 30k transitions (30% buffer fill)
    grad_clip: float = 10.0           # Gradient clipping for DQN stability
    
    # Dyna-specific (WM → Policy)
    imagined_rollout_len: int = 3     # K steps of imagination (start short)
    imagined_ratio: float = 0.05      # Fraction of imagined data (5% to start)
    imagined_update_freq: int = 200   # Less frequent to reduce overhead
    imagined_batch_size: int = 64     # Larger batch for efficiency
    imagined_buffer_ttl: int = 0      # 0 = disabled, use ring buffer (no hard clears)
    
    # Adaptive imagination (based on WM quality)
    adaptive_imagined_ratio: bool = True  # Enable adaptive ratio based on WM loss
    imagined_ratio_thresholds: tuple = (  # (wm_loss_threshold, imagined_ratio)
        (0.8, 0.02),   # WM loss < 0.8 → 2% imagined
        (0.6, 0.05),   # WM loss < 0.6 → 5% imagined
        (0.5, 0.10),   # WM loss < 0.5 → 10% imagined
        (0.4, 0.15),   # WM loss < 0.4 → 15% imagined
    )
    
    # Bidirectional: WM fine-tuning (Policy -> WM)
    wm_finetune: bool = True          # Enable WM fine-tuning for proper Dyna loop
    wm_update_freq: int = 200         # WM update frequency
    wm_updates_per_step: int = 1      # Number of WM gradient steps each time
    wm_batch_size: int = 32           # Larger batch for efficiency
    wm_lr: float = 1e-5               # Lower LR than initial WM training
    wm_grad_clip: float = 1.0         # Gradient clipping for WM
    wm_recent_k: int = 50000          # Recency window for WM sampling
    wm_recent_frac: float = 0.5       # 50% recent, 50% uniform
    
    # Prioritized replay
    use_prioritized: bool = True      # Use prioritized experience replay
    priority_alpha: float = 0.6       # Priority exponent
    priority_beta_start: float = 0.4  # IS correction, anneals to 1.0
    priority_beta_increment: float = 0.001  # Beta annealing rate
    
    # Buffers
    real_buffer_size: int = 250000    # 250k transitions (larger to reduce forgetting)
    imagined_buffer_size: int = 50000 # 50k imagined transitions
    
    # Evaluation - reduced since stochastic env provides variance
    eval_episodes: int = 10           # More episodes for stable eval metrics
    eval_random_noops: int = 30       # Random no-ops at eval reset for stochasticity
    
    # DQN hyperparameters
    learning_rate: float = 5e-5       # Lower LR to reduce catastrophic forgetting
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_epochs: int = 40    # Slower decay (was 20) for more exploration
    target_update_freq: int = 1000    # Steps between target network updates


@dataclass
class TrainingStats:
    """Track training statistics for plotting."""
    epoch: List[int] = field(default_factory=list)
    step: List[int] = field(default_factory=list)
    
    # Training metrics
    train_loss: List[float] = field(default_factory=list)
    avg_q_value: List[float] = field(default_factory=list)
    max_q_value: List[float] = field(default_factory=list)
    avg_td_error: List[float] = field(default_factory=list)
    
    # Episode metrics  
    episode_reward: List[float] = field(default_factory=list)
    episode_length: List[float] = field(default_factory=list)
    
    # Evaluation metrics
    eval_reward_mean: List[float] = field(default_factory=list)
    eval_reward_std: List[float] = field(default_factory=list)
    
    # Exploration
    epsilon: List[float] = field(default_factory=list)
    
    # Buffer stats
    buffer_real_size: List[int] = field(default_factory=list)
    buffer_imagined_size: List[int] = field(default_factory=list)
    
    # Dyna instrumentation
    n_real_updates: List[int] = field(default_factory=list)
    n_imagined_updates: List[int] = field(default_factory=list)
    actual_imagined_ratio: List[float] = field(default_factory=list)
    wm_loss: List[float] = field(default_factory=list)


class AtariEnvWrapper:
    """Wrapper for Atari environment that encodes frames to tokens."""
    
    def __init__(
        self,
        game: str,
        vqvae: VQVAEHiRes,
        device: str = 'cuda',
        frame_skip: int = 4,
        history_len: int = 4,
        target_size: Tuple[int, int] = (84, 64),
        max_episode_steps: int = 10000,
        random_noops: int = 0,  # Max random no-ops at reset for eval stochasticity
        stochastic: bool = False,  # Use stochastic action repeat (0.25) for evaluation
    ):
        self.vqvae = vqvae
        self.device = device
        self.history_len = history_len
        self.target_size = target_size
        self.random_noops = random_noops
        self.stochastic = stochastic
        
        # Get token dimensions from VQ-VAE
        with torch.no_grad():
            dummy = torch.zeros(1, 3, target_size[0], target_size[1], device=device)
            dummy_tokens = vqvae.encode(dummy)
            self.token_h, self.token_w = dummy_tokens.shape[1], dummy_tokens.shape[2]
            self.n_tokens = self.token_h * self.token_w
        
        # Create environment
        # Use stochastic=True for evaluation to get variance in episode outcomes
        # Keep deterministic (0.0) for training to reduce learning variance
        repeat_action_prob = 0.25 if stochastic else 0.0
        self.env = gym.make(
            game,
            frameskip=frame_skip,
            repeat_action_probability=repeat_action_prob,
            render_mode=None,
            max_episode_steps=max_episode_steps,
        )
        self.n_actions = self.env.action_space.n
        
        # State
        self.token_history = None
        self.episode_reward = 0
        self.episode_length = 0
    
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial token history."""
        obs, info = self.env.reset()
        
        # Random no-op starts for evaluation stochasticity
        if self.random_noops > 0:
            n_noops = np.random.randint(0, self.random_noops + 1)
            for _ in range(n_noops):
                obs, _, terminated, truncated, _ = self.env.step(0)  # NOOP action
                if terminated or truncated:
                    obs, info = self.env.reset()
        
        # Preprocess and encode
        frame = self._preprocess(obs)
        tokens = self._encode(frame)
        
        # Initialize history with copies
        self.token_history = tokens.unsqueeze(0).repeat(self.history_len, 1)
        
        self.episode_reward = 0
        self.episode_length = 0
        
        return self.token_history.clone()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """Take action and return next state, reward, done."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Preprocess and encode
        frame = self._preprocess(obs)
        tokens = self._encode(frame)
        
        # Update history
        self.token_history = torch.roll(self.token_history, shifts=-1, dims=0)
        self.token_history[-1] = tokens
        
        self.episode_reward += reward
        self.episode_length += 1
        
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        
        return self.token_history.clone(), reward, done, info
    
    def _preprocess(self, obs: np.ndarray) -> torch.Tensor:
        """Preprocess observation to (1, C, H, W) tensor."""
        import cv2
        
        # Resize
        frame = cv2.resize(obs, (self.target_size[1], self.target_size[0]))
        
        # To tensor and normalize
        frame = torch.from_numpy(frame).float().to(self.device)
        frame = frame.permute(2, 0, 1) / 127.5 - 1.0  # (C, H, W), [-1, 1]
        
        return frame.unsqueeze(0)  # (1, C, H, W)
    
    def _encode(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode frame to tokens."""
        with torch.no_grad():
            tokens = self.vqvae.encode(frame)  # (1, H, W)
            return tokens.flatten()  # (N,)
    
    def close(self):
        """Close environment."""
        self.env.close()


class VectorizedAtariEnv:
    """
    Vectorized environment that runs N environments in parallel.
    
    Benefits:
    - More diverse experience per step
    - Better GPU utilization (batch encoding)
    - Faster buffer filling
    """
    
    def __init__(
        self,
        game: str,
        vqvae: VQVAEHiRes,
        n_envs: int = 8,
        device: str = 'cuda',
        frame_skip: int = 4,
        history_len: int = 4,
        target_size: Tuple[int, int] = (84, 64),
        max_episode_steps: int = 10000,
    ):
        self.n_envs = n_envs
        self.vqvae = vqvae
        self.device = device
        self.history_len = history_len
        self.target_size = target_size
        
        # Get token dimensions from VQ-VAE
        with torch.no_grad():
            dummy = torch.zeros(1, 3, target_size[0], target_size[1], device=device)
            dummy_tokens = vqvae.encode(dummy)
            self.token_h, self.token_w = dummy_tokens.shape[1], dummy_tokens.shape[2]
            self.n_tokens = self.token_h * self.token_w
        
        # Create N environments
        self.envs = []
        for _ in range(n_envs):
            env = gym.make(
                game,
                frameskip=frame_skip,
                repeat_action_probability=0.0,
                render_mode=None,
                max_episode_steps=max_episode_steps,
            )
            self.envs.append(env)
        
        self.n_actions = self.envs[0].action_space.n
        
        # State for each env: (history_len, n_tokens) on GPU
        self.token_histories = None  # (n_envs, history_len, n_tokens)
        self.episode_rewards = np.zeros(n_envs)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)
        
        # Completed episode tracking
        self.completed_episodes = []
        
        # Life tracking for life loss penalty
        self.prev_lives = [None] * n_envs
    
    def reset(self) -> torch.Tensor:
        """Reset all environments and return initial states."""
        frames = []
        for env in self.envs:
            obs, _ = env.reset()
            frames.append(self._preprocess_single(obs))
        
        # Batch encode all frames
        frames_batch = torch.cat(frames, dim=0)  # (n_envs, C, H, W)
        tokens = self._encode_batch(frames_batch)  # (n_envs, n_tokens)
        
        # Initialize histories
        self.token_histories = tokens.unsqueeze(1).repeat(1, self.history_len, 1)
        self.episode_rewards = np.zeros(self.n_envs)
        self.episode_lengths = np.zeros(self.n_envs, dtype=np.int32)
        self.completed_episodes = []
        
        # Initialize lives tracking
        self.prev_lives = [None] * self.n_envs
        
        return self.token_histories.clone().cpu()  # (n_envs, history_len, n_tokens) on CPU for replay
    
    def step(self, actions: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments with given actions.
        
        Args:
            actions: (n_envs,) array of actions
            
        Returns:
            states: (n_envs, history_len, n_tokens)
            rewards: (n_envs,) 
            dones: (n_envs,) bool
            infos: list of dicts
        """
        frames = []
        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            frames.append(self._preprocess_single(obs))
            rewards[i] = reward
            dones[i] = done
            
            # Life loss penalty: -1 reward when losing a life (but not game over)
            current_lives = info.get('lives', None)
            if current_lives is not None and self.prev_lives[i] is not None:
                if current_lives < self.prev_lives[i] and not done:
                    rewards[i] -= 1.0  # Life loss penalty
            self.prev_lives[i] = current_lives
            
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            info['episode_reward'] = self.episode_rewards[i]
            info['episode_length'] = self.episode_lengths[i]
            
            # Handle episode end - auto-reset
            if done:
                self.completed_episodes.append({
                    'reward': self.episode_rewards[i],
                    'length': self.episode_lengths[i],
                })
                # Reset this environment
                obs, _ = env.reset()
                frames[-1] = self._preprocess_single(obs)
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0
                self.prev_lives[i] = None  # Reset lives tracking for new episode
            
            infos.append(info)
        
        # Batch encode all frames
        frames_batch = torch.cat(frames, dim=0)  # (n_envs, C, H, W)
        tokens = self._encode_batch(frames_batch)  # (n_envs, n_tokens)
        
        # Update histories
        self.token_histories = torch.roll(self.token_histories, shifts=-1, dims=1)
        self.token_histories[:, -1, :] = tokens
        
        # For done envs, reset their histories
        for i in range(self.n_envs):
            if dones[i]:
                self.token_histories[i] = tokens[i:i+1].repeat(self.history_len, 1)
        
        return self.token_histories.clone().cpu(), rewards, dones, infos  # CPU for replay storage
    
    def get_completed_episodes(self) -> List[dict]:
        """Get and clear completed episodes."""
        episodes = self.completed_episodes
        self.completed_episodes = []
        return episodes
    
    def _preprocess_single(self, obs: np.ndarray) -> torch.Tensor:
        """Preprocess single observation."""
        import cv2
        frame = cv2.resize(obs, (self.target_size[1], self.target_size[0]))
        frame = torch.from_numpy(frame).float().to(self.device)
        frame = frame.permute(2, 0, 1) / 127.5 - 1.0
        return frame.unsqueeze(0)  # (1, C, H, W)
    
    def _encode_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """Batch encode frames to tokens."""
        with torch.no_grad():
            tokens = self.vqvae.encode(frames)  # (B, H, W)
            return tokens.view(frames.shape[0], -1)  # (B, N)
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class ImaginedRolloutGenerator:
    """Generate imagined experience using the world model."""
    
    def __init__(
        self,
        world_model: TemporalVisualWorldModel,
        policy: DQNAgent,
        device: str = 'cuda',
    ):
        self.world_model = world_model
        self.policy = policy
        self.device = device
        self.world_model.eval()
    
    def generate(
        self,
        start_states: torch.Tensor,  # (B, T, N)
        rollout_len: int = 3,
        done_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate imagined transitions from start states."""
        with torch.no_grad():
            B = start_states.shape[0]
            
            all_states = []
            all_actions = []
            all_rewards = []
            all_next_states = []
            all_dones = []
            
            current_states = start_states.to(self.device)
            active = torch.ones(B, dtype=torch.bool, device=self.device)
            
            for k in range(rollout_len):
                if not active.any():
                    break
                
                # Select actions using policy (BATCHED - single forward pass!)
                # Only compute for active states, but we batch them together
                actions_np = self.policy.select_actions_batch(current_states)
                actions = torch.from_numpy(actions_np).long().to(self.device)
                
                # World model step
                next_tokens, reward_pred, done_pred = self.world_model.forward_with_heads(
                    current_states,
                    actions,
                    deterministic=True,
                )
                
                # Build next states
                next_states = torch.roll(current_states, shifts=-1, dims=1)
                next_states[:, -1, :] = next_tokens
                
                # Check termination (before storing, so we can use binary flags)
                terminated = done_pred > done_threshold
                done_flags = terminated.float()  # Binary 0/1 for DQN (not soft probabilities)
                
                # Store transitions (only for active states)
                for i in range(B):
                    if active[i]:
                        all_states.append(current_states[i])
                        all_actions.append(actions[i])
                        all_rewards.append(reward_pred[i])
                        all_next_states.append(next_states[i])
                        all_dones.append(done_flags[i])  # Binary done flag, not probability
                active = active & ~terminated
                
                # Update for next step
                current_states = next_states
            
            if len(all_states) == 0:
                T, N = start_states.shape[1], start_states.shape[2]
                return (
                    torch.zeros(0, T, N, device=self.device, dtype=torch.long),
                    torch.zeros(0, device=self.device, dtype=torch.long),
                    torch.zeros(0, device=self.device),
                    torch.zeros(0, T, N, device=self.device, dtype=torch.long),
                    torch.zeros(0, device=self.device),
                )
            
            return (
                torch.stack(all_states),
                torch.stack(all_actions),
                torch.stack(all_rewards),
                torch.stack(all_next_states),
                torch.stack(all_dones),
            )


def wm_update_step(
    world_model: TemporalVisualWorldModel,
    wm_optimizer: torch.optim.Optimizer,
    buffer: PrioritizedReplayBuffer,
    config: DynaConfig,
    device: str,
) -> float:
    """
    Single world model fine-tuning step on REAL transitions only.
    
    Uses recency-weighted sampling to emphasize recent on-policy data
    while maintaining coverage of older states.
    
    Args:
        world_model: The world model to fine-tune
        wm_optimizer: Optimizer for world model
        buffer: Replay buffer with real transitions
        config: Training configuration
        device: Device to use
        
    Returns:
        wm_loss: The world model loss value
    """
    world_model.train()
    
    # Sample with recency mix (50% recent, 50% uniform)
    batch = buffer.sample_with_recency(
        batch_size=config.wm_batch_size,
        recent_k=config.wm_recent_k,
        recent_frac=config.wm_recent_frac,
    )
    batch = batch.to(device)
    
    # Extract inputs for WM training
    # states: (B, T, N) - full history
    # next_states: (B, T, N) - we want the last frame as target
    frame_history = batch.states  # (B, T=4, N=336)
    actions = batch.actions       # (B,)
    target_tokens = batch.next_states[:, -1, :]  # (B, N) - newest frame only
    target_rewards = batch.rewards  # (B,)
    target_dones = batch.dones      # (B,)
    
    # Forward pass with loss computation
    _, wm_loss, aux = world_model.forward(
        frame_history,
        actions,
        target_tokens,
        target_rewards,
        target_dones,
    )
    
    # Backward pass
    wm_optimizer.zero_grad(set_to_none=True)
    wm_loss.backward()
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), config.wm_grad_clip)
    wm_optimizer.step()
    
    world_model.eval()
    
    return wm_loss.item()


def compute_ema(values: list, alpha: float = 0.3) -> list:
    """Compute exponential moving average of a list of values."""
    if not values:
        return []
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def plot_training_progress(stats: TrainingStats, save_path: str):
    """Generate training progress plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Training Loss
    ax = axes[0, 0]
    if stats.train_loss:
        ax.plot(stats.step, stats.train_loss, 'b-', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
    
    # 2. Episode Rewards
    ax = axes[0, 1]
    if stats.episode_reward:
        ax.plot(stats.step, stats.episode_reward, 'g-', alpha=0.4, label='Train')
        if stats.eval_reward_mean:
            eval_steps = [stats.step[i] for i in range(len(stats.step)) 
                         if i < len(stats.eval_reward_mean) and stats.eval_reward_mean[i] is not None]
            eval_rewards = [r for r in stats.eval_reward_mean if r is not None]
            if eval_steps and eval_rewards and len(eval_steps) == len(eval_rewards):
                # Plot raw eval points
                ax.plot(eval_steps[:len(eval_rewards)], eval_rewards, 'ro', 
                       markersize=6, alpha=0.5, label='Eval')
                # Plot EMA trend line
                if len(eval_rewards) >= 2:
                    eval_ema = compute_ema(eval_rewards, alpha=0.3)
                    ax.plot(eval_steps[:len(eval_ema)], eval_ema, 'r-', 
                           linewidth=2.5, label='Eval EMA')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Epsilon (exploration)
    ax = axes[0, 2]
    if stats.epsilon:
        ax.plot(stats.step, stats.epsilon, 'purple', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate')
        ax.grid(True, alpha=0.3)
    
    # 4. Q-Values
    ax = axes[1, 0]
    if stats.avg_q_value:
        ax.plot(stats.step, stats.avg_q_value, 'orange', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Avg Q')
        ax.set_title('Average Q-Value')
        ax.grid(True, alpha=0.3)
    
    # 5. Episode Length
    ax = axes[1, 1]
    if stats.episode_length:
        ax.plot(stats.step, stats.episode_length, 'cyan', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Length')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
    
    # 6. Buffer Sizes
    ax = axes[1, 2]
    if stats.buffer_real_size:
        ax.plot(stats.step, stats.buffer_real_size, 'b-', label='Real', alpha=0.7)
        ax.plot(stats.step, stats.buffer_imagined_size, 'r-', label='Imagined', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Size')
        ax.set_title('Buffer Sizes')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_training_stats(stats: TrainingStats, save_path: str):
    """Save training stats to text file with per-epoch breakdown."""
    with open(save_path, 'w') as f:
        f.write("Policy Training Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary
        if stats.eval_reward_mean:
            valid_evals = [r for r in stats.eval_reward_mean if r is not None]
            if valid_evals:
                f.write(f"Best Eval Reward: {max(valid_evals):.1f}\n")
                f.write(f"Final Eval Reward: {valid_evals[-1]:.1f}\n")
        
        if stats.episode_reward:
            f.write(f"Final Train Reward (avg last 10): {np.mean(stats.episode_reward[-10:]):.1f}\n")
        
        if stats.train_loss:
            f.write(f"Final Loss: {stats.train_loss[-1]:.4f}\n")
        
        f.write(f"\nTotal Steps: {stats.step[-1] if stats.step else 0}\n")
        f.write(f"Total Epochs: {stats.epoch[-1] if stats.epoch else 0}\n")
        
        # Per-epoch breakdown
        f.write("\n" + "=" * 50 + "\n")
        f.write("Per-Epoch Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Epoch':>6} {'Step':>10} {'Loss':>10} {'Epsilon':>8} {'Eval Mean':>10} {'Eval Std':>10}\n")
        f.write("-" * 60 + "\n")
        
        n_epochs = len(stats.epoch) if stats.epoch else 0
        for i in range(n_epochs):
            epoch = stats.epoch[i] if i < len(stats.epoch) else 0
            step = stats.step[i] if i < len(stats.step) else 0
            loss = stats.train_loss[i] if i < len(stats.train_loss) else 0
            eps = stats.epsilon[i] if i < len(stats.epsilon) else 0
            eval_mean = stats.eval_reward_mean[i] if i < len(stats.eval_reward_mean) else 0
            eval_std = stats.eval_reward_std[i] if i < len(stats.eval_reward_std) else 0
            
            f.write(f"{epoch:>6} {step:>10} {loss:>10.4f} {eps:>8.3f} {eval_mean:>10.1f} {eval_std:>10.1f}\n")


def evaluate_policy(
    env: AtariEnvWrapper,
    agent: DQNAgent,
    n_episodes: int = 5,
) -> Tuple[float, float, List[float]]:
    """Evaluate policy over multiple episodes."""
    rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Greedy evaluation
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    agent.epsilon = old_epsilon
    return np.mean(rewards), np.std(rewards), rewards


def train_dyna(
    base_dir: str = None,
    config: Optional[DynaConfig] = None,
    device: str = None,
    resume_policy: str = None,
    resume_wm: str = None,
    epsilon_override: float = None,
    vqvae_path: str = None,
    wm_path: str = None,
):
    """Main Dyna-style training loop with epochs."""
    config = config or DynaConfig()
    
    # Use canonical checkpoint directory (relative to this script)
    if base_dir is None:
        base_dir = os.path.join(CHECKPOINTS_ROOT, "mspacman")
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Dyna-Style Policy Training (Epoch-Based)")
    print("=" * 60)
    print(f"Game: {config.game}")
    print(f"Device: {device}")
    print(f"Epochs: {config.n_epochs}")
    print(f"Steps/epoch: {config.steps_per_epoch}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient steps: {config.gradient_steps}")
    if resume_policy:
        print(f"Resuming policy from: {resume_policy}")
    if resume_wm:
        print(f"Resuming WM from: {resume_wm}")
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{base_dir}/policy_runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save config
    with open(f"{run_dir}/config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Load VQ-VAE
    print("\nLoading VQ-VAE...")
    if vqvae_path is None:
        vqvae_path = f"{base_dir}/vqvae_hires.pt"
    vqvae_ckpt = torch.load(vqvae_path, map_location=device, weights_only=False)
    vqvae = VQVAEHiRes(
        in_channels=3,
        latent_channels=vqvae_ckpt.get('latent_channels', 256),
        n_embeddings=vqvae_ckpt.get('n_embeddings', 512),
    ).to(device)
    vqvae.load_state_dict(vqvae_ckpt['model_state_dict'])
    vqvae.eval()
    print(f"  Loaded from {vqvae_path}")
    
    # Load world model
    print("\nLoading World Model...")
    # Prioritize resume_wm if wm_path not provided
    if wm_path is None and resume_wm is not None:
        wm_path = resume_wm
        resume_wm = None  # Don't load twice
    elif wm_path is None:
        wm_path = f"{base_dir}/world_model_best.pt"
    
    wm_ckpt = torch.load(wm_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint format and state_dict only format
    if 'model_state_dict' in wm_ckpt:
        # Full checkpoint format
        state_dict = wm_ckpt['model_state_dict']
        n_vocab = wm_ckpt['n_vocab']
        n_actions = wm_ckpt['n_actions']
        n_layers = wm_ckpt.get('n_layers', 10)
        token_h = wm_ckpt['token_h']
        token_w = wm_ckpt['token_w']
    else:
        # State dict only format - infer architecture from weights
        state_dict = wm_ckpt
        n_actions = state_dict['action_embed.weight'].shape[0]
        n_vocab = state_dict['token_embed.weight'].shape[0]
        n_layers = sum(1 for k in state_dict.keys() if k.startswith('transformer.') and k.endswith('.self_attn.in_proj_weight'))
        # Infer token grid from spatial position embeddings
        if 'spatial_pos.row_embed.weight' in state_dict:
            token_h = state_dict['spatial_pos.row_embed.weight'].shape[0]
            token_w = state_dict['spatial_pos.col_embed.weight'].shape[0]
        else:
            token_h, token_w = 21, 16  # Default hi-res grid
        print(f"  (Inferred from state_dict: {n_actions} actions, {n_vocab} vocab, {n_layers} layers, {token_h}x{token_w})")
    
    world_model = TemporalVisualWorldModel(
        n_vocab=n_vocab,
        n_actions=n_actions,
        d_model=256,
        n_heads=8,
        n_layers=n_layers,
        token_h=token_h,
        token_w=token_w,
        max_history=4,
    ).to(device)
    world_model.load_state_dict(state_dict)
    print(f"  Loaded WM from {wm_path}")
    
    # Optionally load fine-tuned WM checkpoint on top
    if resume_wm:
        print(f"  Loading fine-tuned WM from {resume_wm}...")
        wm_finetune_state = torch.load(resume_wm, map_location=device, weights_only=True)
        world_model.load_state_dict(wm_finetune_state)
        print(f"  Loaded fine-tuned WM!")
    
    world_model.eval()
    
    n_tokens = token_h * token_w
    
    # Create vectorized environment (parallel envs for more GPU usage)
    print("\nCreating vectorized environment...")
    vec_env = VectorizedAtariEnv(
        game=config.game,
        vqvae=vqvae,
        n_envs=config.n_envs,
        device=device,
        frame_skip=config.frame_skip,
        history_len=4,
        max_episode_steps=config.max_episode_steps,
    )
    print(f"  Parallel envs: {config.n_envs}")
    print(f"  Actions: {n_actions}")
    print(f"  Token grid: {vec_env.token_h}x{vec_env.token_w} = {n_tokens}")
    
    # Single env for evaluation (with stochastic action repeat + random no-ops)
    eval_env = AtariEnvWrapper(
        game=config.game,
        vqvae=vqvae,
        device=device,
        frame_skip=config.frame_skip,
        history_len=4,
        max_episode_steps=config.max_episode_steps,
        random_noops=config.eval_random_noops,
        stochastic=True,  # Use repeat_action_probability=0.25 for realistic eval variance
    )
    
    # Calculate epsilon decay
    total_steps = config.n_epochs * config.steps_per_epoch
    decay_steps = config.epsilon_decay_epochs * config.steps_per_epoch
    
    # Create DQN agent
    print("\nCreating DQN agent...")
    dqn_config = DQNConfig(
        d_embed=256,             # Larger embeddings for more GPU usage
        hidden_dim=1024,         # Much larger network
        n_hidden=4,              # More layers
        use_dueling=True,
        use_double_dqn=True,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_steps=decay_steps,
    )
    agent = DQNAgent(
        n_vocab=n_vocab,
        n_actions=n_actions,
        history_len=4,
        n_tokens=n_tokens,
        config=dqn_config,
        device=device,
    )
    print(f"  Policy parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    
    # Optionally load policy checkpoint to resume training
    if resume_policy:
        print(f"  Loading policy from {resume_policy}...")
        agent.load(resume_policy)
        print(f"  Resumed! Epsilon: {agent.epsilon:.3f}, Steps: {agent.train_steps:,}")
    
    # Override epsilon if specified (for more exploration with new WM)
    if epsilon_override is not None:
        old_eps = agent.epsilon
        agent.epsilon = epsilon_override
        print(f"  Epsilon overridden: {old_eps:.3f} -> {epsilon_override:.3f}")
    
    # Create buffers - use PrioritizedReplayBuffer for bidirectional training
    if config.use_prioritized:
        real_buffer = PrioritizedReplayBuffer(
            capacity=config.real_buffer_size,
            history_len=4,
            n_tokens=n_tokens,
            alpha=config.priority_alpha,
            beta=config.priority_beta_start,
            beta_increment=config.priority_beta_increment,
        )
        print(f"  Using PrioritizedReplayBuffer (alpha={config.priority_alpha})")
    else:
        real_buffer = ReplayBuffer(
            capacity=config.real_buffer_size,
            history_len=4,
            n_tokens=n_tokens,
        )
        print(f"  Using standard ReplayBuffer")
    
    # Separate buffer for imagined transitions (simpler, no priorities needed)
    imagined_buffer = ReplayBuffer(
        capacity=config.imagined_buffer_size,
        history_len=4,
        n_tokens=n_tokens,
    )
    
    # Create imagined rollout generator (if using imagination)
    imagination = None
    if config.imagined_ratio > 0:
        imagination = ImaginedRolloutGenerator(world_model, agent, device)
        print(f"  Imagination enabled: ratio={config.imagined_ratio}, rollout_len={config.imagined_rollout_len}")
    
    # Create WM optimizer for online fine-tuning (bidirectional: Policy → WM)
    wm_optimizer = None
    if config.wm_finetune:
        wm_optimizer = torch.optim.AdamW(
            world_model.parameters(),
            lr=config.wm_lr,
            weight_decay=0.0,
        )
        print(f"  WM fine-tuning enabled: lr={config.wm_lr}, update_freq={config.wm_update_freq}")
    
    # Training state
    stats = TrainingStats()
    states = vec_env.reset()  # (n_envs, history_len, n_tokens)
    episode_rewards = []
    episode_lengths = []
    train_losses = []
    q_values = []
    best_eval_reward = float('-inf')
    global_step = 0
    
    # Track WM losses for bidirectional training
    wm_losses = []
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    # Warmup phase - collect with random actions from all envs
    print(f"\nWarmup: collecting {config.warmup_steps} random steps ({config.n_envs} envs)...")
    warmup_pbar = tqdm(range(config.warmup_steps), desc="Warmup")
    for _ in warmup_pbar:
        # Random actions for all envs
        actions = np.random.randint(0, n_actions, size=config.n_envs)
        next_states, rewards, dones, infos = vec_env.step(actions)
        
        # Store transitions for each env in REAL buffer
        for i in range(config.n_envs):
            real_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        states = next_states
        
        # Collect completed episodes
        completed = vec_env.get_completed_episodes()
        for ep in completed:
            episode_rewards.append(ep['reward'])
            episode_lengths.append(ep['length'])
    
    print(f"  Warmup complete. Buffer size: {len(real_buffer)}")
    print(f"  Episodes: {len(episode_rewards)}, Avg reward: {np.mean(episode_rewards) if episode_rewards else 0:.1f}")
    
    # Initialize adaptive imagined ratio
    current_imagined_ratio = 0.0 if config.adaptive_imagined_ratio else config.imagined_ratio
    if config.adaptive_imagined_ratio:
        print(f"\n  Adaptive imagination enabled: ratio starts at 0%, scales with WM quality")
    
    # Epoch-based training
    for epoch in range(1, config.n_epochs + 1):
        epoch_start = time.time()
        epoch_losses = []
        epoch_q_values = []
        epoch_max_q_values = []
        epoch_td_errors = []
        epoch_rewards = []
        
        # Dyna instrumentation counters
        epoch_real_updates = 0
        epoch_imagined_updates = 0
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.n_epochs}")
        print('='*60)
        
        pbar = tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch}")
        
        epoch_wm_losses = []
        
        for step in pbar:
            global_step += 1
            
            # ===== 1. COLLECT REAL EXPERIENCE =====
            # Select actions for all envs (BATCHED - single forward pass!)
            # states is already (n_envs, T, N) tensor from vec_env
            actions = agent.select_actions_batch(states)
            
            # Environment step (all envs)
            next_states, rewards, dones, infos = vec_env.step(actions)
            
            # Store transitions for each env in REAL buffer
            for i in range(config.n_envs):
                real_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
            states = next_states
            
            # Collect completed episodes
            completed = vec_env.get_completed_episodes()
            for ep in completed:
                episode_rewards.append(ep['reward'])
                episode_lengths.append(ep['length'])
                epoch_rewards.append(ep['reward'])
            
            # ===== 2. WORLD MODEL FINE-TUNING (Policy → WM) =====
            if (wm_optimizer is not None and 
                global_step % config.wm_update_freq == 0 and 
                real_buffer.is_ready(config.wm_batch_size)):
                for _ in range(config.wm_updates_per_step):
                    wm_loss = wm_update_step(
                        world_model, wm_optimizer, real_buffer, config, device
                    )
                    wm_losses.append(wm_loss)
                    epoch_wm_losses.append(wm_loss)
            
            # ===== 3. GENERATE IMAGINED EXPERIENCE (WM → Policy) =====
            # Offset by 50 to stagger with WM updates (avoids double-pause at step 100, 200, etc.)
            if (imagination is not None and 
                (global_step + 50) % config.imagined_update_freq == 0 and
                len(real_buffer) >= config.imagined_batch_size):
                start_states = real_buffer.sample_states(config.imagined_batch_size)
                start_states = start_states.to(device)
                
                im_states, im_actions, im_rewards, im_next, im_dones = imagination.generate(
                    start_states,
                    rollout_len=config.imagined_rollout_len,
                )
                
                if len(im_states) > 0:
                    imagined_buffer.add_batch(im_states, im_actions, im_rewards, im_next, im_dones)
            
            # ===== 4. TTL: CLEAR STALE IMAGINED DATA =====
            # TTL=0 means disabled (ring buffer overwrites naturally)
            if config.imagined_buffer_ttl > 0 and global_step % config.imagined_buffer_ttl == 0 and len(imagined_buffer) > 0:
                imagined_buffer.clear()
            
            # ===== 5. TRAIN POLICY (Prioritized + Mixed) =====
            if step % config.train_freq == 0 and real_buffer.is_ready(config.batch_size):
                for grad_step in range(config.gradient_steps):
                    # Decide: train on real (prioritized) or imagined (uniform)
                    # Use adaptive ratio if enabled, otherwise static config ratio
                    effective_ratio = current_imagined_ratio if config.adaptive_imagined_ratio else config.imagined_ratio
                    use_imagined = (
                        effective_ratio > 0 and
                        len(imagined_buffer) >= config.batch_size and
                        np.random.random() < effective_ratio
                    )
                    
                    if use_imagined:
                        # Train on imagined data (uniform sampling, no priorities)
                        batch = imagined_buffer.sample(config.batch_size)
                        loss = agent.train_step(batch)
                        epoch_losses.append(loss)
                        epoch_imagined_updates += 1
                    else:
                        # Train on real data with prioritized sampling
                        if config.use_prioritized:
                            batch, weights, indices = real_buffer.sample(config.batch_size)
                            loss, td_errors = agent.train_step_prioritized(batch, weights)
                            # Update priorities with |TD-error| + epsilon
                            real_buffer.update_priorities(indices, td_errors)
                            epoch_losses.append(loss)
                            epoch_td_errors.append(td_errors.abs().mean().item())
                        else:
                            batch = real_buffer.sample(config.batch_size)
                            loss = agent.train_step(batch)
                            epoch_losses.append(loss)
                        epoch_real_updates += 1
                    
                    # Track Q-values occasionally
                    if len(epoch_losses) % 100 == 0:
                        with torch.no_grad():
                            q = agent.policy_net(batch.states.to(device))
                            epoch_q_values.append(q.mean().item())
                            epoch_max_q_values.append(q.max().item())
            
            # Update progress bar
            if step % 100 == 0:
                avg_reward = np.mean(epoch_rewards[-10:]) if epoch_rewards else 0
                avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else 0
                avg_wm_loss = np.mean(epoch_wm_losses[-50:]) if epoch_wm_losses else 0
                pbar.set_postfix({
                    'reward': f'{avg_reward:.1f}',
                    'loss': f'{avg_loss:.4f}',
                    'wm': f'{avg_wm_loss:.4f}',
                    'eps': f'{agent.epsilon:.3f}',
                    'buf': f'{len(real_buffer)//1000}k+{len(imagined_buffer)//1000}k',
                })
        
        # End of epoch - evaluate
        print("\n  Evaluating...")
        eval_mean, eval_std, eval_rewards = evaluate_policy(eval_env, agent, config.eval_episodes)
        
        # Update stats
        stats.epoch.append(epoch)
        stats.step.append(global_step)
        stats.train_loss.append(np.mean(epoch_losses) if epoch_losses else 0)
        stats.avg_q_value.append(np.mean(epoch_q_values) if epoch_q_values else 0)
        stats.max_q_value.append(np.mean(epoch_max_q_values) if epoch_max_q_values else 0)
        stats.avg_td_error.append(np.mean(epoch_td_errors) if epoch_td_errors else 0)
        stats.episode_reward.append(np.mean(epoch_rewards) if epoch_rewards else 0)
        stats.episode_length.append(np.mean(episode_lengths[-len(epoch_rewards):]) if epoch_rewards else 0)
        stats.eval_reward_mean.append(eval_mean)
        stats.eval_reward_std.append(eval_std)
        stats.epsilon.append(agent.epsilon)
        stats.buffer_real_size.append(len(real_buffer))
        stats.buffer_imagined_size.append(len(imagined_buffer))
        
        # Dyna instrumentation
        stats.n_real_updates.append(epoch_real_updates)
        stats.n_imagined_updates.append(epoch_imagined_updates)
        total_updates = epoch_real_updates + epoch_imagined_updates
        actual_ratio = epoch_imagined_updates / total_updates if total_updates > 0 else 0
        stats.actual_imagined_ratio.append(actual_ratio)
        stats.wm_loss.append(np.mean(epoch_wm_losses) if epoch_wm_losses else 0)
        
        epoch_time = time.time() - epoch_start
        avg_wm_loss = np.mean(epoch_wm_losses) if epoch_wm_losses else 0
        
        # Update adaptive imagined ratio based on WM loss
        if config.adaptive_imagined_ratio and avg_wm_loss > 0:
            old_ratio = current_imagined_ratio
            current_imagined_ratio = 0.0  # Default: no imagination if WM is poor
            for threshold, ratio in config.imagined_ratio_thresholds:
                if avg_wm_loss < threshold:
                    current_imagined_ratio = ratio
            if current_imagined_ratio != old_ratio:
                print(f"    Adaptive ratio: {old_ratio*100:.1f}% -> {current_imagined_ratio*100:.1f}% (WM loss={avg_wm_loss:.4f})")
        
        # Print epoch summary
        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train reward (avg): {stats.episode_reward[-1]:.1f}")
        print(f"    Eval reward: {eval_mean:.1f} +/- {eval_std:.1f}")
        print(f"    Policy Loss: {stats.train_loss[-1]:.4f} | Avg TD: {stats.avg_td_error[-1]:.4f}")
        print(f"    Q-values: mean={stats.avg_q_value[-1]:.2f}, max={stats.max_q_value[-1]:.2f}")
        if config.wm_finetune:
            print(f"    WM Loss: {avg_wm_loss:.4f}")
        target_ratio = current_imagined_ratio if config.adaptive_imagined_ratio else config.imagined_ratio
        print(f"    Updates: {epoch_real_updates} real, {epoch_imagined_updates} imag ({actual_ratio*100:.1f}% actual, {target_ratio*100:.1f}% target)")
        print(f"    Epsilon: {agent.epsilon:.3f}")
        print(f"    Buffer: {len(real_buffer):,} real, {len(imagined_buffer):,} imagined")
        print(f"    Time: {epoch_time:.1f}s")
        
        # Save checkpoint (with error handling for disk issues)
        is_best = eval_mean > best_eval_reward
        try:
            if is_best:
                best_eval_reward = eval_mean
                agent.save(f"{run_dir}/policy_best.pt")
                print(f"    New best! Saved to policy_best.pt")
            
            # Save World Model if fine-tuning is enabled (only best, skip epoch saves to save space)
            if config.wm_finetune:
                # Track best WM loss
                if not hasattr(train_dyna, 'best_wm_loss'):
                    train_dyna.best_wm_loss = float('inf')
                avg_wm_loss = np.mean(wm_losses[-100:]) if wm_losses else float('inf')
                if avg_wm_loss < train_dyna.best_wm_loss:
                    train_dyna.best_wm_loss = avg_wm_loss
                    torch.save(world_model.state_dict(), f"{run_dir}/world_model_finetuned_best.pt")
                    print(f"    New best WM! Loss: {avg_wm_loss:.4f}")
            
            # Always save latest checkpoint (overwrite each epoch)
            agent.save(f"{run_dir}/policy_latest.pt")
            if config.wm_finetune:
                torch.save(world_model.state_dict(), f"{run_dir}/world_model_latest.pt")
        except (RuntimeError, OSError) as e:
            print(f"    WARNING: Save failed (disk full?): {e}")
        
        # Update plots and stats
        plot_training_progress(stats, f"{run_dir}/training_progress.png")
        save_training_stats(stats, f"{run_dir}/training_stats.txt")
    
    # Final save (with error handling)
    try:
        agent.save(f"{run_dir}/policy_final.pt")
        if config.wm_finetune:
            torch.save(world_model.state_dict(), f"{run_dir}/world_model_final.pt")
    except (RuntimeError, OSError) as e:
        print(f"WARNING: Final save failed: {e}")
    vec_env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best eval reward: {best_eval_reward:.1f}")
    print(f"  Total steps: {global_step:,}")
    print(f"  Checkpoints: {run_dir}")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dyna-style policy training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh training (no resume):
  python train_policy.py --vqvae path/to/vqvae.pt --wm path/to/world_model.pt

  # Resume from latest run, latest checkpoint:
  python train_policy.py --run latest

  # Resume from specific run, best checkpoint:
  python train_policy.py --run 20260113_004626 --checkpoint best

  # Resume from specific run, specific epoch:
  python train_policy.py --run 20260113_004626 --checkpoint epoch50

  # Resume with explicit paths (legacy):
  python train_policy.py --resume_policy path/to/policy.pt --resume_wm path/to/wm.pt
        """
    )
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Directory with VQ-VAE and world model (default: world_model/checkpoints/v2/mspacman)")
    
    # Resume from run directory (new system)
    parser.add_argument("--run", type=str, default=None,
                        help="Run to resume from: timestamp (e.g., '20260113_004626') or 'latest'")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint within run: 'latest', 'best', or 'epochXX' (default: latest)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: from DynaConfig)")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="Environment steps per epoch (default: from DynaConfig)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (default: from DynaConfig)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    # Legacy resume (explicit paths)
    parser.add_argument("--resume_policy", type=str, default=None,
                        help="[Legacy] Explicit path to policy checkpoint")
    parser.add_argument("--resume_wm", type=str, default=None,
                        help="[Legacy] Explicit path to fine-tuned WM checkpoint")
    
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override epsilon value (e.g., 0.5 for more exploration)")
    parser.add_argument("--no-imagination", action="store_true",
                        help="Disable imagination (pure real experience, no world model rollouts)")
    parser.add_argument("--no-wm-finetune", action="store_true",
                        help="Disable world model fine-tuning")
    parser.add_argument("--vqvae", type=str, default=None,
                        help="Path to VQ-VAE checkpoint (default: base_dir/vqvae_hires.pt)")
    parser.add_argument("--wm", type=str, default=None,
                        help="Path to base world model checkpoint (default: base_dir/world_model_best.pt)")
    
    args = parser.parse_args()
    
    # Resolve run/checkpoint to paths (new system takes precedence)
    resume_policy = args.resume_policy
    resume_wm = args.resume_wm
    
    if args.run:
        print(f"Resolving checkpoint from run: {args.run}, checkpoint: {args.checkpoint or 'latest'}")
        policy_path, wm_path = resolve_policy_resume_path(
            from_run=args.run,
            from_checkpoint=args.checkpoint,
            base_dir=args.base_dir,
        )
        resume_policy = policy_path
        if wm_path:
            resume_wm = wm_path
        print(f"  Policy: {resume_policy}")
        print(f"  WM: {resume_wm or '(none)'}")
    
    # Start with defaults from DynaConfig, only override if explicitly provided
    config = DynaConfig()
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.steps_per_epoch is not None:
        config.steps_per_epoch = args.steps_per_epoch
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    # Disable imagination if requested
    if getattr(args, 'no_imagination', False):
        config.imagined_ratio = 0.0
        config.adaptive_imagined_ratio = False
        print("Imagination DISABLED (pure real experience)")
    
    # Disable WM fine-tuning if requested  
    if getattr(args, 'no_wm_finetune', False):
        config.wm_finetune = False
        print("World model fine-tuning DISABLED")
    
    train_dyna(
        base_dir=args.base_dir, 
        config=config, 
        device=args.device,
        resume_policy=resume_policy,
        resume_wm=resume_wm,
        epsilon_override=args.epsilon,
        vqvae_path=args.vqvae,
        wm_path=args.wm,
    )
