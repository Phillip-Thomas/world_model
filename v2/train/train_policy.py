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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.vqvae_hires import VQVAEHiRes
from models.temporal_world_model import TemporalVisualWorldModel
from agents.dqn_agent import DQNAgent, DQNConfig
from agents.replay_buffer import DualReplayBuffer

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
    n_envs: int = 8                   # Parallel environments for more GPU usage
    
    # Training - epoch based
    n_epochs: int = 20                # Number of training epochs
    steps_per_epoch: int = 10000      # Environment steps per epoch (per env)
    train_freq: int = 4               # Train every N environment steps
    batch_size: int = 1024            # Large batch for GPU utilization
    gradient_steps: int = 2           # Gradient updates per train step
    warmup_steps: int = 2000          # Fewer warmup steps (n_envs fill faster)
    
    # Dyna-specific
    imagined_rollout_len: int = 3     # K steps of imagination
    imagined_ratio: float = 0.0       # Fraction of imagined data (0 = real only)
    imagined_update_freq: int = 500   # Generate imagined data every N steps
    imagined_batch_size: int = 64     # States to start rollouts from
    
    # Buffers
    real_buffer_size: int = 200000    # Larger buffer
    imagined_buffer_size: int = 100000
    
    # Evaluation
    eval_episodes: int = 5            # Episodes per evaluation
    
    # DQN hyperparameters
    learning_rate: float = 3e-4       # Higher LR for faster learning
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_epochs: int = 10    # Decay over N epochs
    target_update_freq: int = 1000    # Steps between target network updates


@dataclass
class TrainingStats:
    """Track training statistics for plotting."""
    epoch: List[int] = field(default_factory=list)
    step: List[int] = field(default_factory=list)
    
    # Training metrics
    train_loss: List[float] = field(default_factory=list)
    avg_q_value: List[float] = field(default_factory=list)
    
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
    ):
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
        
        # Create environment
        self.env = gym.make(
            game,
            frameskip=frame_skip,
            repeat_action_probability=0.0,
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
        
        return self.token_histories.clone()  # (n_envs, history_len, n_tokens)
    
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
        
        return self.token_histories.clone(), rewards, dones, infos
    
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
                
                # Select actions using policy
                actions = torch.zeros(B, dtype=torch.long, device=self.device)
                for i in range(B):
                    if active[i]:
                        actions[i] = self.policy.select_action(current_states[i])
                
                # World model step
                next_tokens, reward_pred, done_pred = self.world_model.forward_with_heads(
                    current_states,
                    actions,
                    deterministic=True,
                )
                
                # Build next states
                next_states = torch.roll(current_states, shifts=-1, dims=1)
                next_states[:, -1, :] = next_tokens
                
                # Store transitions (only for active states)
                for i in range(B):
                    if active[i]:
                        all_states.append(current_states[i])
                        all_actions.append(actions[i])
                        all_rewards.append(reward_pred[i])
                        all_next_states.append(next_states[i])
                        all_dones.append(done_pred[i])
                
                # Check termination
                terminated = done_pred > done_threshold
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
        ax.plot(stats.step, stats.episode_reward, 'g-', alpha=0.7, label='Train')
        if stats.eval_reward_mean:
            eval_steps = [stats.step[i] for i in range(len(stats.step)) 
                         if i < len(stats.eval_reward_mean) and stats.eval_reward_mean[i] is not None]
            eval_rewards = [r for r in stats.eval_reward_mean if r is not None]
            if eval_steps and eval_rewards and len(eval_steps) == len(eval_rewards):
                ax.plot(eval_steps[:len(eval_rewards)], eval_rewards, 'r-o', 
                       linewidth=2, markersize=8, label='Eval')
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
    base_dir: str = "checkpoints/v2/mspacman",
    config: Optional[DynaConfig] = None,
    device: str = None,
):
    """Main Dyna-style training loop with epochs."""
    config = config or DynaConfig()
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
    wm_path = f"{base_dir}/world_model_best.pt"
    wm_ckpt = torch.load(wm_path, map_location=device, weights_only=False)
    
    world_model = TemporalVisualWorldModel(
        n_vocab=wm_ckpt['n_vocab'],
        n_actions=wm_ckpt['n_actions'],
        d_model=256,
        n_heads=8,
        n_layers=wm_ckpt.get('n_layers', 10),
        token_h=wm_ckpt['token_h'],
        token_w=wm_ckpt['token_w'],
        max_history=4,
    ).to(device)
    world_model.load_state_dict(wm_ckpt['model_state_dict'])
    world_model.eval()
    print(f"  Loaded from {wm_path}")
    
    n_actions = wm_ckpt['n_actions']
    n_tokens = wm_ckpt['token_h'] * wm_ckpt['token_w']
    
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
    
    # Single env for evaluation
    eval_env = AtariEnvWrapper(
        game=config.game,
        vqvae=vqvae,
        device=device,
        frame_skip=config.frame_skip,
        history_len=4,
        max_episode_steps=config.max_episode_steps,
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
        n_vocab=wm_ckpt['n_vocab'],
        n_actions=n_actions,
        history_len=4,
        n_tokens=n_tokens,
        config=dqn_config,
        device=device,
    )
    print(f"  Policy parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    
    # Create buffers
    buffer = DualReplayBuffer(
        real_capacity=config.real_buffer_size,
        imagined_capacity=config.imagined_buffer_size,
        history_len=4,
        n_tokens=n_tokens,
        real_ratio=1.0 - config.imagined_ratio,
    )
    
    # Create imagined rollout generator (if using imagination)
    imagination = None
    if config.imagined_ratio > 0:
        imagination = ImaginedRolloutGenerator(world_model, agent, device)
    
    # Training state
    stats = TrainingStats()
    states = vec_env.reset()  # (n_envs, history_len, n_tokens)
    episode_rewards = []
    episode_lengths = []
    train_losses = []
    q_values = []
    best_eval_reward = float('-inf')
    global_step = 0
    
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
        
        # Store transitions for each env
        for i in range(config.n_envs):
            buffer.add_real(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        states = next_states
        
        # Collect completed episodes
        completed = vec_env.get_completed_episodes()
        for ep in completed:
            episode_rewards.append(ep['reward'])
            episode_lengths.append(ep['length'])
    
    print(f"  Warmup complete. Buffer size: {buffer.real_size}")
    print(f"  Episodes: {len(episode_rewards)}, Avg reward: {np.mean(episode_rewards) if episode_rewards else 0:.1f}")
    
    # Epoch-based training
    for epoch in range(1, config.n_epochs + 1):
        epoch_start = time.time()
        epoch_losses = []
        epoch_q_values = []
        epoch_rewards = []
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.n_epochs}")
        print('='*60)
        
        pbar = tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch}")
        
        for step in pbar:
            global_step += 1
            
            # Select actions for all envs (batch)
            actions = np.zeros(config.n_envs, dtype=np.int64)
            for i in range(config.n_envs):
                actions[i] = agent.select_action(states[i])
            
            # Environment step (all envs)
            next_states, rewards, dones, infos = vec_env.step(actions)
            
            # Store transitions for each env
            for i in range(config.n_envs):
                buffer.add_real(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
            states = next_states
            
            # Collect completed episodes
            completed = vec_env.get_completed_episodes()
            for ep in completed:
                episode_rewards.append(ep['reward'])
                episode_lengths.append(ep['length'])
                epoch_rewards.append(ep['reward'])
            
            # Generate imagined experience (if enabled)
            if (imagination is not None and 
                step % config.imagined_update_freq == 0 and 
                buffer.real_size >= config.imagined_batch_size):
                start_states = buffer.sample_real_states(config.imagined_batch_size)
                start_states = start_states.to(device)
                
                im_states, im_actions, im_rewards, im_next, im_dones = imagination.generate(
                    start_states,
                    rollout_len=config.imagined_rollout_len,
                )
                
                if len(im_states) > 0:
                    buffer.add_imagined_batch(im_states, im_actions, im_rewards, im_next, im_dones)
            
            # Train policy (multiple gradient steps)
            if step % config.train_freq == 0 and buffer.is_ready(config.batch_size):
                for _ in range(config.gradient_steps):
                    batch = buffer.sample_mixed(config.batch_size)
                    loss = agent.train_step(batch)
                    epoch_losses.append(loss)
                    
                    # Track Q-values occasionally
                    if len(epoch_losses) % 100 == 0:
                        with torch.no_grad():
                            q = agent.policy_net(batch.states.to(device))
                            epoch_q_values.append(q.mean().item())
            
            # Update progress bar
            if step % 100 == 0:
                avg_reward = np.mean(epoch_rewards[-10:]) if epoch_rewards else 0
                avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else 0
                pbar.set_postfix({
                    'reward': f'{avg_reward:.1f}',
                    'loss': f'{avg_loss:.4f}',
                    'eps': f'{agent.epsilon:.3f}',
                    'buf': f'{buffer.real_size//1000}k',
                })
        
        # End of epoch - evaluate
        print("\n  Evaluating...")
        eval_mean, eval_std, eval_rewards = evaluate_policy(eval_env, agent, config.eval_episodes)
        
        # Update stats
        stats.epoch.append(epoch)
        stats.step.append(global_step)
        stats.train_loss.append(np.mean(epoch_losses) if epoch_losses else 0)
        stats.avg_q_value.append(np.mean(epoch_q_values) if epoch_q_values else 0)
        stats.episode_reward.append(np.mean(epoch_rewards) if epoch_rewards else 0)
        stats.episode_length.append(np.mean(episode_lengths[-len(epoch_rewards):]) if epoch_rewards else 0)
        stats.eval_reward_mean.append(eval_mean)
        stats.eval_reward_std.append(eval_std)
        stats.epsilon.append(agent.epsilon)
        stats.buffer_real_size.append(buffer.real_size)
        stats.buffer_imagined_size.append(buffer.imagined_size)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train reward (avg): {stats.episode_reward[-1]:.1f}")
        print(f"    Eval reward: {eval_mean:.1f} +/- {eval_std:.1f}")
        print(f"    Loss: {stats.train_loss[-1]:.4f}")
        print(f"    Epsilon: {agent.epsilon:.3f}")
        print(f"    Buffer: {buffer.real_size:,} real, {buffer.imagined_size:,} imagined")
        print(f"    Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = eval_mean > best_eval_reward
        if is_best:
            best_eval_reward = eval_mean
            agent.save(f"{run_dir}/policy_best.pt")
            print(f"    New best! Saved to policy_best.pt")
        
        agent.save(f"{run_dir}/policy_epoch{epoch}.pt")
        
        # Update plots and stats
        plot_training_progress(stats, f"{run_dir}/training_progress.png")
        save_training_stats(stats, f"{run_dir}/training_stats.txt")
    
    # Final save
    agent.save(f"{run_dir}/policy_final.pt")
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
    
    parser = argparse.ArgumentParser(description="Dyna-style policy training")
    parser.add_argument("--base_dir", type=str, default="checkpoints/v2/mspacman",
                        help="Directory with VQ-VAE and world model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=10000,
                        help="Environment steps per epoch")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    config = DynaConfig(
        n_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )
    train_dyna(base_dir=args.base_dir, config=config, device=args.device)
