"""
Training Configuration (v3)
============================
Unified configuration for the continuous training loop.

Key improvements over v2:
- Continuous training mode with periodic VQ-VAE updates
- Efficient buffer settings (single frame storage)
- Codebook rehearsal parameters
- Curiosity-driven exploration settings
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os


@dataclass
class VQVAEConfig:
    """VQ-VAE configuration."""
    # Architecture
    in_channels: int = 3
    hidden_channels: int = 64
    latent_channels: int = 256
    n_embeddings: int = 512  # Codebook size
    n_residual: int = 2
    input_size: Tuple[int, int] = (84, 64)  # (H, W)
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 64
    
    # EMA codebook settings
    ema_decay: float = 0.999  # Slow decay for stability during fine-tuning
    ema_update_every: int = 1
    dead_code_threshold: float = 0.01
    
    # Continuous training
    update_every_n_steps: int = 5000  # Fine-tune VQ-VAE every N env steps
    warmup_steps: int = 10000  # Don't update VQ-VAE until this many steps
    use_rehearsal: bool = True  # Use codebook rehearsal buffer
    rehearsal_frac: float = 0.5  # Fraction of batch from rehearsal
    
    @property
    def token_h(self) -> int:
        return self.input_size[0] // 4
    
    @property
    def token_w(self) -> int:
        return self.input_size[1] // 4
    
    @property
    def n_tokens(self) -> int:
        return self.token_h * self.token_w


@dataclass
class WorldModelConfig:
    """World Model configuration."""
    # Architecture
    n_tokens: int = 336  # 21 × 16 for 84×64 input
    n_vocab: int = 512  # VQ-VAE codebook size
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    n_actions: int = 9  # Atari actions
    history_len: int = 4  # Frame stacking
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 64  # Larger batches for stable WM training
    sequence_length: int = 32
    grad_clip: float = 1.0
    
    # Continuous fine-tuning
    finetune_lr: float = 1e-5  # Lower LR for fine-tuning
    update_every_n_steps: int = 100  # Fine-tune WM every N env steps
    updates_per_step: int = 1  # Gradient steps per update
    recent_k: int = 50000  # Recency window for sampling
    recent_frac: float = 0.5  # Fraction from recent data


@dataclass
class PolicyConfig:
    """DQN Policy configuration."""
    # Architecture
    n_tokens: int = 336
    n_vocab: int = 512
    n_actions: int = 9
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    history_len: int = 4
    
    # Training
    learning_rate: float = 5e-5  # Lower LR to reduce catastrophic forgetting
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    target_update_freq: int = 1  # Update target every step (with soft update)
    grad_clip: float = 10.0
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100000  # Linear decay over steps
    
    # Prioritized replay
    use_prioritized: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_increment: float = 0.001


@dataclass
class TrainingConfig:
    """
    Unified training configuration for continuous loop.
    
    The v3 training loop supports three modes:
    1. Staged (like v2): Train VQ-VAE → WM → Policy sequentially
    2. Continuous-WM: VQ-VAE frozen, WM + Policy update together
    3. Continuous-Full: All three models update (requires raw frame storage)
    """
    
    # Environment
    game: str = "ALE/MsPacman-v5"
    frame_skip: int = 4
    max_episode_steps: int = 10000
    n_envs: int = 8  # Parallel envs for faster collection
    
    # Training mode
    mode: str = "continuous_wm"  # "staged", "continuous_wm", or "continuous_full"
    
    # Steps and epochs
    total_steps: int = 1_000_000  # Total env steps
    steps_per_epoch: int = 5000  # Steps between checkpoints
    warmup_steps: int = 5000  # Steps before training starts
    
    # Training frequency
    train_freq: int = 4  # Train policy every N env steps
    gradient_steps: int = 2  # Policy gradient steps per training call
    
    # Dyna-style imagination (trust-weighted)
    imagined_rollout_len: int = 3
    imagined_ratio: float = 0.05  # Fraction of imagined data in policy batches
    imagined_update_freq: int = 1000  # Generate imagined rollouts every N steps
    imagined_batch_size: int = 32  # Seeds per imagination batch
    trust_scale: float = 1.0  # Scale for trust = exp(-scale * wm_loss)
    trust_threshold: float = 0.3  # Min trust to keep seed (exp(-1.2) ≈ 0.3)
    trust_horizon_decay: float = 0.95  # Decay trust per rollout step
    
    # Buffer settings (efficient storage!)
    real_buffer_size: int = 1_000_000  # ~680 MB with efficient storage
    imagined_buffer_size: int = 50000
    history_len: int = 4
    
    # Raw frame buffer (for continuous_full mode)
    frame_buffer_size: int = 100000  # ~1.6 GB for RGB frames
    
    # Evaluation
    eval_episodes: int = 10
    eval_freq: int = 5000  # Eval every N steps
    
    # Checkpointing
    save_freq: int = 50000  # Save every N steps
    keep_last_n: int = 3  # Keep last N checkpoints
    
    # Curiosity-driven exploration (optional)
    use_curiosity: bool = False
    curiosity_scale: float = 0.1  # Intrinsic reward scale
    
    # Sub-configs
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    
    # Paths (set at runtime)
    base_dir: Optional[str] = None
    run_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set derived values."""
        # Ensure consistency
        self.vqvae.n_embeddings = self.world_model.n_vocab
        self.world_model.n_tokens = self.vqvae.n_tokens
        self.world_model.history_len = self.history_len
        self.policy.n_tokens = self.vqvae.n_tokens
        self.policy.n_vocab = self.vqvae.n_embeddings
        self.policy.history_len = self.history_len
    
    def get_run_dir(self, base_dir: str = None) -> str:
        """Get or create run directory."""
        if self.run_dir:
            return self.run_dir
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if base_dir is None:
            base_dir = self.base_dir or "checkpoints"
        
        self.run_dir = os.path.join(base_dir, "runs", timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        return self.run_dir
    
    def save(self, path: str):
        """Save config to JSON."""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON."""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Handle nested dataclasses
        vqvae = VQVAEConfig(**config_dict.pop('vqvae', {}))
        world_model = WorldModelConfig(**config_dict.pop('world_model', {}))
        policy = PolicyConfig(**config_dict.pop('policy', {}))
        
        return cls(
            vqvae=vqvae,
            world_model=world_model,
            policy=policy,
            **config_dict
        )
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("Training Configuration (v3)")
        print("=" * 60)
        print(f"\nMode: {self.mode}")
        print(f"Game: {self.game}")
        print(f"Total steps: {self.total_steps:,}")
        
        print(f"\n--- Buffer Memory ---")
        # Efficient token buffer
        token_mem = self.real_buffer_size * self.vqvae.n_tokens * 2 / (1024**2)
        print(f"Token buffer: {token_mem:.1f} MB ({self.real_buffer_size:,} transitions)")
        
        if self.mode == "continuous_full":
            frame_mem = self.frame_buffer_size * 3 * 84 * 64 / (1024**2)
            print(f"Frame buffer: {frame_mem:.1f} MB ({self.frame_buffer_size:,} frames)")
        
        print(f"\n--- VQ-VAE ---")
        print(f"Codebook: {self.vqvae.n_embeddings} codes")
        print(f"Tokens: {self.vqvae.n_tokens} ({self.vqvae.token_h}×{self.vqvae.token_w})")
        
        print(f"\n--- World Model ---")
        print(f"d_model: {self.world_model.d_model}, layers: {self.world_model.n_layers}")
        print(f"Fine-tune every: {self.world_model.update_every_n_steps} steps")
        
        print(f"\n--- Policy ---")
        print(f"LR: {self.policy.learning_rate}")
        print(f"Batch size: {self.policy.batch_size}")
        print("=" * 60)


def get_default_config(game: str = "mspacman") -> TrainingConfig:
    """Get default configuration for a game."""
    game_names = {
        "mspacman": "ALE/MsPacman-v5",
        "breakout": "ALE/Breakout-v5",
        "pong": "ALE/Pong-v5",
        "spaceinvaders": "ALE/SpaceInvaders-v5",
    }
    
    config = TrainingConfig(
        game=game_names.get(game.lower(), game),
    )
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_default_config("mspacman")
    config.print_summary()
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        config.save(f.name)
        loaded = TrainingConfig.load(f.name)
        print(f"\n[OK] Config save/load works!")
        print(f"Loaded mode: {loaded.mode}")
