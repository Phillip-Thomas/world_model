"""
Default Configuration for World Model
======================================
Single source of truth for all hyperparameters.
CLI arguments override these defaults; saved config captures exact settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ModelConfig:
    """World model architecture parameters."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 10
    dropout: float = 0.1
    history_len: int = 4
    
    # These are typically loaded from VQ-VAE checkpoint, but provide defaults
    n_vocab: int = 32
    token_h: int = 21
    token_w: int = 16
    n_actions: int = 4


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    n_epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_batches: int = 200  # 0 = all batches
    full_val_every: int = 999  # Full gold eval frequency
    
    # Multi-step rollout training
    rollout_steps: int = 5
    rollout_ratio: float = 0.3
    step_discount: float = 0.7
    teacher_forcing_prob: float = 0.75


@dataclass  
class WeightingConfig:
    """Token importance weighting parameters."""
    use_hybrid: bool = True  # v2.0 hybrid vs v1.x legacy
    
    # v3.0 focal loss (auto-upweights hard tokens)
    use_focal_loss: bool = True
    focal_gamma: float = 3.0  # Higher = more focus on hard examples
    
    # v2.0 hybrid params
    motion_scale: float = 2.0
    eventness_scale: float = 2.0
    persistence_scale: float = 2.0
    max_ratio: float = 6.0
    
    # Legacy v1.x params (only used if use_hybrid=False)
    motion_weight: float = 4.0
    continuous_bonus: float = 2.0
    max_weight: float = 8.0


@dataclass
class InferenceConfig:
    """Inference/playback parameters."""
    deterministic: bool = True
    temperature: float = 0.8
    top_k: int = 5
    
    # Advanced inference (game-agnostic improvements)
    logit_smoothing: float = 0.0       # Blend with previous logits (0.0-0.5)
    n_candidates: int = 1              # Sample N, pick best by continuity (1=disabled)
    adaptive_temp: bool = False        # Adjust temp based on model confidence
    temp_boost: float = 0.3            # Extra temp for uncertain tokens (when adaptive)


@dataclass
class WorldModelConfig:
    """Complete configuration for world model training and inference."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Runtime info (populated during training)
    timestamp: Optional[str] = None
    run_dir: Optional[str] = None
    vqvae_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to nested dictionary for JSON serialization."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'weighting': asdict(self.weighting),
            'inference': asdict(self.inference),
            'timestamp': self.timestamp,
            'run_dir': self.run_dir,
            'vqvae_path': self.vqvae_path,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, d: dict) -> 'WorldModelConfig':
        """Create config from dictionary."""
        # Helper to filter dict to only known dataclass fields
        def filter_fields(data: dict, dataclass_type) -> dict:
            if not data:
                return {}
            valid_fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
            return {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(
            model=ModelConfig(**filter_fields(d.get('model', {}), ModelConfig)),
            training=TrainingConfig(**filter_fields(d.get('training', {}), TrainingConfig)),
            weighting=WeightingConfig(**filter_fields(d.get('weighting', {}), WeightingConfig)),
            inference=InferenceConfig(**filter_fields(d.get('inference', {}), InferenceConfig)),
            timestamp=d.get('timestamp'),
            run_dir=d.get('run_dir'),
            vqvae_path=d.get('vqvae_path'),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WorldModelConfig':
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, path: str) -> 'WorldModelConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_json(f.read())
    
    def update_from_args(self, args) -> 'WorldModelConfig':
        """
        Update config from argparse namespace.
        Only updates fields that are explicitly set (not None).
        """
        # Training params
        if hasattr(args, 'epochs') and args.epochs is not None:
            self.training.n_epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'lr') and args.lr is not None:
            self.training.learning_rate = args.lr
        if hasattr(args, 'max_batches') and args.max_batches is not None:
            self.training.max_batches = args.max_batches
        if hasattr(args, 'full_val_every') and args.full_val_every is not None:
            self.training.full_val_every = args.full_val_every
        if hasattr(args, 'rollout_steps') and args.rollout_steps is not None:
            self.training.rollout_steps = args.rollout_steps
        if hasattr(args, 'rollout_ratio') and args.rollout_ratio is not None:
            self.training.rollout_ratio = args.rollout_ratio
        if hasattr(args, 'teacher_forcing') and args.teacher_forcing is not None:
            self.training.teacher_forcing_prob = args.teacher_forcing
        
        # Weighting params
        if hasattr(args, 'use_hybrid') and args.use_hybrid is not None:
            self.weighting.use_hybrid = args.use_hybrid
        if hasattr(args, 'motion_scale') and args.motion_scale is not None:
            self.weighting.motion_scale = args.motion_scale
        if hasattr(args, 'eventness_scale') and args.eventness_scale is not None:
            self.weighting.eventness_scale = args.eventness_scale
        if hasattr(args, 'persistence_scale') and args.persistence_scale is not None:
            self.weighting.persistence_scale = args.persistence_scale
        if hasattr(args, 'max_ratio') and args.max_ratio is not None:
            self.weighting.max_ratio = args.max_ratio
        if hasattr(args, 'motion_weight') and args.motion_weight is not None:
            self.weighting.motion_weight = args.motion_weight
        if hasattr(args, 'continuous_bonus') and args.continuous_bonus is not None:
            self.weighting.continuous_bonus = args.continuous_bonus
        if hasattr(args, 'max_weight') and args.max_weight is not None:
            self.weighting.max_weight = args.max_weight
        
        # Inference params
        if hasattr(args, 'stochastic'):
            self.inference.deterministic = not args.stochastic
        if hasattr(args, 'deterministic'):
            self.inference.deterministic = args.deterministic
        if hasattr(args, 'temperature') and args.temperature is not None:
            self.inference.temperature = args.temperature
        if hasattr(args, 'top_k') and args.top_k is not None:
            self.inference.top_k = args.top_k
        
        return self


# Global default config instance
DEFAULT_CONFIG = WorldModelConfig()
