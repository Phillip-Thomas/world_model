"""
Config Loader Utilities
=======================
Helpers for loading, merging, and resolving configs.
"""

import os
import json
from typing import Optional
from .defaults import WorldModelConfig, DEFAULT_CONFIG


def load_run_config(run_dir: str) -> Optional[WorldModelConfig]:
    """
    Load config from a training run directory.
    
    Args:
        run_dir: Path to run directory (e.g., "checkpoints/v2/atari/runs/20260109_120021")
        
    Returns:
        WorldModelConfig if config.json exists, None otherwise
    """
    config_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(config_path):
        return WorldModelConfig.load(config_path)
    return None


def get_config_for_checkpoint(checkpoint_path: str) -> WorldModelConfig:
    """
    Get config for a checkpoint, loading from its run directory if available.
    Falls back to defaults if no config found.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        WorldModelConfig (from run dir or defaults)
    """
    # Try to find config.json in same directory as checkpoint
    run_dir = os.path.dirname(checkpoint_path)
    config = load_run_config(run_dir)
    
    if config is not None:
        print(f"  Loaded config from {run_dir}/config.json")
        return config
    
    # Fall back to defaults
    print(f"  No config.json found, using defaults")
    return WorldModelConfig()


def merge_legacy_config(legacy_dict: dict) -> WorldModelConfig:
    """
    Convert legacy config.json format to new WorldModelConfig.
    Handles old configs that don't have the nested structure.
    
    Args:
        legacy_dict: Dictionary from old-style config.json
        
    Returns:
        WorldModelConfig with values from legacy config
    """
    config = WorldModelConfig()
    
    # Map legacy fields to new structure
    if 'training' in legacy_dict:
        t = legacy_dict['training']
        config.training.n_epochs = t.get('n_epochs', config.training.n_epochs)
        config.training.batch_size = t.get('batch_size', config.training.batch_size)
        config.training.learning_rate = t.get('learning_rate', config.training.learning_rate)
        config.training.max_batches = t.get('max_batches', config.training.max_batches)
        config.training.full_val_every = t.get('full_val_every', config.training.full_val_every)
        config.training.rollout_steps = t.get('rollout_steps', config.training.rollout_steps)
        config.training.rollout_ratio = t.get('rollout_ratio', config.training.rollout_ratio)
    
    if 'model' in legacy_dict:
        m = legacy_dict['model']
        config.model.d_model = m.get('d_model', config.model.d_model)
        config.model.n_heads = m.get('n_heads', config.model.n_heads)
        config.model.n_layers = m.get('n_layers', config.model.n_layers)
        config.model.dropout = m.get('dropout', config.model.dropout)
        config.model.history_len = m.get('history_len', config.model.history_len)
        config.model.n_vocab = m.get('n_vocab', config.model.n_vocab)
        config.model.n_actions = m.get('n_actions', config.model.n_actions)
    
    if 'weighting' in legacy_dict:
        w = legacy_dict['weighting']
        config.weighting.use_hybrid = w.get('use_hybrid', config.weighting.use_hybrid)
        config.weighting.motion_scale = w.get('motion_scale', config.weighting.motion_scale)
        config.weighting.eventness_scale = w.get('eventness_scale', config.weighting.eventness_scale)
        config.weighting.persistence_scale = w.get('persistence_scale', config.weighting.persistence_scale)
        config.weighting.max_ratio = w.get('max_ratio', config.weighting.max_ratio)
        config.weighting.motion_weight = w.get('motion_weight', config.weighting.motion_weight)
        config.weighting.continuous_bonus = w.get('continuous_bonus', config.weighting.continuous_bonus)
        config.weighting.max_weight = w.get('max_weight', config.weighting.max_weight)
        config.weighting.step_discount = w.get('step_discount', config.training.step_discount)
        # Handle teacher_forcing in weighting (legacy location)
        if 'teacher_forcing_prob' in w:
            config.training.teacher_forcing_prob = w['teacher_forcing_prob']
    
    config.timestamp = legacy_dict.get('timestamp')
    
    return config


def load_config_with_fallback(run_dir: str) -> WorldModelConfig:
    """
    Load config from run directory, handling both new and legacy formats.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        WorldModelConfig
    """
    config_path = os.path.join(run_dir, 'config.json')
    
    if not os.path.exists(config_path):
        print(f"  No config.json found in {run_dir}, using defaults")
        return WorldModelConfig()
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's new format (has nested 'inference' key) or legacy
    if 'inference' in data:
        # New format
        return WorldModelConfig.from_dict(data)
    else:
        # Legacy format
        print(f"  Converting legacy config format from {run_dir}")
        return merge_legacy_config(data)
