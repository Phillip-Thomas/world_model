"""
World Model Configuration
=========================
Centralized config system for training and inference.
"""

from .defaults import (
    WorldModelConfig,
    ModelConfig,
    TrainingConfig,
    WeightingConfig,
    InferenceConfig,
    DEFAULT_CONFIG,
)

from .loader import (
    load_run_config,
    get_config_for_checkpoint,
    load_config_with_fallback,
)

__all__ = [
    'WorldModelConfig',
    'ModelConfig',
    'TrainingConfig', 
    'WeightingConfig',
    'InferenceConfig',
    'DEFAULT_CONFIG',
    'load_run_config',
    'get_config_for_checkpoint',
    'load_config_with_fallback',
]
