"""
RL Agents for World Model
=========================
Agents that use the learned world model for control.

Available agents:
- MPCAgent: Model Predictive Control with random/CEM action sequence search
- DQNAgent: DQN operating in token/latent space (Dyna-style)

Utilities:
- ReplayBuffer: Experience replay for latent transitions
- DualReplayBuffer: Separate real/imagined buffers for Dyna training
"""

from .mpc_agent import MPCAgent, MPCConfig
from .dqn_agent import DQNAgent, DQNConfig, LatentDQN
from .replay_buffer import ReplayBuffer, DualReplayBuffer, PrioritizedReplayBuffer

__all__ = [
    'MPCAgent', 'MPCConfig',
    'DQNAgent', 'DQNConfig', 'LatentDQN',
    'ReplayBuffer', 'DualReplayBuffer', 'PrioritizedReplayBuffer',
]
