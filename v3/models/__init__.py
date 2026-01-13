"""
Models for v3
=============
Reuses v2 model architectures with improved training loop.

The core model architectures (VQ-VAE, World Model, DQN) remain the same.
The improvements in v3 are in:
1. Memory-efficient buffer storage
2. Continuous training loop
3. Codebook rehearsal
"""

import sys
import os

# Add v2 to path for model imports
v2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "v2")
if v2_path not in sys.path:
    sys.path.insert(0, v2_path)

# Import models from v2
from models.vqvae_hires import VQVAEHiRes
from models.temporal_world_model import TemporalVisualWorldModel

# Re-export
__all__ = ["VQVAEHiRes", "TemporalVisualWorldModel"]
