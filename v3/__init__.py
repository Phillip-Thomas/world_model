"""
World Model v3
==============
Next-generation world model training with:
- Efficient token storage (single frame per step, reconstruct on sample)
- Continuous training loop support
- Codebook rehearsal for VQ-VAE fine-tuning
- Raw frame buffer for VQ-VAE updates

Key improvements over v2:
1. 30-60x memory reduction in replay buffer
2. Support for continuous VQ-VAE/WM/Policy training
3. Curiosity-driven exploration option
"""

__version__ = "3.0.0"
