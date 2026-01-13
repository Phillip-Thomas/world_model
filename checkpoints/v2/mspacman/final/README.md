# Ms. Pac-Man Production Checkpoints

User-selected best checkpoints for Ms. Pac-Man world model pipeline.

## Checkpoints

| File | Description | Source Run | Size |
|------|-------------|------------|------|
| `vqvae_best.pt` | VQ-VAE encoder (64 codes, 21x16 tokens) | 20260112_095815 | 22.7 MB |
| `world_model_best.pt` | Transformer world model (10 layers) | 20260112_101247 | 33.3 MB |
| `policy_best.pt` | DQN policy (Dueling + PER) | 20260113_113048 | 40.8 MB |

## Usage

```python
# Load VQ-VAE
vqvae_ckpt = torch.load("final/vqvae_best.pt")
vqvae = VQVAEHiRes(n_embeddings=64)
vqvae.load_state_dict(vqvae_ckpt['model_state_dict'])

# Load World Model
wm_ckpt = torch.load("final/world_model_best.pt")
world_model = TemporalVisualWorldModel(...)
world_model.load_state_dict(wm_ckpt)

# Load Policy
agent = DQNAgent(...)
agent.load("final/policy_best.pt")
```

## Performance

- **VQ-VAE**: L1 error 0.006, 40/64 codes used per frame
- **Policy**: ~1200+ avg eval reward on Ms. Pac-Man

## Last Updated

2026-01-13
