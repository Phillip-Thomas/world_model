# Transformer Visual World Model (v2)

A visual world model for Atari games using VQ-VAE tokenization and transformer-based next-frame prediction. Trains on a single GPU in ~2 hours.

---

## Architecture (v2.0)

```
                              TRAINING PIPELINE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Frame History      VQ-VAE              Tokens         World Model        │
│   ─────────────      ──────              ──────         ───────────        │
│                                                                             │
│   ┌─────────┐       ┌──────────┐        ┌──────┐       ┌────────────┐      │
│   │ 84×64×3 │──┬───►│ Encoder  │───────►│21×16 │──┬───►│Transformer │      │
│   │   RGB   │  │    │ CNN+VQ   │        │tokens│  │    │ 10 layers  │      │
│   └─────────┘  │    └──────────┘        └──────┘  │    └─────┬──────┘      │
│       ×4       │                                  │          │             │
│   (history)    │                                  │          │             │
│                │                                  │          ▼             │
│   Action ──────┴──────────────────────────────────┘    ┌──────────┐        │
│   (0-3)                                                │21×16 pred│        │
│                                                        │  logits  │        │
│   Next Frame         VQ-VAE              Tokens        └────┬─────┘        │
│   ┌─────────┐        ┌──────────┐        ┌──────┐           │              │
│   │ 84×64×3 │◄───────│ Decoder  │◄───────│21×16 │◄──────────┘              │
│   │   RGB   │        │   CNN    │        │tokens│                          │
│   └─────────┘        └──────────┘        └──────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Aspect-preserved resolution**: 84×64 input (vs 64×64 in v1) preserves Atari's native aspect ratio
- **Temporal context**: 4-frame history for motion understanding
- **Multi-step rollout training**: K-step unrolling with teacher forcing
- **Hybrid importance weighting**: Game-agnostic token weighting for better ball/projectile tracking
- **Strong action conditioning**: FiLM + spatial bias for reliable action effects

---

## VQ-VAE

| Spec | Value |
|------|-------|
| Input/Output | 84×64×3 |
| Latent | 21×16×256 |
| Codebook | 32 vectors |
| Compression | ~48× |
| Parameters | ~6M |

---

## World Model

| Spec | Value |
|------|-------|
| d_model | 256 |
| Heads | 8 |
| Layers | 10 |
| FFN | 1024 |
| Vocab | 32 |
| Tokens | 336 (21×16) |
| History | 4 frames |
| Parameters | 8.3M |

---

## Training

### Stage 1: VQ-VAE
```bash
python v2/train/train_vqvae_hires.py --epochs 50
```

### Stage 2: World Model
```bash
python v2/train/train_wm_hires.py --epochs 100 --rollout-steps 5 --rollout-ratio 0.5
```

### Key Training Features (v2.1)
- **Multi-step rollout**: K=5 steps with scheduled sampling
- **Hybrid importance weighting**: Embedding-distance + eventness + persistence
- **Normalized discounted loss**: Stable across K values
- **Teacher forcing**: 10% probability to prevent error accumulation

---

## Results

| Metric | v1.0 | v2.0 |
|--------|------|------|
| Token Accuracy | 42% | 90%+ |
| Rollout@5 | Poor | Good |
| Rollout@10 | Very Poor | Fair |
| Ball Tracking | Poor | Good |

---

## Inference / Play

```bash
# Play with latest model
python v2/play_atari_hires.py --latest

# Play with specific run
python v2/play_atari_hires.py --run 20260108_205042
```

**Controls:**
- ← → : Move paddle
- SPACE : Fire
- P : Pause
- R : Sync AI to real game
- Q : Quit

---

## Project Structure

```
world_model/
├── v2/
│   ├── models/
│   │   ├── vqvae_hires.py         # VQ-VAE encoder/decoder
│   │   └── temporal_world_model.py # Transformer world model
│   ├── data/
│   │   └── atari_dataset.py       # Dataset + collector
│   ├── train/
│   │   ├── train_vqvae_hires.py   # VQ-VAE training
│   │   └── train_wm_hires.py      # World model training
│   ├── eval/
│   │   └── gold_eval.py           # Frozen evaluation suite
│   └── play_atari_hires.py        # Interactive play
│
├── checkpoints/
│   └── v2/atari/
│       ├── atari_vqvae_hires.pt   # VQ-VAE weights
│       ├── atari_world_model_hires.pt  # World model weights
│       ├── atari_game_data.npz    # Training data
│       ├── atari_tokens_hires.npz # Pre-tokenized data
│       └── runs/                  # Per-run checkpoints
│
├── README.md
├── ARCHITECTURE.md
├── ROADMAP.md
└── requirements.txt
```

---

## Installation

```bash
# Clone and install
git clone <repo>
cd world_model
pip install -r requirements.txt

# Install Atari ROMs
pip install gymnasium[accept-rom-license]
```

---

## References

- VQ-VAE: [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
- Oasis: [oasis.decart.ai](https://oasis.decart.ai/)
- GameNGen: [gamengen.github.io](https://gamengen.github.io/)
- Genie: [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
