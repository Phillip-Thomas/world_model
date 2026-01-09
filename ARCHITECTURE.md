# Atari World Model Architecture Specification (v2.1)

## Overview

Two-stage architecture for learning game dynamics:
1. **VQ-VAE v2.3**: Encodes 84×64 RGB frames → 21×16 discrete tokens (336 tokens)
2. **World Model v2.1**: Transformer with multi-step rollouts, hybrid importance weighting

```
Frame (84×64×3) → VQ-VAE Encoder → Tokens (21×16) → World Model → Next Tokens → VQ-VAE Decoder → Frame
     ↑                                    ↑
  4-frame history                    Action + Teacher Forcing
```

---

## Model 1: VQ-VAE v2.3

### Purpose
Compress continuous images into discrete token sequences for transformer processing.

### Architecture

```
Input: (B, 3, 84, 64) normalized to [-1, 1]

ENCODER:
├─ Conv2d(3→64, k=3, p=1)
├─ Conv2d(64→128, k=4, s=2, p=1) + GroupNorm(8) + GELU    # 84×64 → 42×32
├─ Conv2d(128→256, k=4, s=2, p=1) + GroupNorm(8) + GELU  # 42×32 → 21×16
├─ ResidualBlock(256) × 2
└─ Conv2d(256→256, k=1)  # to latent

QUANTIZER (VectorQuantizerHiRes):
├─ Embedding(32, 256)  # 32 codebook entries (v2.3: reduced for Atari)
├─ EMA updates (decay=0.99, update_every=10)
└─ Dead code reset (threshold=1%, every 500 batches)

DECODER (v2.1 - no checkerboard):
├─ Conv2d(256→256, k=1)
├─ ResidualBlock(256) × 2
├─ Upsample(×2, nearest) + Conv2d(256→128, k=3, p=1) + GroupNorm + GELU
├─ Upsample(×2, nearest) + Conv2d(128→64, k=3, p=1) + GroupNorm + GELU
├─ Conv2d(64→3, k=3, p=1)
└─ Tanh()

Output: (B, 3, 84, 64), Token indices: (B, 21, 16)
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `in_channels` | 3 | RGB input |
| `hidden_channels` | 64 | Base channel width |
| `latent_channels` | 256 | Codebook embedding dimension |
| `n_embeddings` | 32 | Codebook size (reduced for Atari) |
| `n_residual` | 2 | Residual blocks at bottleneck |
| `input_size` | (84, 64) | Aspect-preserved Atari frame |

### Total Parameters: ~6.0M

---

## Model 2: Temporal World Model v2.1

### Purpose
Predict next frame tokens given 4-frame history and action, with multi-step rollout training.

### Architecture

```
Input: 
  - frame_history: (B, 4, 336) - 4 frames × 336 tokens each
  - action: (B,) - discrete action index

EMBEDDINGS:
├─ Token Embedding: Embedding(32, 256)       # codebook → d_model
├─ Action Embedding: Embedding(4, 256)       # action → d_model
├─ Spatial Position: Embedding(21, 128) + Embedding(16, 128)  # 2D position
├─ Temporal Position: Embedding(5, 256)      # frame index (0-3=history, 4=target)
└─ Action Spatial Bias: (4, 21, 16, 256)     # v1.6: per-action spatial modulation

ACTION CONDITIONING (v1.6 - Strong):
├─ Per-action learned spatial bias
├─ Scaled action embedding (learnable scale)
└─ FiLM modulation: query_tokens * (1 + 0.2 * tanh(action))

SEQUENCE CONSTRUCTION:
├─ [frame_0 tokens (336)] + temporal_pos_0 + spatial_pos
├─ [frame_1 tokens (336)] + temporal_pos_1 + spatial_pos
├─ [frame_2 tokens (336)] + temporal_pos_2 + spatial_pos
├─ [frame_3 tokens (336)] + temporal_pos_3 + spatial_pos
├─ [action (1)]
└─ [query tokens (336)] + temporal_pos_4 + spatial_pos + action_bias + FiLM

Total sequence length: 336×4 + 1 + 336 = 1681 tokens

TRANSFORMER:
├─ TransformerEncoderLayer × 10
│   ├─ d_model: 256
│   ├─ n_heads: 8
│   ├─ dim_feedforward: 1024 (4× d_model)
│   ├─ dropout: 0.1
│   ├─ activation: GELU
│   └─ norm_first: True (Pre-LN)
└─ LayerNorm(256)

OUTPUT:
├─ Extract last 336 positions (query outputs)
├─ LayerNorm(256)
└─ Linear(256, 32)  # project to vocab logits

Output: (B, 336, 32) logits over codebook
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_vocab` | 32 | VQ-VAE codebook size |
| `n_actions` | 4 | Breakout: NOOP, FIRE, RIGHT, LEFT |
| `d_model` | 256 | Transformer hidden dimension |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 10 | Transformer layers |
| `token_h` | 21 | Token grid height |
| `token_w` | 16 | Token grid width |
| `dropout` | 0.1 | Dropout rate |
| `max_history` | 4 | Number of past frames |

### Total Parameters: ~8.3M

---

## Training: World Model v2.1

### Multi-Step Rollout Training

```
Standard training (1-rollout_ratio fraction):
  loss = CE(model(history, action), target)

Multi-step rollout (rollout_ratio fraction):
  For k in 1..K:
    logits = model(current_history, action_k)
    loss += discount^k * weighted_CE(logits, target_k)
    
    # Teacher forcing (10%): use ground truth sometimes
    if random() < 0.1:
      next_history = target_k
    else:
      next_history = argmax(logits)
    
    current_history = roll(current_history) + next_history
```

### Hybrid Importance Weighting (v2.0)

```python
weights = (
    1.0
    + motion_scale * embedding_distance_motion   # L2 in embedding space
    + eventness_scale * spike_detection          # Unusual motion = important
    + persistence_scale * multi_frame_motion     # Ball moves every frame
)
# Safe percentile capping: 95th/median ≤ 6.0
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 8 | Reduced for K=5 rollouts |
| `learning_rate` | 5e-4 | Optimal from LR sweep |
| `optimizer` | AdamW (weight_decay=0.01) | |
| `grad_clip` | 1.0 | |
| `rollout_steps` | 5 | K-step unrolling |
| `rollout_ratio` | 0.5 | 50% multi-step batches |
| `teacher_forcing_prob` | 0.1 | Prevent error accumulation |
| `motion_scale` | 2.0 | Embedding-distance motion |
| `eventness_scale` | 2.0 | Spike detection |
| `persistence_scale` | 1.0 | Multi-frame consistency |
| `max_ratio` | 6.0 | Percentile weight cap |
| `step_discount` | 0.7 | Discount for future steps |

---

## Data Pipeline

### Collection (AtariCollector v3)

```
Game: ALE/Breakout-v5
Data: ~959K frames (random + human gameplay)

Frame preprocessing:
  1. RGB observation (210×160)
  2. Max-pool last 2 frames (reduce Atari flicker)
  3. Resize to 84×64 (preserve aspect ratio)
  4. uint8 storage, normalize to [-1,1] at training time

Frame/Action Alignment (v3 fix):
  - frames: length N+1 (includes initial state per episode)
  - actions/rewards/dones: length N
  - action[i] transitions frames[i] → frames[i+1]
```

### Data Split

```python
# Episode-based split (no frame leakage)
train_episodes: 90%
val_episodes: 10%

# Human gameplay weighted 5× in sampler
expert_weight = 5.0
```

---

## File Structure

```
world_model/
├── v2/
│   ├── models/
│   │   ├── vqvae_hires.py           # VQ-VAE v2.3
│   │   └── temporal_world_model.py  # World Model v2.1
│   ├── train/
│   │   ├── train_vqvae_hires.py     # VQ-VAE training
│   │   └── train_wm_hires.py        # World model training
│   ├── data/
│   │   ├── atari_dataset.py         # Data collection & loading
│   │   └── multistep_dataset.py     # K-step rollout dataset
│   ├── eval/
│   │   └── gold_eval.py             # Frozen evaluation suite
│   ├── utils/
│   │   ├── importance_weights.py    # Hybrid token weighting
│   │   ├── analyze_vqvae.py         # VQ-VAE analysis tools
│   │   └── check_action_conditioning.py
│   ├── play_atari_hires.py          # Interactive play (15 FPS)
│   ├── record_human_play.py         # Human data collection
│   └── collect_and_train.py         # End-to-end pipeline
├── tests/
│   ├── test_full_pipeline.py
│   └── test_new_dataloader.py
├── scripts/
│   └── expand_dataset.py            # Dataset expansion utilities
└── checkpoints/v2/atari/
    ├── atari_game_data.npz          # Raw frames + actions
    ├── atari_tokens_hires.npz       # Cached tokens
    ├── human_gameplay.npz           # Human demonstration data
    ├── atari_vqvae_hires.pt         # VQ-VAE checkpoint
    ├── atari_world_model_hires.pt   # World model checkpoint
    ├── gold_eval_indices.json       # Frozen eval indices
    └── runs/                        # Per-run checkpoints
```

---

## Evaluation

### Gold Eval Suite (Frozen Indices)

| Tier | Size | Frequency | Purpose |
|------|------|-----------|---------|
| Canary | 500 | Every epoch | Quick regression detection |
| FastVal | 6000 | Every epoch | Balanced action distribution |
| Gold | Full | Every N epochs | Checkpoint criterion |

### Metrics

| Metric | Description |
|--------|-------------|
| `val_acc` | 1-step token prediction accuracy |
| `rollout_acc_h5` | 5-step open-loop accuracy |
| `rollout_acc_h10` | 10-step open-loop accuracy |
| `mirror_consistency` | LEFT/RIGHT equivariance |
| `action_sensitivity` | KL(LEFT \|\| RIGHT) divergence |

---

## Results

| Metric | v1.0 | v2.1 |
|--------|------|------|
| Val Accuracy | 42% | **90%+** |
| Rollout@5 | Poor | Good |
| Rollout@10 | Very Poor | Fair |
| Ball Tracking | Poor | Good |
| Action Response | Weak | Strong |

---

## Implemented Improvements (v2.0-v2.1)

- ✅ Multi-step rollout training (K=5)
- ✅ Teacher forcing / scheduled sampling (10%)
- ✅ Hybrid importance weighting (embedding-distance + eventness + persistence)
- ✅ Normalized discounted loss
- ✅ Efficient history update (roll instead of cat)
- ✅ Percentile-based weight capping
- ✅ Strong action conditioning (FiLM + spatial bias)
- ✅ Real-time play at 15 FPS

## Remaining Improvements

1. **Small-component detection**: Weight connected regions of motion (bullets/projectiles)
2. **Pixel-space motion**: Decode and compute actual optical flow
3. **Hierarchical prediction**: Predict at multiple temporal scales
4. **Diffusion decoder**: Replace VQ-VAE decoder with conditional diffusion
