# World Model Evolution Roadmap

**Goal**: Evolve from current working model to next-gen architecture (2026-competitive)

**Principle**: Each phase is self-contained, testable, and preserves previous functionality.

---

## Current State (v1.0) ✅

```
Single Frame → VQ-VAE → Tokens → Transformer → Tokens → VQ-VAE → Next Frame
                                      ↑
                                   Action
```

| Component | Status |
|-----------|--------|
| VQ-VAE (128 codes) | Working |
| Transformer (6L, 256d) | Working |
| Single frame context | Working |
| 64×64 resolution | Working |
| 4 discrete actions | Working |

---

## Phase 1: Temporal Conditioning 🎯

**Priority**: HIGH  
**Estimated time**: 1-2 weeks  
**Goal**: Model sees past N frames, not just current frame

### 1.1 Architecture Change

```
CURRENT (v1.0)
──────────────
[frame_t tokens] + [action] → Transformer → [frame_t+1 tokens]

NEW (v1.1)
──────────
[frame_t-3] [frame_t-2] [frame_t-1] [frame_t] + [action] → Transformer → [frame_t+1]
     ↓           ↓           ↓          ↓
   pos=0       pos=1       pos=2      pos=3     (temporal position embedding)
```

### 1.2 Implementation Tasks

```
□ 1.1.1  Create TemporalVisualWorldModel class (new file, don't modify original)
□ 1.1.2  Add frame history buffer in data collection
□ 1.1.3  Modify dataset to return (frame_history, action, next_frame) tuples
□ 1.1.4  Add temporal position embeddings (separate from spatial)
□ 1.1.5  Train on same game data with history context
□ 1.1.6  Compare metrics: v1.0 vs v1.1 accuracy
□ 1.1.7  Side-by-side visual comparison
```

### 1.3 Key Code Changes

```python
# New temporal embedding
self.temporal_embed = nn.Embedding(max_history + 1, d_model)  # +1 for prediction

# Forward with history
def forward(self, frame_history, action, target=None):
    """
    frame_history: (batch, n_frames, n_tokens)  # e.g., (B, 4, 64)
    action: (batch,)
    """
    B, T, N = frame_history.shape
    
    # Embed all frames
    all_tokens = []
    for t in range(T):
        frame_emb = self.token_embed(frame_history[:, t, :])
        frame_emb = frame_emb + self.spatial_pos(...)
        frame_emb = frame_emb + self.temporal_embed(t)  # NEW
        all_tokens.append(frame_emb)
    
    # Rest of forward...
```

### 1.4 Expected Improvements

| Metric | v1.0 | v1.1 (expected) |
|--------|------|-----------------|
| Token accuracy | 42% | 55-65% |
| Motion blur handling | Poor | Good |
| Object permanence | None | Some |
| Temporal consistency | Poor | Better |

---

## Phase 2: Causal (Autoregressive) Token Prediction

**Priority**: MEDIUM  
**Estimated time**: 1 week  
**Goal**: Predict tokens one-by-one for better spatial coherence

### 2.1 Architecture Change

```
CURRENT: Parallel prediction (all 64 tokens at once)
NEW: Autoregressive (predict token 0, then 1, then 2, ...)

Can use BOTH modes:
- Autoregressive for quality (offline generation)
- Parallel for speed (real-time play)
```

### 2.2 Implementation Tasks

```
□ 2.2.1  Add causal attention mask option
□ 2.2.2  Implement autoregressive sampling loop
□ 2.2.3  Add KV-cache for efficient autoregressive inference
□ 2.2.4  Compare quality: parallel vs autoregressive
□ 2.2.5  Benchmark speed tradeoff
```

---

## Phase 3: Higher Resolution

**Priority**: MEDIUM  
**Estimated time**: 2 weeks  
**Goal**: 64×64 → 128×128 or 256×256

### 3.1 Challenges

```
Resolution    Tokens    Transformer cost
64×64         8×8=64    O(64²) = 4K
128×128       16×16=256 O(256²) = 65K   ← 16× more expensive
256×256       32×32=1024 O(1024²) = 1M  ← 256× more expensive
```

### 3.2 Solutions

```
□ 3.2.1  Hierarchical VQ-VAE (multi-scale tokens)
□ 3.2.2  Patch-based processing (Oasis-style tiling)
□ 3.2.3  Sparse attention patterns
□ 3.2.4  Latent downsampling before transformer
```

---

## Phase 4: Diffusion Decoder

**Priority**: MEDIUM  
**Estimated time**: 2-3 weeks  
**Goal**: Replace VQ-VAE decoder with conditional diffusion

### 4.1 Architecture

```
Current:  Tokens → VQ-VAE Decoder → Frame
New:      Tokens → Condition → Diffusion U-Net → Frame
                                    ↑
                              4-8 denoising steps
```

### 4.2 Implementation Tasks

```
□ 4.2.1  Create small U-Net conditioned on token embeddings
□ 4.2.2  Implement DDPM/DDIM training
□ 4.2.3  Train decoder on (tokens, frame) pairs from existing data
□ 4.2.4  Tune step count for speed/quality tradeoff
□ 4.2.5  A/B test visual quality
```

---

## Phase 5: Latent Diffusion Dynamics

**Priority**: LOW (major architecture shift)  
**Estimated time**: 4+ weeks  
**Goal**: Replace transformer token prediction with DiT in latent space

### 5.1 Architecture (Oasis-style)

```
Frame → VAE Encoder → Latent (32×32×4)
                           ↓
              Diffusion Transformer (DiT)
              [latent_noisy, action, timestep] → noise_pred
                           ↓
                    Denoise (4-8 steps)
                           ↓
               VAE Decoder → Next Frame
```

### 5.2 This is a bigger rewrite - defer until Phases 1-4 complete

---

## Phase 6: Multi-Game Training

**Priority**: LOW  
**Estimated time**: Ongoing  
**Goal**: Single model that works across multiple games

### 6.1 Requirements

```
□ 6.1.1  Game-agnostic action representation
□ 6.1.2  Multiple game data loaders
□ 6.1.3  Game conditioning token/embedding
□ 6.1.4  Larger model capacity
```

---

## Immediate Next Steps

### Week 1: Setup

```
□ Create v2/ directory for new development
□ Copy working v1 files as starting point
□ Set up comparison infrastructure (v1 vs v2 metrics)
□ Create temporal dataset loader
```

### Week 2: Temporal v1.1

```
□ Implement TemporalVisualWorldModel
□ Train with 4-frame history
□ Evaluate and compare
□ Document findings
```

---

## File Structure (Proposed)

```
world_model/
├── v1/                          # Current working model (frozen)
│   ├── vqvae.py
│   ├── visual_world_model.py
│   └── ...
│
├── v2/                          # Next-gen development
│   ├── models/
│   │   ├── vqvae.py            # Copy from v1, may modify later
│   │   ├── temporal_world_model.py   # Phase 1
│   │   ├── causal_world_model.py     # Phase 2
│   │   └── diffusion_decoder.py      # Phase 4
│   ├── data/
│   │   ├── temporal_dataset.py  # Frame history support
│   │   └── game_env.py          # Upgraded environment
│   ├── train/
│   │   ├── train_temporal.py
│   │   └── train_diffusion.py
│   └── eval/
│       ├── compare_versions.py
│       └── metrics.py
│
├── checkpoints/
│   ├── v1/                      # v1 weights (don't touch)
│   └── v2/                      # v2 experiments
│
└── ROADMAP.md                   # This file
```

---

## Success Metrics

| Metric | v1.0 | v2.0 Target |
|--------|------|-------------|
| Token Accuracy | 42% | 75%+ |
| Visual Quality (subjective) | Good | Excellent |
| Temporal Consistency | Poor | Good |
| Resolution | 64×64 | 128×128+ |
| Real-time FPS | 30+ | 20+ |
| Model Size | 11M | 50-100M |

---

## Risk Mitigation

1. **Never modify v1/** - Only add new code in v2/
2. **Version checkpoints** - v1/model.pt, v2/model.pt, etc.
3. **Comparison scripts** - Always measure before/after
4. **Incremental commits** - One feature per PR/commit
5. **Regression tests** - Ensure v1 still works after each change

---

## Ready to Start?

Phase 1.1.1: Create the temporal world model architecture.

Switch to Agent mode and say "start phase 1" to begin.




