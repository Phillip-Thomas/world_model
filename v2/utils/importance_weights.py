"""
Token Importance Weighting (v2.0)
=================================
Game-agnostic importance weighting for world model training.

Combines multiple signals:
- Embedding-distance motion (robust to codebook jitter)
- Eventness spike detection (catches "fire" moments)
- Multi-frame persistence (stabilizes tracking)
- Percentile-based safe capping (prevents context starvation)
"""

import torch
import torch.nn as nn


def compute_hybrid_importance_weights(
    target_tokens: torch.Tensor,      # (B, N)
    history_tokens: torch.Tensor,     # (B, T, N)
    token_embedding: nn.Embedding,    # Model's token embedding layer
    device: torch.device,
    motion_scale: float = 2.0,        # Embedding-distance motion weight
    eventness_scale: float = 2.0,     # Spike detection weight
    persistence_scale: float = 1.0,   # Multi-frame persistence weight
    max_ratio: float = 6.0,           # Safe cap: 95th percentile / median ≤ max_ratio
) -> torch.Tensor:
    """
    Hybrid importance weighting (v2.0) - game-agnostic, robust.
    
    Combines:
    1. Embedding-distance motion: L2 distance in embedding space (not token ID)
       - Robust to codebook neighbor jitter
       - Captures magnitude of visual change
    
    2. Eventness spike detection: sample-level motion spike vs recent history
       - Upweights frames with unusual motion (ball fire, explosions, spawns)
       - Helps model learn rare but critical transitions
    
    3. Persistence: tokens that move in multiple consecutive frames
       - Distinguishes real movers (ball) from jitter
       - Smooth decay to avoid abrupt weight changes
    
    4. Safe percentile capping: prevents extreme ratios that starve context
       - Caps at max_ratio × median instead of hard max
       - Preserves relative importance while preventing instability
    
    Returns: (B, N) weight tensor (mean ~1.0 per sample)
    """
    B, N = target_tokens.shape
    T = history_tokens.shape[1]
    
    with torch.no_grad():
        # === 1. Embedding-Distance Motion ===
        target_emb = token_embedding(target_tokens)           # (B, N, D)
        prev_emb = token_embedding(history_tokens[:, -1])     # (B, N, D)
        
        # L2 distance in embedding space
        motion_mag = torch.norm(target_emb - prev_emb, dim=-1)  # (B, N)
        
        # Robust normalize: divide by median (not mean) to reduce outlier sensitivity
        motion_median = motion_mag.median(dim=1, keepdim=True).values.clamp(min=1e-6)
        motion_norm = motion_mag / motion_median
        
        # === 2. Eventness Spike Detection ===
        sample_motion = motion_mag.mean(dim=1)  # (B,)
        
        if T >= 2:
            prev2_emb = token_embedding(history_tokens[:, -2])
            hist_motion_mag = torch.norm(prev_emb - prev2_emb, dim=-1)
            hist_motion = hist_motion_mag.mean(dim=1)
            
            # Spike ratio: high = "something just happened"
            eventness = (sample_motion / (hist_motion + 1e-6)).clamp(min=0.5, max=3.0)
            eventness = (eventness - 1.0)  # Center at 0
        else:
            eventness = torch.zeros(B, device=device)
        
        eventness_weight = eventness.unsqueeze(1).expand(B, N)
        
        # === 3. Persistence (multi-frame motion consistency) ===
        if T >= 2:
            persistence = torch.zeros(B, N, device=device)
            prev_frame_emb = token_embedding(history_tokens[:, -1])
            
            for t in range(T - 2, -1, -1):
                curr_frame_emb = token_embedding(history_tokens[:, t])
                frame_motion = torch.norm(prev_frame_emb - curr_frame_emb, dim=-1)
                
                moved = (frame_motion > motion_median * 0.5).float()
                decay = 0.7 ** (T - 1 - t)
                persistence += moved * decay
                
                prev_frame_emb = curr_frame_emb
            
            persistence = persistence / (T - 1)
        else:
            persistence = torch.zeros(B, N, device=device)
        
        # === Combine Signals ===
        weights = (
            1.0
            + motion_scale * (motion_norm - 1.0).clamp(min=0)
            + eventness_scale * eventness_weight.clamp(min=0)
            + persistence_scale * persistence
        )
        
        # === Safe Percentile-Based Capping ===
        p50 = weights.median(dim=1, keepdim=True).values.clamp(min=0.5)
        cap = p50 * max_ratio
        weights = torch.clamp(weights, min=0.5)
        weights = torch.minimum(weights, cap.expand_as(weights))
        
        # Normalize mean=1 per sample
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)
    
    return weights


def compute_token_importance_weights(
    target_tokens: torch.Tensor,
    history_tokens: torch.Tensor,
    device: torch.device,
    base_weight: float = 1.0,
    motion_weight: float = 2.0,
    continuous_bonus: float = 0.5,
    max_weight: float = 4.0,
    token_embedding: nn.Embedding = None,
    use_hybrid: bool = True,
    motion_scale: float = 2.0,
    eventness_scale: float = 2.0,
    persistence_scale: float = 1.0,
    max_ratio: float = 6.0,
) -> torch.Tensor:
    """
    Token importance weighting - dispatches to hybrid (v2.0) or legacy (v1.x).
    """
    if use_hybrid and token_embedding is not None:
        return compute_hybrid_importance_weights(
            target_tokens, history_tokens, token_embedding, device,
            motion_scale=motion_scale,
            eventness_scale=eventness_scale,
            persistence_scale=persistence_scale,
            max_ratio=max_ratio,
        )
    
    # Legacy v1.x behavior
    B, N = target_tokens.shape
    
    with torch.no_grad():
        prev_tokens = history_tokens[:, -1, :]
        moved_now = (target_tokens != prev_tokens).float()
        
        if history_tokens.shape[1] >= 2:
            prev2_tokens = history_tokens[:, -2, :]
            moved_prev = (prev_tokens != prev2_tokens).float()
            moved = torch.clamp(moved_now + moved_prev, 0, 1)
            continuous_motion = moved_now * moved_prev
        else:
            moved = moved_now
            continuous_motion = torch.zeros_like(moved_now)
        
        weights = base_weight + (motion_weight - base_weight) * moved + continuous_bonus * continuous_motion
        weights = weights.clamp(min=base_weight, max=max_weight)
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)
        
    return weights
