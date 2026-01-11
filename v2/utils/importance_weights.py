"""
Token Importance Weighting (v2.0) + Focal Loss (v3.0)
=====================================================
Game-agnostic importance weighting for world model training.

v2.0 Signals:
- Embedding-distance motion (robust to codebook jitter)
- Eventness spike detection (catches "fire" moments)
- Multi-frame persistence (stabilizes tracking)
- Percentile-based safe capping (prevents context starvation)

v3.0 Focal Loss:
- Automatically upweights tokens the model struggles with
- Self-adjusting: no manual detection of "important" tokens needed
- Composes multiplicatively with motion weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_focal_components(
    logits: torch.Tensor,  # (B, N, vocab) or (B*N, vocab)
    targets: torch.Tensor,  # (B, N) or (B*N,)
    gamma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """
    Shared focal loss computation - returns log_p_t, focal_weight, original_shape.
    
    Uses single log_softmax for numerical stability.
    """
    original_shape = targets.shape
    if logits.dim() == 3:
        B, N, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
    else:
        logits_flat = logits
        targets_flat = targets.reshape(-1)
    
    # Single log_softmax for numerical stability
    log_probs = F.log_softmax(logits_flat, dim=-1)  # (B*N, vocab)
    
    # Get log probability of correct class
    log_p_t = log_probs.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)  # (B*N,)
    
    # Convert to probability for focal weight (stable since log_p_t <= 0)
    p_t = log_p_t.exp()  # (B*N,)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    return log_p_t, focal_weight, original_shape


def focal_loss_per_token(
    logits: torch.Tensor,       # (B, N, vocab) or (B*N, vocab)
    targets: torch.Tensor,      # (B, N) or (B*N,)
    gamma: float = 2.0,         # Focusing parameter (higher = more focus on hard examples)
    reduction: str = 'none',    # 'none' returns per-token loss
) -> torch.Tensor:
    """
    Focal Loss for token prediction (v3.0) - numerically stable version.
    
    Standard CE:    -log(p_t)
    Focal Loss:     -(1 - p_t)^γ * log(p_t)
    
    When model is confident and correct (p=0.95): weight = (0.05)^2 = 0.0025
    When model is uncertain (p=0.50): weight = (0.50)^2 = 0.25 → 100x more!
    
    Args:
        logits: Raw model output before softmax
        targets: Ground truth token indices
        gamma: Focusing parameter (default 2.0, higher = more focus on hard)
        reduction: 'none', 'mean', or 'sum'
        
    Returns:
        Focal loss (per-token if reduction='none')
    """
    log_p_t, focal_weight, original_shape = _compute_focal_components(logits, targets, gamma)
    
    # Focal loss: -(1 - p_t)^gamma * log(p_t)
    focal = -focal_weight * log_p_t
    
    # Reshape back to original
    focal = focal.reshape(original_shape)
    
    if reduction == 'mean':
        return focal.mean()
    elif reduction == 'sum':
        return focal.sum()
    return focal


def focal_loss_with_motion_weights(
    logits: torch.Tensor,           # (B, N, vocab)
    targets: torch.Tensor,          # (B, N)
    motion_weights: torch.Tensor,   # (B, N) from compute_hybrid_importance_weights
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Combine focal loss with motion-based importance weights.
    
    Total weight = focal_weight * motion_weight, normalized per-sample.
    
    Key fix: Normalizes combined weights per-sample to prevent gradient scale
    drift as model improves (focal shrinks easy tokens drastically).
    
    This gives maximum upweighting to tokens that are:
    1. Hard to predict (focal loss detects this)
    2. In motion (motion weights detect this)
    """
    B, N = targets.shape
    
    # Reuse shared computation
    log_p_t, focal_weight, _ = _compute_focal_components(logits, targets, gamma)
    log_p_t = log_p_t.reshape(B, N)
    focal_weight = focal_weight.reshape(B, N)
    
    # Combined weight = focal * motion
    combined_weight = focal_weight * motion_weights  # (B, N)
    
    # Normalize per-sample to keep gradient scale stable
    combined_weight = combined_weight / (combined_weight.mean(dim=1, keepdim=True) + 1e-8)
    
    # Apply to negative log likelihood
    loss_per_token = -log_p_t * combined_weight  # (B, N)
    
    return loss_per_token.mean()


def compute_hybrid_importance_weights(
    target_tokens: torch.Tensor,      # (B, N)
    history_tokens: torch.Tensor,     # (B, T, N)
    token_embedding: nn.Embedding,    # VQ-VAE codebook (vqvae.quantizer.embedding)
    device: torch.device,
    motion_scale: float = 2.0,        # Embedding-distance motion weight
    eventness_scale: float = 2.0,     # Spike detection weight (applied to movers only)
    persistence_scale: float = 1.0,   # Multi-frame persistence weight
    max_ratio: float = 6.0,           # Safe cap: 95th percentile / median ≤ max_ratio
    min_movers_for_eventness: int = 4,  # Minimum moved tokens to compute eventness
    mover_top_frac: float = 0.02,     # v2.3: top fraction of tokens considered "movers"
) -> torch.Tensor:
    """
    Hybrid importance weighting (v2.3) - robust mover detection.
    
    v2.3 fixes:
    - Mover selection uses top-k AND absolute threshold (not just "top 10%")
    - Eventness uses blended mover mask (moved_now OR moved_prev) for stability
    - Persistence uses per-step threshold for consistency across frames
    
    IMPORTANT: Pass VQ-VAE codebook (vqvae.quantizer.embedding), NOT transformer embeddings.
    VQ-VAE codebook is trained for visual similarity and is stable from epoch 0.
    
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
        
        # v2.3: Use 75th percentile for normalization (robust to mostly-static frames)
        p75 = torch.quantile(motion_mag, 0.75, dim=1, keepdim=True).clamp(min=1e-3)
        motion_norm = motion_mag / p75
        
        # v2.3 fix: Mover detection with top-k AND absolute threshold
        # This prevents "top 10% no matter what" - only picks real movers
        k = max(min_movers_for_eventness, int(mover_top_frac * N))  # e.g., 2% of tokens, at least 4
        topk_vals, _ = motion_mag.topk(k, dim=1)  # (B, k)
        topk_threshold = topk_vals[:, -1:]  # kth largest value per sample
        
        # Absolute minimum: must be above 50% of p75 to be a "real" mover
        abs_threshold = (p75 * 0.5).clamp(min=1e-4)
        
        # Final threshold: max of top-k cutoff and absolute minimum
        motion_threshold = torch.maximum(topk_threshold, abs_threshold)  # (B, 1)
        moved_mask = (motion_mag >= motion_threshold).float()  # (B, N)
        
        # Count movers per sample for gating eventness
        n_movers = moved_mask.sum(dim=1)  # (B,)
        
        # === 2. Eventness Spike Detection (gated, blended mask) ===
        eventness_weight = torch.zeros(B, N, device=device)
        
        if T >= 2:
            prev2_emb = token_embedding(history_tokens[:, -2])
            hist_motion_mag = torch.norm(prev_emb - prev2_emb, dim=-1)
            
            # v2.3 fix: Use blended mover mask (moved_now OR moved_prev)
            # This stabilizes eventness when the moving set changes rapidly
            hist_topk_vals, _ = hist_motion_mag.topk(k, dim=1)
            hist_topk_threshold = hist_topk_vals[:, -1:]
            hist_threshold = torch.maximum(hist_topk_threshold, abs_threshold)
            hist_moved_mask = (hist_motion_mag >= hist_threshold).float()
            
            # Blended mask: tokens that moved in either step
            blended_mask = torch.clamp(moved_mask + hist_moved_mask, max=1.0)
            n_blended = blended_mask.sum(dim=1)
            
            has_movers = (n_blended >= min_movers_for_eventness)  # (B,)
            
            if has_movers.any():
                # Compare current vs historical motion using blended mask
                sample_motion = (motion_mag * blended_mask).sum(dim=1) / (n_blended + 1e-6)
                hist_motion = (hist_motion_mag * blended_mask).sum(dim=1) / (n_blended + 1e-6)
                
                # Spike ratio: high = "something just happened"
                eventness = (sample_motion / (hist_motion + 1e-6)).clamp(min=0.5, max=3.0)
                eventness = (eventness - 1.0)  # Center at 0, range [-0.5, 2.0]
                
                # Zero out eventness for samples without enough movers
                eventness = eventness * has_movers.float()
                
                # Apply eventness to blended mask (tokens that moved in either step)
                eventness_weight = eventness.unsqueeze(1) * blended_mask  # (B, N)
        
        # === 3. Persistence (multi-frame motion consistency, per-step threshold) ===
        persistence = torch.zeros(B, N, device=device)
        
        if T >= 2:
            prev_frame_emb = token_embedding(history_tokens[:, -1])
            
            for t in range(T - 2, -1, -1):
                curr_frame_emb = token_embedding(history_tokens[:, t])
                frame_motion = torch.norm(prev_frame_emb - curr_frame_emb, dim=-1)
                
                # v2.3 fix: Compute per-step threshold (consistent within each step)
                step_p75 = torch.quantile(frame_motion, 0.75, dim=1, keepdim=True).clamp(min=1e-3)
                step_topk_vals, _ = frame_motion.topk(k, dim=1)
                step_topk_thr = step_topk_vals[:, -1:]
                step_abs_thr = (step_p75 * 0.5).clamp(min=1e-4)
                step_threshold = torch.maximum(step_topk_thr, step_abs_thr)
                
                moved = (frame_motion >= step_threshold).float()
                decay = 0.7 ** (T - 1 - t)
                persistence = persistence + moved * decay
                
                prev_frame_emb = curr_frame_emb
            
            persistence = persistence / (T - 1)
        
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
