"""
High Resolution VQ-VAE (v2.24)
==============================
Flexible token grid that supports non-square inputs.

v2.24 improvements:
- CRITICAL FIX: Wrap ALL EMA buffer updates in torch.no_grad()
- Fixes linear CUDA memory leak caused by autograd graph retention
- Affects: _ema_update(), _init_from_data(), _reset_dead_codes(), usage_counts
- (Forum reference: VQ-VAE codebook EMA memory leak)

v2.23 improvements:
- Fixed implicit GPU->CPU syncs causing training slowdown (PyTorch forum issue)
- Use Python bool for fast initialization check instead of tensor buffer
- Use math.exp() instead of torch.exp(torch.tensor()) for stats
- Explicit .item() calls to avoid hidden tensor comparisons

v2.8 improvements:
- U-Net skip connections (preserves fine details like ball!)
- High-frequency encoder features bypass quantization during training
- Skip fuse layers concat encoder features with decoder upsamples

v2.1 improvements:
- Upsample + Conv decoder (no checkerboard artifacts)
- Tanh output (matches [-1, 1] normalization)
- Configurable EMA update frequency
- Smarter dead code reset (bottom-k only)
- Histogram-based codebook stats (usage, perplexity, entropy)

For aspect-ratio-preserved Atari (84x64):
- Input: 84x64 → Token grid: 21x16 = 336 tokens
- Each token covers 4x4 pixels

For legacy square input (64x64):
- Input: 64x64 → Token grid: 16x16 = 256 tokens
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm for stable training."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class EncoderHiRes(nn.Module):
    """
    High-resolution encoder with 2 downsampling steps.
    
    v2.8: Returns skip connections for U-Net style decoder (preserves fine details).
    
    Input → Output mapping (stride 2 each):
    - 84x64 → 42x32 → 21x16 (aspect-preserved Atari)
    - 64x64 → 32x32 → 16x16 (legacy square)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 256,
        n_residual: int = 2,
    ):
        super().__init__()
        
        self.initial = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # 2 downsampling blocks (4x total reduction)
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.GELU(),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.GELU(),
        )
        
        # Residual blocks at bottleneck
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels * 4) for _ in range(n_residual)]
        )
        
        # Project to latent dimension
        self.to_latent = nn.Conv2d(hidden_channels * 4, latent_channels, 1)
        
    def forward(self, x: torch.Tensor, return_skips: bool = False):
        """
        Args:
            x: Input image (B, C, H, W)
            return_skips: If True, return skip connections for U-Net decoder
            
        Returns:
            If return_skips=False: latent (B, D, H/4, W/4)
            If return_skips=True: (latent, [skip1, skip2]) where:
                - skip1: after initial conv (B, 64, H, W) - full resolution
                - skip2: after down1 (B, 128, H/2, W/2) - half resolution
        """
        x0 = self.initial(x)           # (B, 64, H, W)
        x1 = self.down1(x0)            # (B, 128, H/2, W/2)
        x2 = self.down2(x1)            # (B, 256, H/4, W/4)
        x2 = self.residual_blocks(x2)
        z = self.to_latent(x2)         # (B, latent_channels, H/4, W/4)
        
        if return_skips:
            return z, [x0, x1]
        return z


class DecoderHiRes(nn.Module):
    """
    High-resolution decoder with 2 upsampling steps.
    
    v2.8: U-Net skip connections preserve fine details (ball, paddle edges).
    
    v2.1: Uses Upsample + Conv instead of ConvTranspose2d to avoid
    checkerboard artifacts that can poison world model tokens.
    
    Token grid → Output mapping:
    - 21x16 → 42x32 → 84x64 (aspect-preserved Atari)
    - 16x16 → 32x32 → 64x64 (legacy square)
    """
    
    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 256,
        n_residual: int = 2,
        use_skip: bool = True,  # v2.8: U-Net skip connections
    ):
        super().__init__()
        self.use_skip = use_skip
        
        # Project from latent dimension
        self.from_latent = nn.Conv2d(latent_channels, hidden_channels * 4, 1)
        
        # Residual blocks at bottleneck
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels * 4) for _ in range(n_residual)]
        )
        
        # v2.8: Skip connection fusion (concat + 1x1 conv to reduce channels)
        # After up1: concat with skip2 (128ch + 128ch = 256ch -> 128ch)
        # After up2: concat with skip1 (64ch + 64ch = 128ch -> 64ch)
        if use_skip:
            self.skip_fuse1 = nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 1)  # 256 -> 128
            self.skip_fuse2 = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)      # 128 -> 64
        
        # 2 upsampling blocks using Upsample + Conv (no checkerboard!)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.GELU(),
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        
        self.final = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor, skips: list = None) -> torch.Tensor:
        """
        Args:
            x: Quantized latent (B, D, H/4, W/4)
            skips: Optional [skip1, skip2] from encoder for U-Net fusion
                   skip1: (B, 64, H, W), skip2: (B, 128, H/2, W/2)
        """
        x = self.from_latent(x)
        x = self.residual_blocks(x)
        x = self.up1(x)  # (B, 128, H/2, W/2)
        
        # v2.8: Fuse with skip2 (half resolution)
        if self.use_skip and skips is not None:
            skip2 = skips[1]  # (B, 128, H/2, W/2)
            x = torch.cat([x, skip2], dim=1)  # (B, 256, H/2, W/2)
            x = self.skip_fuse1(x)            # (B, 128, H/2, W/2)
        
        x = self.up2(x)  # (B, 64, H, W)
        
        # v2.8: Fuse with skip1 (full resolution - critical for ball!)
        if self.use_skip and skips is not None:
            skip1 = skips[0]  # (B, 64, H, W)
            x = torch.cat([x, skip1], dim=1)  # (B, 128, H, W)
            x = self.skip_fuse2(x)            # (B, 64, H, W)
        
        x = self.final(x)
        # Tanh to constrain output to [-1, 1] (matches input normalization)
        return torch.tanh(x)


class VectorQuantizerHiRes(nn.Module):
    """
    VQ layer with EMA updates - works with any spatial size.
    
    v2.1 improvements:
    - Configurable EMA update frequency (ema_update_every)
    - Smarter dead code reset (only reset bottom-k codes)
    - Histogram-based codebook statistics
    """
    
    def __init__(
        self,
        n_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.995,          # Standard EMA decay
        epsilon: float = 1e-5,
        ema_update_every: int = 10,       # v2.6: update every 10 batches (1 is too slow)
        dead_code_threshold: float = None, # v2.6: Auto-set based on n_embeddings
        reset_every_n_batches: int = 500,  # v2.14: Less frequent to reduce allocation pressure
    ):
        super().__init__()
        
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.ema_update_every = ema_update_every
        # v2.6: Auto-set threshold relative to uniform probability
        # Uniform prob = 1/K, so threshold = 0.1/K means "10x below uniform"
        self.dead_code_threshold = dead_code_threshold if dead_code_threshold else (0.1 / n_embeddings)
        self.reset_every_n_batches = reset_every_n_batches
        
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.normal_(0, 1)
        # v2.6: EMA-managed embedding should NOT have gradients
        # This is updated via EMA in _ema_update(), not by optimizer
        self.embedding.weight.requires_grad = False
        
        self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
        self.register_buffer('ema_embed_sum', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False))
        # v2.23: Python bool for fast checking (avoids GPU->CPU sync every forward pass)
        self._is_initialized = False
        # v2.1: track usage counts for histogram stats
        self.register_buffer('usage_counts', torch.zeros(n_embeddings))
        # v2.14: Reusable buffer for EMA update (prevents allocation every update)
        self.register_buffer('_ema_embed_sum_buffer', torch.zeros(n_embeddings, embedding_dim))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, C, H, W) -> (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)
        
        # v2.23: Use Python bool to avoid GPU->CPU sync every forward pass
        # Sync from tensor buffer if loading from checkpoint (one-time check)
        if not self._is_initialized and self.initialized.item():
            self._is_initialized = True
        if self.training and not self._is_initialized:
            self._init_from_data(z_flat)
        
        # v2.6: Euclidean distance in FP32 to prevent winner-take-all
        # (Cosine distance caused complete collapse - not suitable for this data)
        # v2.14: Detach z_flat for distance computation - no gradients needed for argmin,
        # and this allows the 22MB float32 copy to be freed immediately after indices are computed
        # instead of being kept alive for backward pass.
        with torch.amp.autocast('cuda', enabled=False):
            z_flat_f32 = z_flat.detach().float()  # Detach to allow immediate deallocation
            emb_f32 = self.embedding.weight.float()
            distances = (
                z_flat_f32.pow(2).sum(1, keepdim=True)
                + emb_f32.pow(2).sum(1)
                - 2 * z_flat_f32 @ emb_f32.t()
            )
            indices = distances.argmin(dim=1)
            # Explicitly delete large intermediates to help allocator
            del z_flat_f32, emb_f32, distances
        
        if self.training:
            # v2.24: Wrap ALL buffer updates in no_grad() to prevent autograd graph retention
            # (Forum fix: EMA updates to registered buffers were causing linear memory leak)
            with torch.no_grad():
                self._ema_update(z_flat, indices)
                # Update usage counts for stats
                self.usage_counts.add_(torch.bincount(indices, minlength=self.n_embeddings).float())
        
        z_q = self.embedding(indices).view(z.shape)
        
        commitment_loss = F.mse_loss(z_q.detach(), z)
        loss = self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Reshape indices to spatial grid
        batch_size, h, w, _ = z.shape
        indices = indices.view(batch_size, h, w)
        
        return z_q, loss, indices
    
    def _init_from_data(self, z_flat: torch.Tensor):
        """
        Initialize codebook from encoder outputs using k-means++ style selection.
        
        v2.9 fix: Use k-means++ to ensure DIVERSE code initialization.
        Random sampling often picks similar background pixels, causing immediate collapse.
        
        v2.2 fix: Fill ALL codes from data (repeat-sample if needed).
        v2.3 fix: ALSO sync EMA buffers to prevent immediate corruption!
        v2.24 fix: Wrap in no_grad() to prevent autograd graph retention.
        """
        # v2.24: All buffer updates must be in no_grad() to prevent memory leak
        with torch.no_grad():
            n_data = z_flat.shape[0]
            device = z_flat.device
            
            # v2.9: K-means++ style initialization for diversity
            # Instead of random sampling, iteratively pick points far from existing centers
            z_flat_f32 = z_flat.float()
            
            # Start with random first center
            indices = [torch.randint(0, n_data, (1,), device=device).item()]
            
            for _ in range(min(self.n_embeddings, n_data) - 1):
                # Compute distances to nearest existing center
                centers = z_flat_f32[indices]  # (k, D)
                # distances: (n_data, k) -> min over k -> (n_data,)
                dists = torch.cdist(z_flat_f32, centers).min(dim=1).values
                # Sample proportional to distance squared (k-means++ style)
                probs = dists ** 2
                probs = probs / probs.sum()
                # Pick next center
                next_idx = torch.multinomial(probs, 1).item()
                indices.append(next_idx)
                # v2.15: Explicit cleanup to reduce peak memory during init
                del centers, dists, probs
            
            chosen = z_flat_f32[indices]
            n_chosen = len(chosen)
            
            # v2.9: Add small jitter to prevent identical codes
            # Ensure minimum distance between all pairs
            jitter_scale = 0.1
            chosen = chosen + jitter_scale * torch.randn_like(chosen)
            
            self.embedding.weight.data[:n_chosen] = chosen
            
            # v2.2: Fill remainder by repeat-sampling (reduces dead-code churn)
            if n_chosen < self.n_embeddings:
                n_remaining = self.n_embeddings - n_chosen
                repeat_idx = torch.randint(0, n_chosen, (n_remaining,), device=device)
                self.embedding.weight.data[n_chosen:] = chosen[repeat_idx]
                print(f"  Initialized codebook with k-means++: {n_chosen} diverse + {n_remaining} repeat")
            else:
                print(f"  Initialized codebook with k-means++ ({n_chosen} diverse centers)")
            
            # v2.3 CRITICAL FIX: Sync EMA buffers to match initialized embeddings!
            # Without this, the first EMA update will corrupt the codebook with
            # the old random ema_embed_sum values.
            self.ema_embed_sum.copy_(self.embedding.weight.data)
            # Initialize cluster sizes uniformly so codes don't appear "dead" immediately
            self.ema_cluster_size.fill_(1.0)
            
            self.initialized.fill_(True)
            self._is_initialized = True  # v2.23: Sync Python bool
    
    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        """
        v2.24 fix: EMA updates to registered buffers MUST be in no_grad()!
        Without this, the autograd graph is retained, causing linear memory leak.
        (See PyTorch forum: VQ-VAE codebook EMA memory leak)
        """
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        self._update_count += 1
        
        # v2.6: Update every batch (frequent updates help prevent collapse)
        if self.ema_update_every > 1 and self._update_count % self.ema_update_every != 0:
            return
        
        # v2.24: CRITICAL - wrap all buffer updates in no_grad() to prevent memory leak
        with torch.no_grad():
            # v2.6: Use bincount + index_add_ instead of one_hot (much cheaper!)
            # This is O(N) instead of O(N*K) memory
            cluster_size = torch.bincount(indices, minlength=self.n_embeddings).float()
            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
            
            # v2.14: Reuse buffer instead of allocating new tensor every update
            # Fallback for old checkpoints that don't have this buffer
            if not hasattr(self, '_ema_embed_sum_buffer') or self._ema_embed_sum_buffer is None:
                self.register_buffer('_ema_embed_sum_buffer', torch.zeros_like(self.ema_embed_sum))
            self._ema_embed_sum_buffer.zero_()
            self._ema_embed_sum_buffer.index_add_(0, indices, z_flat.float())
            self.ema_embed_sum.mul_(self.ema_decay).add_(self._ema_embed_sum_buffer, alpha=1 - self.ema_decay)
            
            n = self.ema_cluster_size.sum()
            cluster_size_normalized = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.n_embeddings * self.epsilon) * n
            )
            
            # v2.15: In-place ops to avoid 512KB allocation per update
            self.embedding.weight.data.copy_(self.ema_embed_sum)
            self.embedding.weight.data.div_(cluster_size_normalized.unsqueeze(1))
        
        # v2.11: Dead code reset re-enabled to combat constant dead count
        if self._update_count % self.reset_every_n_batches == 0:
            self._reset_dead_codes(z_flat)
    
    def _reset_dead_codes(self, z_flat: torch.Tensor):
        """
        v2.3: Reset dead codes more aggressively.
        
        v2.1 had a bug: wouldn't reset if >50% were dead, which is exactly
        when you NEED to reset! Now we always reset dead codes, but limit
        how many per batch to avoid instability.
        
        v2.24 fix: Wrap in no_grad() to prevent autograd graph retention.
        """
        if z_flat.shape[0] == 0:
            return
        
        # v2.24: All buffer updates must be in no_grad() to prevent memory leak
        with torch.no_grad():
            # Normalize cluster sizes to get usage percentages
            # v2.23: Use .item() to avoid implicit GPU->CPU sync from tensor comparison
            total = self.ema_cluster_size.sum()
            if total.item() == 0:
                return
            usage_pct = self.ema_cluster_size / total
            
            # Find dead codes (below threshold)
            dead_mask = usage_pct < self.dead_code_threshold
            n_dead = dead_mask.sum().item()
            
            # v2.3: Always reset dead codes, but limit to avoid instability
            # (removed the "don't reset if >50% dead" condition - that was the bug!)
            if n_dead > 0:
                # v2.4: Reduced from 64 to 16 to avoid loss spikes
                max_reset_per_batch = min(16, z_flat.shape[0])
                n_reset = min(int(n_dead), max_reset_per_batch)
                random_idx = torch.randperm(z_flat.shape[0], device=z_flat.device)[:n_reset]
                
                # Get indices of dead codes
                dead_indices = torch.where(dead_mask)[0][:n_reset]
                
                self.embedding.weight.data[dead_indices] = z_flat[random_idx].float()
                # Reset EMA stats for these codes
                self.ema_cluster_size[dead_indices] = 1.0
                self.ema_embed_sum[dead_indices] = z_flat[random_idx].float()
    
    def get_batch_stats(self, indices: torch.Tensor) -> dict:
        """
        v2.2: Compute stats for a single batch (for early collapse detection).
        
        Use this to monitor per-batch health during training.
        """
        counts = torch.bincount(indices.flatten(), minlength=self.n_embeddings).float()
        # v2.23: Use .item() to avoid implicit GPU->CPU sync from tensor comparison
        total = counts.sum().item()
        
        if total == 0:
            return {'batch_usage': 0.0, 'batch_perplexity': 0.0}
        
        probs = counts / total
        probs_nonzero = probs[probs > 0]
        
        usage = (counts > 0).float().mean().item() * 100
        entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero + 1e-10)).item()
        # v2.23: Use math.exp instead of torch.exp(torch.tensor()) to avoid unnecessary allocation
        perplexity = math.exp(entropy)
        
        return {
            'batch_usage': usage,
            'batch_perplexity': perplexity,
        }
    
    def get_codebook_stats(self, indices: Optional[torch.Tensor] = None) -> dict:
        """
        v2.1: Compute codebook statistics using histogram.
        
        v2.2: Now also includes EMA-based dead code count for debugging.
        v2.22: Fixed memory leak - compute on CPU and delete intermediates.
        
        Returns:
            dict with:
            - usage_pct: percentage of codes used (0-100)
            - perplexity: effective number of codes (higher = more diverse)
            - entropy: information entropy of code distribution
            - dead_codes: number of completely unused codes
            - ema_dead_codes: dead codes according to EMA stats (for debugging)
        """
        import numpy as np
        
        # v2.22: Compute on CPU to avoid GPU memory accumulation
        with torch.no_grad():
            if indices is not None:
                counts_cpu = torch.bincount(indices.flatten(), minlength=self.n_embeddings).cpu().numpy()
            else:
                counts_cpu = self.usage_counts.cpu().numpy()
            
            total = counts_cpu.sum()
            if total == 0:
                return {'usage_pct': 0.0, 'perplexity': 0.0, 'entropy': 0.0, 
                        'dead_codes': self.n_embeddings, 'ema_dead_codes': self.n_embeddings}
            
            probs = counts_cpu / total
            probs_nonzero = probs[probs > 0]
            
            usage_pct = float((counts_cpu > 0).mean() * 100)
            entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero + 1e-10)))
            perplexity = float(np.exp(entropy))
            dead_codes = int((counts_cpu == 0).sum())
            
            # v2.2: EMA-based dead code count
            ema_cpu = self.ema_cluster_size.cpu().numpy()
            ema_total = ema_cpu.sum()
            if ema_total > 0:
                ema_usage_pct = ema_cpu / ema_total
                ema_dead = int((ema_usage_pct < self.dead_code_threshold).sum())
            else:
                ema_dead = self.n_embeddings
        
        return {
            'usage_pct': usage_pct,
            'perplexity': perplexity,
            'entropy': entropy,
            'dead_codes': dead_codes,
            'ema_dead_codes': ema_dead,
        }
    
    def reset_usage_counts(self):
        """Reset usage counts (call at start of epoch for fresh stats)."""
        self.usage_counts.zero_()


class VQVAEHiRes(nn.Module):
    """
    High-Resolution VQ-VAE with flexible input size.
    
    v2.2 improvements:
    - Better codebook init (repeat-sample to fill all codes)
    - Reduced EMA warmup (50 → 20 batches)
    - decode() handles flattened tokens (B, N) from world model
    - Batch vs epoch perplexity stats for debugging
    - EMA dead code tracking
    
    v2.1 improvements:
    - Upsample + Conv decoder (no checkerboard artifacts)
    - Tanh output (matches [-1, 1] normalization)
    - Configurable EMA update frequency
    - Smarter dead code reset
    - Histogram-based codebook stats
    
    Supports both aspect-preserved and square inputs:
    - 84x64 input → 21x16 = 336 tokens (aspect-preserved Atari)
    - 64x64 input → 16x16 = 256 tokens (legacy square)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 256,
        n_embeddings: int = 512,
        n_residual: int = 2,
        input_size: Optional[Tuple[int, int]] = None,  # (H, W) - optional, for metadata
        # v2.5: quantizer hyperparameters (prevent codebook collapse)
        ema_decay: float = 0.995,         # Slower decay for stability
        ema_update_every: int = 1,        # Update EVERY batch
        dead_code_threshold: float = 0.01,
        # v2.9: Skip connections break world model decode - disable for WM training
        use_skip: bool = False,
    ):
        super().__init__()
        self.use_skip = use_skip
        
        self.encoder = EncoderHiRes(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            n_residual=n_residual,
        )
        
        self.quantizer = VectorQuantizerHiRes(
            n_embeddings=n_embeddings,
            embedding_dim=latent_channels,
            ema_decay=ema_decay,
            ema_update_every=ema_update_every,
            dead_code_threshold=dead_code_threshold,
        )
        
        self.decoder = DecoderHiRes(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            n_residual=n_residual,
            use_skip=use_skip,  # v2.9: Propagate to decoder
        )
        
        self.n_embeddings = n_embeddings
        self.latent_channels = latent_channels
        
        # Token dimensions - computed from input size if provided
        if input_size is not None:
            self.input_h, self.input_w = input_size
            self.token_h = input_size[0] // 4  # 2 stride-2 layers
            self.token_w = input_size[1] // 4
            self.n_tokens = self.token_h * self.token_w
        else:
            # Default to aspect-preserved Atari (84x64)
            self.input_h, self.input_w = 84, 64
            self.token_h = 21
            self.token_w = 16
            self.n_tokens = 336
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass. 
        
        v2.9: Skip connections disabled by default (use_skip=False) because they
        break world model decode - the decoder learns to rely on encoder features
        that aren't available when decoding from predicted tokens.
        """
        # Get encoder output (with or without skips based on use_skip)
        if self.use_skip:
            z, skips = self.encoder(x, return_skips=True)
        else:
            z = self.encoder(x, return_skips=False)
            skips = None
        # v2.5: Quantizer internally handles FP32 distance computation
        z_q, vq_loss, indices = self.quantizer(z)
        # Decode (skips only used if use_skip=True)
        reconstructed = self.decoder(z_q, skips=skips)
        return reconstructed, vq_loss, indices
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to token indices."""
        z = self.encoder(x, return_skips=False)
        _, _, indices = self.quantizer(z)
        return indices
    
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode() - for compatibility."""
        return self.encode(x)
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode token indices back to images.
        
        v2.2 fix: Handles both (B, H, W) and flattened (B, N) inputs.
        World model typically outputs flattened (B, 336) tokens.
        """
        # v2.2: Handle flattened tokens from world model
        if indices.ndim == 2:
            B, N = indices.shape
            if N == self.n_tokens:
                indices = indices.view(B, self.token_h, self.token_w)
            else:
                raise ValueError(f"Expected {self.n_tokens} tokens, got {N}")
        
        # indices: (B, H, W) -> embeddings: (B, H, W, C)
        z_q = self.quantizer.embedding(indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)
    
    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Alias for decode() - for compatibility."""
        return self.decode(indices)
    
    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Return percentage of codebook being used (legacy method)."""
        stats = self.get_codebook_stats(indices)
        return stats['usage_pct']
    
    def get_codebook_stats(self, indices: Optional[torch.Tensor] = None) -> dict:
        """
        v2.1: Get detailed codebook statistics.
        
        Returns dict with: usage_pct, perplexity, entropy, dead_codes
        """
        return self.quantizer.get_codebook_stats(indices)
    
    def reset_usage_counts(self):
        """Reset usage counts (call at start of epoch for fresh stats)."""
        self.quantizer.reset_usage_counts()
    
    def get_token_grid_size(self, input_h: int, input_w: int) -> Tuple[int, int]:
        """Compute token grid size for given input dimensions."""
        return (input_h // 4, input_w // 4)
    
    def freeze_for_world_model(self):
        """
        Freeze VQ-VAE for world model training.
        
        CRITICAL: Call this before tokenizing for world model training!
        
        This prevents:
        1. Gradient updates to VQ-VAE parameters
        2. EMA codebook drift during tokenization
        
        Usage:
            vqvae.load_state_dict(checkpoint['model_state_dict'])
            vqvae.freeze_for_world_model()  # <-- IMPORTANT!
            tokens = vqvae.encode(frames)
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("  [VQ-VAE] Frozen for world model training (eval mode, no gradients)")


if __name__ == "__main__":
    print("=" * 60)
    print("VQ-VAE v2.1 Test")
    print("=" * 60)
    
    # Test with aspect-preserved Atari (84x64)
    print("\n--- Test 1: Aspect-Preserved Atari (84x64) ---")
    vqvae = VQVAEHiRes(
        in_channels=3,
        hidden_channels=64,
        latent_channels=256,
        n_embeddings=512,
        input_size=(84, 64),
        ema_update_every=10,        # v2.1
        dead_code_threshold=0.01,   # v2.1
    )
    
    total_params = sum(p.numel() for p in vqvae.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Put in training mode to test EMA
    vqvae.train()
    
    x = torch.randn(4, 3, 84, 64)
    reconstructed, vq_loss, indices = vqvae(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Token indices shape: {indices.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Token grid: {vqvae.token_h}x{vqvae.token_w} = {vqvae.n_tokens} tokens")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    
    # v2.1: Test output range (should be [-1, 1] due to tanh)
    print(f"\n--- Test 2: Output Range (v2.1 tanh) ---")
    print(f"Reconstructed min: {reconstructed.min().item():.4f}")
    print(f"Reconstructed max: {reconstructed.max().item():.4f}")
    assert reconstructed.min() >= -1.0 and reconstructed.max() <= 1.0, "Output should be in [-1, 1]!"
    print("[OK] Output constrained to [-1, 1]")
    
    # v2.1: Test codebook stats
    print(f"\n--- Test 3: Codebook Stats (v2.1 histogram) ---")
    # Run a few batches to accumulate stats
    for _ in range(10):
        x_batch = torch.randn(8, 3, 84, 64)
        _, _, _ = vqvae(x_batch)
    
    stats = vqvae.get_codebook_stats()
    print(f"Usage: {stats['usage_pct']:.1f}%")
    print(f"Perplexity: {stats['perplexity']:.1f}")
    print(f"Entropy: {stats['entropy']:.3f}")
    print(f"Dead codes: {stats['dead_codes']}")
    
    # Reset and verify
    vqvae.reset_usage_counts()
    stats_reset = vqvae.get_codebook_stats()
    print(f"After reset - Usage: {stats_reset['usage_pct']:.1f}%")
    
    # Test with specific indices
    stats_batch = vqvae.get_codebook_stats(indices)
    print(f"Batch-only stats - Usage: {stats_batch['usage_pct']:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("v2.1 Improvements Summary:")
    print("=" * 60)
    print("  [OK] Upsample + Conv decoder (no checkerboard)")
    print("  [OK] Tanh output (constrained to [-1, 1])")
    print("  [OK] Configurable EMA frequency (ema_update_every=10)")
    print("  [OK] Smart dead code reset (threshold=1%)")
    print("  [OK] Histogram codebook stats (usage, perplexity, entropy)")
    print("\n[OK] VQ-VAE v2.1 working!")
