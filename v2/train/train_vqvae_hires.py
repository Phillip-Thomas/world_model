"""
VQ-VAE Training Script v2.14 (Run Directory Support)
=====================================================
Adds timestamped run directories for experiment tracking.

v2.14 improvements:
- Timestamped run directories (like world model training)
- Config saving to JSON for reproducibility
- Training stats logging to text file
- --from-run and --from-checkpoint for easy resuming
- Better checkpoint organization (best, latest, periodic)

v2.13 performance optimizations:
- Reduce tqdm update frequency to avoid GPU sync every batch
- Increase stats_log_every to reduce sync overhead
- GPU-only validation stats (no .cpu() per batch)
"""

import sys
import os
import gc
import glob
import re
import itertools
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.vqvae_hires import VQVAEHiRes
from config.defaults import VQVAEConfig


# =============================================================================
# Run Directory Management
# =============================================================================


def resolve_resume_path(from_run: str = None, from_checkpoint: str = None, base_dir: str = "checkpoints/v2/atari") -> str:
    """
    Resolve checkpoint path for resuming training.
    
    Args:
        from_run: Run directory name (e.g., "20260110_111307") or "latest"
        from_checkpoint: Checkpoint name (e.g., "best", "epoch20") or None for latest
        base_dir: Base directory containing vqvae_runs folder
        
    Returns:
        Full path to checkpoint file, or None if from_run not specified
    """
    if not from_run:
        return None
    
    # Build runs directory from base_dir
    vqvae_runs_dir = os.path.join(base_dir, "vqvae_runs")
    
    # Find run directory
    if from_run == "latest":
        if not os.path.exists(vqvae_runs_dir):
            raise FileNotFoundError(f"No runs directory found: {vqvae_runs_dir}")
        run_dirs = [d for d in os.listdir(vqvae_runs_dir) if os.path.isdir(os.path.join(vqvae_runs_dir, d))]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found in: {vqvae_runs_dir}")
        run_dirs.sort(reverse=True)  # Timestamp format sorts correctly
        run_name = run_dirs[0]
    else:
        run_name = from_run
    
    run_path = os.path.join(vqvae_runs_dir, run_name)
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # Find checkpoint
    if from_checkpoint:
        candidates = [
            f"vqvae_{from_checkpoint}.pt",       # best checkpoint
            f"vqvae_epoch{from_checkpoint}.pt",  # epoch checkpoints
            f"{from_checkpoint}.pt",             # direct name
        ]
        for ckpt_name in candidates:
            ckpt_path = os.path.join(run_path, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        raise FileNotFoundError(f"Checkpoint not found: {from_checkpoint} in {run_path}\n  Tried: {candidates}")
    else:
        # Find latest checkpoint by epoch number
        checkpoints = glob.glob(os.path.join(run_path, "vqvae*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in: {run_path}")
        
        best_ckpt = None
        best_epoch = -1
        for ckpt in checkpoints:
            basename = os.path.basename(ckpt)
            match = re.search(r'epoch(\d+)\.pt$', basename)
            if match:
                epoch = int(match.group(1))
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_ckpt = ckpt
            elif basename == "vqvae_latest.pt" and best_epoch < 0:
                best_ckpt = ckpt
                best_epoch = 0
        
        if best_ckpt is None:
            raise FileNotFoundError(f"No valid checkpoint found in: {run_path}")
        return best_ckpt

# Hoisted constants for sobel_edge_loss (CPU templates - will be cached on GPU)
_SOBEL_X_CPU = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y_CPU = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_RGB_W_CPU = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)

# GPU-cached versions (initialized on first use)
_SOBEL_CACHE = {}

def _get_sobel_tensors(device, dtype):
    """Get cached GPU Sobel tensors (create once per device/dtype combo)."""
    key = (device, dtype)
    if key not in _SOBEL_CACHE:
        _SOBEL_CACHE[key] = (
            _SOBEL_X_CPU.to(device=device, dtype=dtype),
            _SOBEL_Y_CPU.to(device=device, dtype=dtype),
            _RGB_W_CPU.to(device=device, dtype=dtype),
        )
    return _SOBEL_CACHE[key]


def sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute edge-aware loss using Sobel filters.
    v2.14: GPU tensors are cached to prevent allocation every batch.
    """
    C = pred.shape[1]
    
    # Get cached GPU tensors (no allocation after first call!)
    sobel_x, sobel_y, weights = _get_sobel_tensors(pred.device, pred.dtype)
    
    # Convert to grayscale based on channel count
    if C == 1:
        pred_gray = pred
        target_gray = target
    elif C == 3:
        pred_gray = (pred * weights).sum(dim=1, keepdim=True)
        target_gray = (target * weights).sum(dim=1, keepdim=True)
    else:
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
    
    # Compute edges
    pred_edge_x = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred_gray, sobel_y, padding=1)
    target_edge_x = F.conv2d(target_gray, sobel_x, padding=1)
    target_edge_y = F.conv2d(target_gray, sobel_y, padding=1)
    
    pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
    target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)
    
    return F.l1_loss(pred_edge, target_edge)


class FrameWithPrevDataset(torch.utils.data.Dataset):
    """
    Dataset that returns (frame_t, frame_{t-1}) for motion-weighted loss.
    
    v2.21: Stores frames as uint8 numpy array to save ~4x RAM.
    Normalization to float32 [-1, 1] happens per-batch in __getitem__.
    """
    def __init__(self, frames_uint8: np.ndarray, episode_starts_set: set):
        # Keep as uint8 numpy - only ~2.4GB for 150k frames instead of ~10GB float32
        self.x = frames_uint8  # (N, H, W, C) uint8
        self.episode_starts = episode_starts_set
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        # Convert to float32 and normalize per-sample (not upfront!)
        frame = torch.from_numpy(self.x[i]).float()
        frame = frame.permute(2, 0, 1) / 127.5 - 1.0  # (H,W,C) -> (C,H,W), normalize
        
        # Use same frame if at episode start or index 0
        j = i if (i in self.episode_starts or i == 0) else (i - 1)
        prev_frame = torch.from_numpy(self.x[j]).float()
        prev_frame = prev_frame.permute(2, 0, 1) / 127.5 - 1.0
        
        return frame, prev_frame


def episode_based_split(episode_starts: np.ndarray, total_frames: int, val_ratio: float = 0.1):
    """
    Split by episode to prevent data leakage.
    Returns train_indices, val_indices as lists.
    """
    n_episodes = len(episode_starts)
    n_val_episodes = max(1, int(n_episodes * val_ratio))
    
    # Hold out last N episodes for validation
    val_episode_start_idx = n_episodes - n_val_episodes
    
    if val_episode_start_idx < n_episodes:
        val_start_frame = int(episode_starts[val_episode_start_idx])
    else:
        val_start_frame = total_frames
    
    train_indices = list(range(0, val_start_frame))
    val_indices = list(range(val_start_frame, total_frames))
    
    return train_indices, val_indices


def train_vqvae_hires(
    data_path: str = "checkpoints/v2/atari/atari_game_data.npz",
    base_dir: str = "checkpoints/v2/atari",
    n_epochs: int = 25,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    beta: float = 0.1,  # VQ loss weight (lowered to prevent collapse)
    edge_weight: float = 0.05,
    n_embeddings: int = 64,  # Codebook size (64 works well for Atari)
    hidden_channels: int = 64,  # Encoder/decoder capacity (try 96 for more detail)
    ema_decay: float = 0.95,  # Per-update decay (with ema_update_every=10)
    ema_update_every: int = 10,
    resume: bool = False,
    resume_path: str = None,  # Explicit path to resume from
    workers: int = 0,
    max_batches_per_epoch: int = None,
    max_frames: int = 200000,  # Limit frames loaded to avoid OOM
):
    """
    Train VQ-VAE v2.14 with run directory support.
    
    Features:
    - Timestamped run directories for experiment tracking
    - Config saving for reproducibility
    - Training stats logging to text file
    - Better checkpoint organization
    
    Performance optimizations:
    - Update tqdm every 20 batches (not every batch)
    - Compute detailed stats every 500 batches (not 50)
    - GPU-only validation histogram
    """
    # Change to project root for consistent paths
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create timestamped run directory (uses base_dir, not hardcoded atari path)
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vqvae_runs_dir = os.path.join(base_dir, "vqvae_runs")
    run_dir = os.path.join(vqvae_runs_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Create and save config
    run_config = VQVAEConfig()
    run_config.timestamp = timestamp
    run_config.run_dir = run_dir
    run_config.data_path = data_path
    run_config.training.n_epochs = n_epochs
    run_config.training.batch_size = batch_size
    run_config.training.learning_rate = learning_rate
    run_config.training.beta = beta
    run_config.training.edge_weight = edge_weight
    run_config.training.ema_decay = ema_decay
    run_config.training.ema_update_every = ema_update_every
    run_config.training.max_batches = max_batches_per_epoch or 0
    run_config.training.max_frames = max_frames
    run_config.training.workers = workers
    run_config.model.n_embeddings = n_embeddings
    
    # v2.22: Limit GPU memory to prevent spilling into shared memory
    if device.type == "cuda":
        # Get total GPU memory and limit to 90% to leave headroom
        total_mem = torch.cuda.get_device_properties(0).total_memory
        # Set memory fraction to prevent shared memory usage
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
        print(f"  GPU memory limit: {total_mem * 0.85 / 1e9:.1f}GB (85% of {total_mem / 1e9:.1f}GB)")
    
    checkpoint_path = Path(run_dir)
    
    # Load data
    print("\nLoading game data...")
    data = np.load(data_path, allow_pickle=True)
    frames = data['frames']
    episode_starts = data.get('episode_starts', np.array([0]))
    
    # Limit frames to avoid OOM (VQ-VAE doesn't need all frames to learn codebook)
    if max_frames and len(frames) > max_frames:
        print(f"  Limiting to {max_frames:,} frames (from {len(frames):,}) to fit in RAM")
        frames = frames[:max_frames]
        # Adjust episode_starts to only include valid ones
        episode_starts = episode_starts[episode_starts < max_frames]
    
    print(f"  Loaded {len(frames):,} frames")
    print(f"  Frame size: {frames.shape[1]}x{frames.shape[2]}x{frames.shape[3]}")
    print(f"  Episodes: {len(episode_starts)}")
    ram_mb = frames.nbytes / 1024 / 1024
    print(f"  RAM usage: {ram_mb:.0f} MB (uint8, ~4x smaller than float32)")
    
    input_h, input_w = frames.shape[1], frames.shape[2]
    token_h, token_w = input_h // 4, input_w // 4
    print(f"  Token grid: {token_h}x{token_w} = {token_h * token_w} tokens")
    
    # v2.21: Keep frames as uint8 numpy - normalization happens per-batch in Dataset
    # This saves ~4x RAM (2.4GB vs 10GB for 150k frames)
    data.close()  # Close the npz file handle
    del data
    
    # Episode-based split
    print(f"\n  Splitting by episode (no frame leakage)...")
    train_indices, val_indices = episode_based_split(episode_starts, len(frames))
    print(f"  Train: {len(train_indices):,}, Val: {len(val_indices):,}")
    
    # Create datasets - pass uint8 numpy array directly
    episode_starts_set = set(int(v) for v in episode_starts.tolist()) if len(episode_starts) else set()
    full_dataset = FrameWithPrevDataset(frames, episode_starts_set)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # v2.20: Disable persistent_workers when using max_batches to prevent memory leak on break
    use_persistent = (workers > 0) and (max_batches_per_epoch is None)
    
    # v2.22: Disable pin_memory to prevent dedicated GPU memory accumulation
    # Pinned memory shows up as "dedicated" and doesn't get released
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,  # Disabled to prevent memory accumulation
        drop_last=True,
        persistent_workers=use_persistent,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,  # Disabled to prevent memory accumulation
        drop_last=False,
        persistent_workers=use_persistent,
    )
    
    # Create model
    print(f"\nCreating VQ-VAE v2.1 for {input_h}x{input_w} input (hidden={hidden_channels})...")
    model = VQVAEHiRes(
        in_channels=3,
        hidden_channels=hidden_channels,
        latent_channels=256,
        n_embeddings=n_embeddings,
        input_size=(input_h, input_w),
        ema_decay=ema_decay,
        ema_update_every=ema_update_every,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Token grid: {model.token_h}x{model.token_w} = {model.n_tokens} tokens")
    print(f"  Each token: 4x4 = 16 pixels")
    
    # Optimizer (exclude embedding from weight decay - handled by EMA)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    start_epoch = 0
    best_val_loss = float('inf')
    # Default history with all keys we'll use
    history = {'train_loss': [], 'val_loss': [], 'usage': [], 'perplexity': []}
    
    def _ensure_history_keys(h):
        """Ensure history dict has all required keys."""
        defaults = {'train_loss': [], 'val_loss': [], 'usage': [], 'perplexity': []}
        for k, v in defaults.items():
            if k not in h:
                h[k] = v
        return h
    
    # Resume from checkpoint
    resumed_from_path = None
    if resume:
        # Use explicit resume_path if provided
        # Try new naming first, fall back to legacy
        if resume_path:
            ckpt_path = resume_path
        else:
            ckpt_path = os.path.join(base_dir, "vqvae_hires.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(base_dir, "atari_vqvae_hires.pt")  # Legacy
        
        if os.path.exists(ckpt_path):
            print(f"\nResuming from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            # v2.14: Use strict=False to allow new buffers that old checkpoints don't have
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            if 'history' in checkpoint:
                history = _ensure_history_keys(checkpoint['history'])
            resumed_from_path = ckpt_path
            print(f"  Resumed at epoch {start_epoch + 1}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"  No checkpoint found at {ckpt_path}, starting fresh")
    
    # Update config with detected dimensions and save
    run_config.model.input_h = input_h
    run_config.model.input_w = input_w
    run_config.save(os.path.join(run_dir, "config.json"))
    print(f"  Config saved to {run_dir}/config.json")
    
    # Write session header to stats file
    _write_session_header(
        save_dir=run_dir,
        start_epoch=start_epoch + 1,
        end_epoch=start_epoch + n_epochs,
        resumed_from=resumed_from_path,
        resumed_epoch=start_epoch if resumed_from_path else None,
        resumed_loss=best_val_loss if resumed_from_path else None
    )
    
    # Effective batches per epoch
    effective_batches = max_batches_per_epoch if max_batches_per_epoch else len(train_loader)
    
    # Training config
    print("\n" + "=" * 60)
    print(f"Training VQ-VAE v2.13 ({input_h}x{input_w} -> {token_h}x{token_w} tokens)")
    print(f"  Loss: L1 + 0.1*MSE + {edge_weight}*edge + {beta}*VQ")
    print(f"  Distance: Euclidean (FP32)")
    print(f"  EMA: efficient bincount+index_add, every {ema_update_every} batches")
    print(f"  Split: episode-based (no frame leakage)")
    print(f"  Batch size: {batch_size}, Batches/epoch: {effective_batches}")
    print(f"  PERF: tqdm update every 20, stats every 500, no empty_cache")
    print("=" * 60)
    
    # Performance settings
    tqdm_update_every = 20  # Only sync/update tqdm every N batches
    stats_log_every = 500   # Detailed stats much less frequently
    memory_cleanup_every = 50  # Periodic cleanup (per-batch GC didn't help, issue is elsewhere)
    
    # Note: GC stays enabled but we call it periodically to prevent buildup
    
    for epoch in range(start_epoch, n_epochs):
        # v2.22: Clear GPU cache at START of each epoch (like world model training)
        torch.cuda.empty_cache()
        gc.collect()
        
        model.train()
        model.reset_usage_counts()
        
        # Commitment annealing: ramp from 0 to 0.25 over first 5 epochs
        if epoch < 5:
            model.quantizer.commitment_cost = 0.25 * (epoch / 5)
        else:
            model.quantizer.commitment_cost = 0.25
        
        train_loss = 0.0
        train_l1 = 0.0
        batch_count = 0
        
        # v2.20: Use islice to iterate exactly N batches (clean termination, no break leak)
        train_iter = itertools.islice(train_loader, effective_batches)
        pbar = tqdm(train_iter, desc=f"Epoch {epoch+1}/{n_epochs}", total=effective_batches)
        last_loss = 0.0
        last_l1 = 0.0
        
        for batch, prev_batch in pbar:
            batch = batch.to(device, non_blocking=True)
            prev_batch = prev_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                recon, vq_loss, indices = model(batch)
                
                # Motion-weighted + brightness-preserving loss
                motion = (batch - prev_batch).abs().mean(dim=1, keepdim=True)
                brightness = ((batch + 1.0) * 0.5).mean(dim=1, keepdim=True)
                
                motion_thr = 0.05
                bright_thr = 0.4
                motion_alpha = 8.0
                bright_alpha = 3.0
                
                motion_mask = (motion > motion_thr).float()
                bright_mask = (brightness > bright_thr).float()
                w = 1.0 + motion_alpha * motion_mask + bright_alpha * bright_mask
                # Normalize per-image to prevent background drift
                w = w / (w.mean(dim=[2, 3], keepdim=True) + 1e-8)
                
                diff = recon - batch
                recon_l1 = (diff.abs() * w).mean()
                recon_mse = ((diff * diff) * w).mean()
                recon_loss = recon_l1 + 0.1 * recon_mse
                
                # Edge loss
                if edge_weight > 0:
                    edge_loss = sobel_edge_loss(recon, batch)
                    recon_loss = recon_loss + edge_weight * edge_loss
                
                loss = recon_loss + beta * vq_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track losses as Python floats to prevent GPU tensor accumulation
            last_loss_val = loss.detach().item()
            last_l1_val = recon_l1.detach().item()
            train_loss += last_loss_val
            train_l1 += last_l1_val
            batch_count += 1
            
            # v2.19: Delete ALL intermediate tensors to prevent memory growth
            del recon, vq_loss, indices, loss
            del motion, brightness, motion_mask, bright_mask, w, diff
            del recon_l1, recon_mse, recon_loss
            del batch, prev_batch
            if edge_weight > 0:
                del edge_loss
            
            # v2.22: Periodic cache clear to prevent VRAM growth into shared memory
            if batch_count % 500 == 0:
                torch.cuda.empty_cache()
            
            # Update tqdm
            if batch_count % tqdm_update_every == 0:
                pbar.set_postfix(loss=f"{last_loss_val:.4f}", L1=f"{last_l1_val:.4f}")
        
        # v2.20: No break needed - islice handles batch limit cleanly
        pbar.close()
        scheduler.step()
        
        # v2.20: Compute epoch stats on CPU to avoid VRAM spike from clone()
        with torch.no_grad():
            epoch_counts_cpu = model.quantizer.usage_counts.cpu().numpy()
            epoch_total = epoch_counts_cpu.sum()
            if epoch_total > 0:
                epoch_probs = epoch_counts_cpu / epoch_total
                nonzero_mask = epoch_probs > 0
                probs_nonzero = epoch_probs[nonzero_mask]
                epoch_entropy = -np.sum(probs_nonzero * np.log(probs_nonzero + 1e-10))
                epoch_perplexity = float(np.exp(epoch_entropy))
                epoch_usage = float(nonzero_mask.mean() * 100)
                epoch_dead = int((~nonzero_mask).sum())
            else:
                epoch_perplexity, epoch_usage, epoch_dead = 0.0, 0.0, n_embeddings
        
        # v2.17: Clean up memory ONCE per epoch
        torch.cuda.empty_cache()
        
        # Average losses (already Python floats, no sync needed)
        avg_train_loss = train_loss / batch_count
        avg_train_l1 = train_l1 / batch_count
        
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        # v2.20: Validation with explicit tensor cleanup
        with torch.no_grad():
            for batch, _ in val_loader:
                batch = batch.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    recon, vq_loss, indices = model(batch)
                    recon_l1 = F.l1_loss(recon, batch)
                    recon_mse = F.mse_loss(recon, batch)
                    recon_loss = recon_l1 + 0.1 * recon_mse
                    if edge_weight > 0:
                        edge_loss = sobel_edge_loss(recon, batch)
                        recon_loss = recon_loss + edge_weight * edge_loss
                    loss = recon_loss + beta * vq_loss
                
                val_loss += loss.item()
                val_count += 1
                
                # v2.20: Explicit cleanup to reduce VRAM peak
                del recon, vq_loss, indices, loss, recon_l1, recon_mse, recon_loss, batch
                if edge_weight > 0:
                    del edge_loss
        
        # Compute average
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        
        # Get EMA dead codes count (extract Python int immediately)
        ema_dead = model.get_codebook_stats().get('ema_dead_codes', 0)
        
        is_best = avg_val_loss < best_val_loss
        print(f"\n  Epoch {epoch+1}: loss={avg_val_loss:.4f}, L1={avg_train_l1:.4f}, "
              f"usage={epoch_usage:.1f}%, pplx={int(epoch_perplexity)}, "
              f"dead={int(epoch_dead)}, ema_dead={ema_dead}")
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['usage'].append(epoch_usage)
        history['perplexity'].append(epoch_perplexity)
        
        # Write epoch stats to text file
        epoch_stats = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'usage': epoch_usage,
            'perplexity': epoch_perplexity,
            'dead_codes': int(epoch_dead),
            'ema_dead': ema_dead,
        }
        _write_epoch_stats(epoch + 1, epoch_stats, run_dir, is_best=is_best)
        
        # Save best model
        if is_best:
            best_val_loss = avg_val_loss
            # Save to run directory
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'codebook_usage': epoch_usage,
                'perplexity': epoch_perplexity,
                'n_embeddings': n_embeddings,
                'hidden_channels': hidden_channels,
                'input_h': input_h,
                'input_w': input_w,
                'history': history,
            }, checkpoint_path / "vqvae_best.pt")
            # Also save to base_dir for easy access by world model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'n_embeddings': n_embeddings,
                'hidden_channels': hidden_channels,
                'input_h': input_h,
                'input_w': input_w,
            }, os.path.join(base_dir, "vqvae_hires.pt"))
            print(f"  * New best model saved!")
        
        # Always save latest (for resuming)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch + 1,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'n_embeddings': n_embeddings,
            'hidden_channels': hidden_channels,
            'input_h': input_h,
            'input_w': input_w,
            'history': history,
        }, checkpoint_path / "vqvae_latest.pt")
        
        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'n_embeddings': n_embeddings,
                'hidden_channels': hidden_channels,
                'input_h': input_h,
                'input_w': input_w,
                'history': history,
            }, checkpoint_path / f"vqvae_epoch{epoch+1}.pt")
            print(f"  Periodic checkpoint saved (epoch {epoch+1})")
        
        # v2.22: Aggressive cleanup after checkpoint saves to prevent memory buildup
        # torch.save() creates copies of state_dicts that need to be freed
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()  # Double GC to catch circular references
    
    # Plot training
    _plot_training(history, checkpoint_path, n_embeddings)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Run directory: {run_dir}")
    print(f"  Best model:    {checkpoint_path / 'vqvae_best.pt'}")
    print(f"  Latest:        {checkpoint_path / 'vqvae_latest.pt'}")
    print(f"  Also saved to: {base_dir}/vqvae_hires.pt (for world model)")
    print("=" * 60)
    
    return model


def _write_session_header(save_dir: str, start_epoch: int, end_epoch: int, 
                          resumed_from: str = None, resumed_epoch: int = None,
                          resumed_loss: float = None):
    """Write a session header to the stats file when training starts/resumes."""
    stats_file = os.path.join(save_dir, "training_stats.txt")
    
    with open(stats_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Training Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if resumed_from:
            f.write(f"  RESUMED FROM: {resumed_from}\n")
            f.write(f"  Resume Epoch: {resumed_epoch}, Loss: {resumed_loss:.4f}\n")
        else:
            f.write("  NEW TRAINING RUN\n")
        f.write(f"  Epochs: {start_epoch} -> {end_epoch}\n")
        f.write("=" * 80 + "\n\n")


def _write_epoch_stats(epoch: int, stats: dict, save_dir: str, is_best: bool = False):
    """Write epoch stats to a text file for easy review."""
    stats_file = os.path.join(save_dir, "training_stats.txt")
    
    # Create header if file doesn't exist
    if not os.path.exists(stats_file):
        with open(stats_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VQ-VAE Training Statistics Log\n")
            f.write("=" * 80 + "\n\n")
    
    with open(stats_file, 'a') as f:
        best_marker = " *BEST*" if is_best else ""
        f.write(f"Epoch {epoch}{best_marker}\n")
        f.write(f"  Train Loss: {stats['train_loss']:.4f}  |  Val Loss: {stats['val_loss']:.4f}\n")
        f.write(f"  Codebook Usage: {stats['usage']:.1f}%  |  Perplexity: {stats['perplexity']:.1f}\n")
        f.write(f"  Dead Codes: {stats['dead_codes']}  |  EMA Dead: {stats['ema_dead']}\n")
        f.write("-" * 40 + "\n")


def _plot_training(history: dict, checkpoint_path: Path, n_embeddings: int):
    """Plot training metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0].plot(history['val_loss'], label='Val', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Usage
    axes[1].plot(history['usage'], 'g-', alpha=0.7)
    axes[1].axhline(y=100, color='g', linestyle='--', alpha=0.3, label='Ideal')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Usage %')
    axes[1].set_title('Codebook Usage')
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Perplexity
    axes[2].plot(history['perplexity'], 'b-', alpha=0.7)
    axes[2].axhline(y=n_embeddings, color='b', linestyle='--', alpha=0.3, label=f'Max ({n_embeddings})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Perplexity')
    axes[2].set_title('Codebook Perplexity')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(checkpoint_path / "vqvae_training.png", dpi=150)
    plt.close()
    print(f"\n  Training plot saved to {checkpoint_path / 'vqvae_training.png'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train VQ-VAE v2.14 with run directory support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data and output paths
    parser.add_argument("--data", type=str, default="checkpoints/v2/atari/atari_game_data.npz",
                        help="Path to game data file")
    parser.add_argument("--base-dir", type=str, default="checkpoints/v2/atari",
                        help="Base directory for shared assets")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--beta", type=float, default=None, help="VQ loss weight")
    parser.add_argument("--edge-weight", type=float, default=None, help="Edge loss weight")
    
    # Model architecture
    parser.add_argument("--n-embeddings", type=int, default=None, help="Codebook size")
    parser.add_argument("--hidden-channels", type=int, default=None, help="Encoder/decoder hidden channels (default: 64, try 96 for more detail)")
    
    # EMA settings
    parser.add_argument("--ema-decay", type=float, default=None, help="EMA decay rate")
    parser.add_argument("--ema-update-every", type=int, default=None, help="EMA update frequency")
    
    # Resume functionality
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--from-run", type=str, default=None,
                        help='Run directory to resume from (e.g., "20260110_111307" or "latest")')
    parser.add_argument("--from-checkpoint", type=str, default=None,
                        help='Checkpoint within run (e.g., "best", "epoch20"). Default: latest')
    
    # Performance settings
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches per epoch (0=all)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to load")
    
    args = parser.parse_args()
    
    # Load defaults, then override with CLI args
    config = VQVAEConfig()
    config.update_from_args(args)
    
    # Resolve resume path if --from-run specified
    resume_path = None
    if args.from_run:
        args.resume = True  # Automatically enable resume if --from-run specified
        resume_path = resolve_resume_path(args.from_run, args.from_checkpoint, args.base_dir)
        print(f"\n  Resume from: {resume_path}")
    
    # Print effective config
    print("\n" + "=" * 60)
    print("Effective Configuration (defaults + CLI overrides)")
    print("=" * 60)
    print(f"  Training: epochs={config.training.n_epochs}, batch={config.training.batch_size}, lr={config.training.learning_rate}")
    print(f"  Loss: beta={config.training.beta}, edge_weight={config.training.edge_weight}")
    print(f"  Model: n_embeddings={config.model.n_embeddings}, hidden_channels={config.model.hidden_channels}")
    print(f"  EMA: decay={config.training.ema_decay}, update_every={config.training.ema_update_every}")
    print("=" * 60 + "\n")
    
    train_vqvae_hires(
        data_path=args.data,
        base_dir=args.base_dir,
        n_epochs=config.training.n_epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        beta=config.training.beta,
        edge_weight=config.training.edge_weight,
        n_embeddings=config.model.n_embeddings,
        hidden_channels=config.model.hidden_channels,
        ema_decay=config.training.ema_decay,
        ema_update_every=config.training.ema_update_every,
        resume=args.resume,
        resume_path=resume_path,
        workers=config.training.workers,
        max_batches_per_epoch=config.training.max_batches if config.training.max_batches > 0 else None,
        max_frames=config.training.max_frames,
    )
