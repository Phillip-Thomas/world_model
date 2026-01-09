"""
Train World Model with High-Res VQ-VAE
======================================
Supports flexible token grid sizes:
- 21x16 = 336 tokens (84x64 aspect-preserved)
- 16x16 = 256 tokens (64x64 legacy square)

v1.2: 10 transformer layers (was 8)
v1.3: Multi-step rollout training (scheduled sampling / DAgger-lite)
      - Occasionally unroll K steps using model's own predictions
      - Helps actions "stick" in long rollouts
v1.4: Token-importance weighting
      - Weight moving tokens more heavily in loss

v2.0: Hybrid importance weighting (game-agnostic, robust)
      - Embedding-distance motion: L2 in embedding space (robust to codebook jitter)
      - Eventness spike detection: upweight "fire/spawn" moments
      - Multi-frame persistence: real movers vs jitter
      - Percentile-based safe capping: prevents context starvation

v2.1: Multi-step rollout improvements
      - Normalized discounted loss: stable across K / discount values
      - Scheduled sampling: teacher forcing mix prevents "ball death spiral"
      - Token importance weights applied per rollout step
      - Efficient history update using roll() instead of cat()
      - Frame/action alignment assertion
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.vqvae_hires import VQVAEHiRes
from models.temporal_world_model import TemporalVisualWorldModel
from data.atari_dataset import AtariTemporalDataset
from data.multistep_dataset import MultiStepDataset
from eval.gold_eval import GoldEvalSuite, run_gold_eval, format_gold_metrics
from utils.importance_weights import compute_hybrid_importance_weights, compute_token_importance_weights


def multistep_rollout_loss(
    model: TemporalVisualWorldModel,
    history: torch.Tensor,      # (B, T, N) initial history
    actions: torch.Tensor,      # (B, K) actions for K steps
    targets: torch.Tensor,      # (B, K, N) targets for K steps
    device: torch.device,
    step_discount: float = 0.7, # v1.9: discount factor per step (game-agnostic)
    teacher_forcing_prob: float = 0.1,  # v2.1: scheduled sampling to prevent ball death spiral
    token_embedding: torch.nn.Embedding = None,  # v2.1: for importance weighting
    use_importance_weights: bool = True,  # v2.1: apply weights per rollout step
    motion_scale: float = 2.0,
    eventness_scale: float = 2.0,
    persistence_scale: float = 1.0,
    max_ratio: float = 6.0,
) -> tuple:
    """
    Compute multi-step rollout loss with discounted step weights.
    
    v1.9: Later steps weighted less (1.0, 0.7, 0.49, 0.34, ...)
    v2.1: Normalized loss, scheduled sampling, importance weighting per step.
    
    Key improvements:
    - Loss normalized by weight sum (stable across K / discount values)
    - Teacher forcing mix (prevents "ball death spiral" early in training)
    - Token importance weights applied per rollout step
    - Efficient history update using roll() instead of cat()
    
    Returns:
        total_loss: normalized discounted sum of losses across all steps
        step_accuracies: list of accuracies at each step
    """
    B, T, N = history.shape
    K = actions.shape[1]
    
    total_loss = 0.0
    weight_sum = 0.0  # v2.1: track for normalization
    step_accs = []
    current_history = history.clone()
    step_weight = 1.0
    
    for k in range(K):
        # Get action and target for this step
        action_k = actions[:, k]  # (B,)
        target_k = targets[:, k].contiguous()  # (B, N)
        
        # Forward pass
        logits, _ = model(current_history, action_k, None)  # (B, N, vocab)
        
        # v2.1: Compute token importance weights if enabled
        if use_importance_weights and token_embedding is not None:
            token_weights = compute_hybrid_importance_weights(
                target_k, current_history, token_embedding, device,
                motion_scale=motion_scale,
                eventness_scale=eventness_scale,
                persistence_scale=persistence_scale,
                max_ratio=max_ratio,
            )
            # Weighted CE loss
            ce_per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_k.reshape(-1),
                reduction='none'
            ).reshape(B, N)
            loss = (ce_per_token * token_weights).mean()
        else:
            # Standard CE loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_k.reshape(-1),
            )
        
        total_loss = total_loss + step_weight * loss
        weight_sum += step_weight  # v2.1: accumulate for normalization
        step_weight *= step_discount
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == target_k).float().mean().item()
        step_accs.append(acc)
        
        # v2.1: Scheduled sampling - mix teacher forcing to prevent ball death spiral
        # With probability teacher_forcing_prob, use true target instead of prediction
        with torch.no_grad():
            if teacher_forcing_prob > 0 and k < K - 1:  # Don't apply on last step
                use_tf = (torch.rand(B, 1, device=device) < teacher_forcing_prob).float()
                next_tokens = use_tf * target_k.float() + (1 - use_tf) * preds.float()
                next_tokens = next_tokens.long()
            else:
                next_tokens = preds
            
            # v2.1: Efficient history update using roll() instead of cat()
            current_history = current_history.roll(shifts=-1, dims=1)
            current_history[:, -1, :] = next_tokens
        
        del logits, preds
    
    # v2.1: Normalize loss by weight sum (keeps scale stable across K / discount)
    total_loss = total_loss / (weight_sum + 1e-8)
    
    return total_loss, step_accs


def train_world_model_hires(
    n_epochs: int = 30,
    batch_size: int = 8,  # Reduced for K=5 rollouts
    learning_rate: float = 5e-4,  # Increased from 3e-4 to shake things up
    resume: bool = False,
    max_batches: int = 100,  # Limit batches per epoch for faster iteration
    full_val_every: int = 999,  # Disabled for quick iteration (only final epoch)
    # v1.3: Multi-step rollout training
    rollout_steps: int = 5,      # K steps to unroll (0 = disabled)
    rollout_ratio: float = 0.3,  # Fraction of batches to use multi-step
    # v2.0: Hybrid importance weighting (replaces v1.x motion weighting)
    use_hybrid_weights: bool = True,  # Use new hybrid (v2.0) vs legacy (v1.x)
    motion_scale: float = 2.0,        # Embedding-distance motion boost
    eventness_scale: float = 2.0,     # Spike detection boost (fire/spawn moments)
    persistence_scale: float = 1.0,   # Multi-frame consistency boost
    max_ratio: float = 6.0,           # Safe cap: 95th/median ratio limit
    # Legacy v1.x params (only used if use_hybrid_weights=False)
    motion_weight: float = 4.0,
    continuous_bonus: float = 2.0,
    max_weight: float = 8.0,
):
    """Train world model with high-res tokenization and multi-step rollouts."""
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Base dir for shared assets (VQ-VAE, data)
    base_dir = "checkpoints/v2/atari"
    os.makedirs(base_dir, exist_ok=True)
    
    # Timestamped run directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{base_dir}/runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save run config for reproducibility
    run_config = {
        'timestamp': timestamp,
        'training': {
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_batches': max_batches,
            'full_val_every': full_val_every,
            'rollout_steps': rollout_steps,
            'rollout_ratio': rollout_ratio,
            'resume': resume,
        },
        'model': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 10,
            'dropout': 0.1,
            'history_len': 4,
        },
        'weighting': {
            'use_hybrid': use_hybrid_weights,
            # v2.0 hybrid params
            'motion_scale': motion_scale,
            'eventness_scale': eventness_scale,
            'persistence_scale': persistence_scale,
            'max_ratio': max_ratio,
            # Legacy v1.x params
            'motion_weight': motion_weight,
            'continuous_bonus': continuous_bonus,
            'max_weight': max_weight,
            'step_discount': 0.7,
        },
    }
    with open(f"{run_dir}/config.json", 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"  Config saved to {run_dir}/config.json")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Enable optimized attention (SDPA) if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("  SDPA optimizations enabled")
    
    # === Load High-Res VQ-VAE ===
    print("\nLoading VQ-VAE...")
    vqvae_ckpt = torch.load(f"{base_dir}/atari_vqvae_hires.pt", map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    input_h = vqvae_ckpt.get('input_h', 84)
    input_w = vqvae_ckpt.get('input_w', 64)
    n_embeddings = vqvae_ckpt.get('n_embeddings', 32)  # Match checkpoint
    
    vqvae = VQVAEHiRes(
        in_channels=3, hidden_channels=64, latent_channels=256,
        n_embeddings=n_embeddings, n_residual=2,
        input_size=(input_h, input_w),
    ).to(device)
    vqvae.load_state_dict(vqvae_ckpt['model_state_dict'])
    
    # CRITICAL: Freeze VQ-VAE to prevent codebook drift during world model training!
    vqvae.freeze_for_world_model()
    print(f"  Input: {input_h}x{input_w}")
    
    # === Load Data (with optional human gameplay merge) ===
    print("\nLoading game data...")
    data = dict(np.load(f"{base_dir}/atari_game_data.npz", allow_pickle=True))
    n_actions = int(data['n_actions'])  # Must be in dataset (game-agnostic)
    print(f"  Random data: {len(data['frames'])} frames")
    
    # Merge human gameplay if exists (with higher sampling weight)
    human_path = f"{base_dir}/human_gameplay.npz"
    human_start_idx = len(data['frames'])  # Track where human data starts
    human_frame_count = 0
    EXPERT_WEIGHT = 5.0  # Expert samples are 5x more likely to be sampled
    
    if os.path.exists(human_path):
        print(f"  Merging human gameplay data...")
        human = dict(np.load(human_path, allow_pickle=True))
        human_frame_count = len(human['frames'])
        
        # Offset episode starts for human data
        human_ep_starts = human['episode_starts'] + human_start_idx
        
        # Merge all arrays
        data['frames'] = np.concatenate([data['frames'], human['frames']])
        data['actions'] = np.concatenate([data['actions'], human['actions']])
        data['rewards'] = np.concatenate([data['rewards'], human['rewards']])
        data['dones'] = np.concatenate([data['dones'], human['dones']])
        data['episode_starts'] = np.concatenate([data['episode_starts'], human_ep_starts])
        
        # Create sample weights: 1.0 for random, EXPERT_WEIGHT for human
        random_weights = np.ones(human_start_idx, dtype=np.float32)
        expert_weights = np.full(human_frame_count, EXPERT_WEIGHT, dtype=np.float32)
        data['sample_weights'] = np.concatenate([random_weights, expert_weights])
        
        print(f"  Human data: {human_frame_count} frames (weight={EXPERT_WEIGHT}x)")
        print(f"  Combined: {len(data['frames'])} frames")
    else:
        # No human data - uniform weights
        data['sample_weights'] = np.ones(len(data['frames']), dtype=np.float32)
    
    print(f"  Total frames: {len(data['frames'])}, Actions: {n_actions}")
    
    # === Tokenize Frames with High-Res VQ-VAE ===
    tokens_path = f"{base_dir}/atari_tokens_hires.npz"
    
    if os.path.exists(tokens_path):
        print("\nLoading cached high-res tokens...")
        cached = np.load(tokens_path)
        all_tokens = cached['tokens']
    else:
        print("\nTokenizing frames with 16x16 VQ-VAE...")
        frames = data['frames']
        all_tokens = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(frames), 64), desc="Tokenizing"):
                batch = frames[i:i+64]
                tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2)
                tensor = tensor / 127.5 - 1.0
                tensor = tensor.to(device)
                
                tokens = vqvae.encode(tensor)  # (B, 16, 16)
                all_tokens.append(tokens.cpu().numpy())
        
        all_tokens = np.concatenate(all_tokens, axis=0)
        np.savez_compressed(tokens_path, tokens=all_tokens)
        print(f"  Cached to {tokens_path}")
    
    print(f"  Token shape: {all_tokens.shape}")
    
    # === Create Dataset ===
    print("\nCreating dataset...")
    
    all_actions = data['actions']  # For gold eval
    
    token_data = {
        'frames': all_tokens,
        'actions': all_actions,
        'rewards': data['rewards'],
        'dones': data['dones'],
        'episode_starts': data['episode_starts'],
        'sample_weights': data.get('sample_weights', np.ones(len(all_tokens))),
        'n_actions': n_actions,
    }
    
    dataset = AtariTemporalDataset(token_data, history_len=4, preprocessed_tokens=True)
    
    # === Stratified train/val split by action ===
    # v4 FIX: Get actions for each valid sample using dataset's valid_indices
    all_indices = list(range(len(dataset)))
    # Actions corresponding to each sample (use dataset's valid_indices, not arbitrary offset)
    sample_actions = data['actions'][dataset.valid_indices]
    
    # Stratified split: ensure each action is proportionally represented
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.1,
        stratify=sample_actions,
        random_state=42
    )
    
    # Report action distribution in splits
    train_actions = sample_actions[train_indices]
    val_actions = sample_actions[val_indices]
    print(f"  Train action dist: {[f'{(train_actions==i).sum()/len(train_actions)*100:.1f}%' for i in range(n_actions)]}")
    print(f"  Val action dist:   {[f'{(val_actions==i).sum()/len(val_actions)*100:.1f}%' for i in range(n_actions)]}")
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"  Training samples: {len(train_indices)}")
    
    # v4 FIX: Get sample weights using dataset's valid_indices
    # train_indices are indices into dataset, valid_indices maps to action indices
    train_action_indices = dataset.valid_indices[train_indices]
    train_sample_weights = data['sample_weights'][train_action_indices]
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_sample_weights).float(),
        num_samples=len(train_indices),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Report expert data fraction
    expert_samples = (train_sample_weights > 1.0).sum()
    if expert_samples > 0:
        print(f"  Expert samples: {expert_samples} ({100*expert_samples/len(train_sample_weights):.1f}% of training, {EXPERT_WEIGHT}x weight)")
    
    # === Gold Eval Suite (frozen indices) ===
    gold_suite = GoldEvalSuite(
        save_dir=base_dir,  # Keep gold eval indices frozen across runs
        dataset_size=len(dataset),
        n_actions=n_actions,
        canary_size=500,
        fast_val_per_action=1500,
        rollout_sequences=50,
        rollout_horizons=[5, 10, 20],
        seed=42,
    )
    gold_suite.initialize(
        train_indices=np.array(train_indices),
        val_indices=np.array(val_indices),
        sample_actions=sample_actions,
        human_start_idx=human_start_idx,
        human_frame_count=human_frame_count,
    )
    
    # Create fast_val loader from frozen indices
    fast_val_indices = gold_suite.fast_val_indices
    fast_val_dataset = Subset(dataset, fast_val_indices)
    fast_val_loader = DataLoader(fast_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, FastVal: {len(fast_val_indices)} (frozen)")
    
    # === v1.3: Create Multi-Step Dataset for Rollout Training ===
    multistep_loader = None
    if rollout_steps > 0:
        print(f"\nCreating multi-step dataset (K={rollout_steps} steps, {rollout_ratio*100:.0f}% of batches)...")
        multistep_dataset = MultiStepDataset(
            tokens=all_tokens,
            actions=data['actions'],
            dones=data['dones'],
            episode_starts=data['episode_starts'],
            history_len=4,
            rollout_steps=rollout_steps,
        )
        # Use smaller batch size for multi-step (more memory)
        ms_batch_size = max(8, batch_size // 2)
        multistep_loader = DataLoader(
            multistep_dataset, batch_size=ms_batch_size, shuffle=True, 
            num_workers=0, pin_memory=True, drop_last=True
        )
        print(f"  Multi-step samples: {len(multistep_dataset)}, batch_size: {ms_batch_size}")
    
    # === Get token dimensions from data ===
    token_h, token_w = all_tokens.shape[1], all_tokens.shape[2]
    n_tokens = token_h * token_w
    
    # === Create World Model ===
    print(f"\nCreating World Model for {token_h}x{token_w} tokens (10 layers, {n_embeddings} vocab)...")
    model = TemporalVisualWorldModel(
        n_vocab=n_embeddings,  # Match VQ-VAE codebook size
        n_actions=n_actions,
        d_model=256,
        n_heads=8,
        n_layers=10,        # v1.2: 10 layers (was 8)
        token_h=token_h,
        token_w=token_w,
        max_history=4,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # === Resume from checkpoint if requested ===
    start_epoch = 0
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'fast_val_loss': [], 'fast_val_acc': [], 'entropy': [], 'max_prob': [], 'its': [], 'gpu_mem': [], 'val_loss': [], 'val_acc': [], 'val_epochs': []}
    
    if resume:
        ckpt_path = f"{base_dir}/atari_world_model_hires.pt"  # Resume from base_dir
        if os.path.exists(ckpt_path):
            print(f"\n  Loading checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_acc = ckpt.get('val_acc', 0)
            print(f"  Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.1f}%")
        else:
            print(f"  No checkpoint found at {ckpt_path}, starting fresh")
    
    # Update config with runtime info
    run_config['vqvae'] = {
        'input_h': input_h,
        'input_w': input_w,
        'n_embeddings': n_embeddings,
        'token_h': token_h,
        'token_w': token_w,
        'n_tokens': n_tokens,
    }
    run_config['dataset'] = {
        'total_frames': len(all_tokens),
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'n_actions': n_actions,
    }
    run_config['model']['n_vocab'] = n_embeddings
    run_config['model']['n_actions'] = n_actions
    run_config['model']['parameters'] = sum(p.numel() for p in model.parameters())
    
    # Re-save config with full info
    with open(f"{run_dir}/config.json", 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # === Training ===
    print("\n" + "=" * 60)
    print(f"Training World Model ({token_h}x{token_w}={n_tokens} tokens, 10 layers)")
    print(f"  Epochs {start_epoch+1} to {start_epoch+n_epochs}")
    if rollout_steps > 0:
        print(f"  Multi-step rollouts: K={rollout_steps}, ratio={rollout_ratio*100:.0f}%")
    print(f"  v1.5: Gold Eval (frozen indices + rollout metrics)")
    print(f"  Fast-val every epoch, GOLD eval every {full_val_every} epochs")
    print(f"  Checkpoint criterion: 50% 1-step + 50% rollout avg")
    print("=" * 60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    # Create iterator for multi-step batches
    ms_iter = iter(multistep_loader) if multistep_loader else None
    
    last_val_acc = best_val_acc
    
    for epoch in range(n_epochs):
        actual_epoch = start_epoch + epoch
        torch.cuda.empty_cache()
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        batch_count = 0
        ms_batches = 0  # Track multi-step batches
        
        # Track timing and memory
        epoch_start = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        effective_batches = min(max_batches, len(train_loader)) if max_batches > 0 else len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {actual_epoch+1}/{start_epoch+n_epochs}", total=effective_batches)
        for batch_hist, batch_act, batch_target in pbar:
            batch_hist = batch_hist.to(device)
            batch_act = batch_act.to(device)
            batch_target = batch_target.to(device)
            
            # Decide: single-step or multi-step?
            use_multistep = (
                ms_iter is not None and 
                np.random.random() < rollout_ratio
            )
            
            if use_multistep:
                # === Multi-step rollout training ===
                try:
                    ms_hist, ms_acts, ms_targets = next(ms_iter)
                except StopIteration:
                    ms_iter = iter(multistep_loader)
                    ms_hist, ms_acts, ms_targets = next(ms_iter)
                
                ms_hist = ms_hist.to(device)
                ms_acts = ms_acts.to(device)
                ms_targets = ms_targets.to(device)
                
                with torch.amp.autocast('cuda'):
                    loss, step_accs = multistep_rollout_loss(
                        model, ms_hist, ms_acts, ms_targets, device,
                        teacher_forcing_prob=0.1,  # v2.1: prevent ball death spiral
                        token_embedding=model.token_embed,
                        use_importance_weights=use_hybrid_weights,
                        motion_scale=motion_scale,
                        eventness_scale=eventness_scale,
                        persistence_scale=persistence_scale,
                        max_ratio=max_ratio,
                    )
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Use first step accuracy for tracking
                loss_val = loss.item()
                train_loss += loss_val / rollout_steps
                train_correct += int(step_accs[0] * ms_targets[:, 0].numel())
                train_total += ms_targets[:, 0].numel()
                ms_batches += 1
                
                # CRITICAL: Free multi-step tensors immediately
                del loss, step_accs, ms_hist, ms_acts, ms_targets
            else:
                # === Standard single-step training with token importance weighting ===
                with torch.amp.autocast('cuda'):
                    logits, _ = model(batch_hist, batch_act, None)  # Get logits without loss
                    
                    # v2.0: Compute hybrid token importance weights
                    token_weights = compute_token_importance_weights(
                        batch_target, batch_hist, device,
                        token_embedding=model.token_embed,
                        use_hybrid=use_hybrid_weights,
                        # v2.0 hybrid params
                        motion_scale=motion_scale,
                        eventness_scale=eventness_scale,
                        persistence_scale=persistence_scale,
                        max_ratio=max_ratio,
                        # Legacy v1.x params (used if use_hybrid=False)
                        motion_weight=motion_weight,
                        continuous_bonus=continuous_bonus,
                        max_weight=max_weight,
                    )
                    
                    # Weighted cross-entropy
                    ce_per_token = F.cross_entropy(
                        logits.view(-1, model.n_vocab),
                        batch_target.view(-1),
                        reduction='none'
                    ).view_as(batch_target)  # (B, N)
                    
                    loss = (ce_per_token * token_weights).mean()
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                loss_val = loss.item()
                train_loss += loss_val
                preds = logits.argmax(dim=-1)
                train_correct += (preds == batch_target).sum().item()
                train_total += batch_target.numel()
                
                # Free single-step tensors
                del loss, logits, preds, token_weights, ce_per_token
            
            batch_count += 1
            
            # Memory check: run after first multi-step batch OR after 5 batches
            if epoch == 0 and batch_count <= 5:
                torch.cuda.synchronize()
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                mem_limit = 11.0  # GB - stop before spilling to shared memory
                if mem_gb > mem_limit:
                    print(f"\n  [ABORT] GPU memory {mem_gb:.1f}GB exceeds {mem_limit}GB limit!")
                    print(f"  Reduce rollout_steps or rollout_ratio and retry.")
                    return
                if batch_count == 5:
                    print(f"  Memory check (5 batches): {mem_gb:.1f}GB (limit: {mem_limit}GB) - OK")
            
            # Periodic memory cleanup
            if batch_count % 25 == 0:
                torch.cuda.empty_cache()
            
            ms_pct = f" ms:{ms_batches}" if ms_batches > 0 else ""
            pbar.set_postfix({'loss': f'{loss_val:.3f}', 'acc': f'{train_correct/train_total*100:.1f}%{ms_pct}'})
            
            if max_batches > 0 and batch_count >= max_batches:
                break
        
        scheduler.step()
        train_loss /= batch_count
        train_acc = train_correct / train_total * 100
        
        # Compute epoch stats
        epoch_time = time.time() - epoch_start
        epoch_its = batch_count / epoch_time
        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        
        # === Fast validation every epoch (quick trend check) ===
        model.eval()
        fast_val_correct = 0
        fast_val_total = 0
        fast_val_loss_sum = 0.0
        fast_val_batches = 0
        entropy_sum = 0.0
        max_prob_sum = 0.0
        n_entropy_samples = 0
        
        with torch.no_grad():
            for batch_hist, batch_act, batch_target in fast_val_loader:
                batch_hist = batch_hist.to(device)
                batch_act = batch_act.to(device)
                batch_target = batch_target.to(device)
                
                with torch.amp.autocast('cuda'):
                    logits, _ = model(batch_hist, batch_act, None)
                    loss = F.cross_entropy(logits.view(-1, model.n_vocab), batch_target.view(-1))
                
                fast_val_loss_sum += loss.item()
                fast_val_batches += 1
                
                preds = logits.argmax(dim=-1)
                fast_val_correct += (preds == batch_target).sum().item()
                fast_val_total += batch_target.numel()
                
                # Compute entropy and max probability stats (for temperature tuning)
                probs = F.softmax(logits.float(), dim=-1)  # (B, N, vocab)
                max_probs = probs.max(dim=-1).values  # (B, N)
                
                # Entropy: -sum(p * log(p)) per token position
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum(dim=-1)  # (B, N)
                
                entropy_sum += entropy.sum().item()
                max_prob_sum += max_probs.sum().item()
                n_entropy_samples += entropy.numel()
        
        fast_val_acc = fast_val_correct / fast_val_total * 100
        fast_val_loss = fast_val_loss_sum / fast_val_batches
        avg_entropy = entropy_sum / n_entropy_samples
        avg_max_prob = max_prob_sum / n_entropy_samples
        
        # === Gold Eval: Full validation every N epochs (for checkpointing) ===
        # Set full_val_every=999 to disable during quick iteration
        do_full_val = (epoch + 1) % full_val_every == 0
        
        if do_full_val:
            # Run full gold eval with rollout metrics
            gold_metrics = run_gold_eval(
                model=model,
                tokens=all_tokens,
                actions=all_actions,
                gold_suite=gold_suite,
                device=device,
                tier="gold",
                token_h=token_h,
                token_w=token_w,
            )
            
            val_acc = gold_metrics.accuracy
            val_loss = gold_metrics.loss
            last_val_acc = val_acc
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_epochs'].append(actual_epoch)
            
            # Report with rollout metrics and softmax stats (for temperature tuning)
            print(f"\n  Epoch {actual_epoch+1}: train={train_acc:.1f}%, fast_val={fast_val_acc:.1f}% | {epoch_its:.1f} it/s, {gpu_mem_gb:.1f}GB")
            print(f"  Softmax stats: entropy={avg_entropy:.3f}, max_prob={avg_max_prob:.3f}")
            print(f"  GOLD: acc={val_acc:.1f}%, imp={gold_metrics.important_token_accuracy:.1f}%, "
                  f"roll@5={gold_metrics.rollout_acc_h5:.1f}%, @10={gold_metrics.rollout_acc_h10:.1f}%, @20={gold_metrics.rollout_acc_h20:.1f}%")
            
            # Save best_full based on GOLD metrics (rollout accuracy is key for world models)
            # Use a weighted score: 50% 1-step acc + 50% average rollout acc
            rollout_avg = (gold_metrics.rollout_acc_h5 + gold_metrics.rollout_acc_h10 + gold_metrics.rollout_acc_h20) / 3
            gold_score = 0.5 * val_acc + 0.5 * rollout_avg
            
            if gold_score > best_val_acc:
                best_val_acc = gold_score
                torch.save({
                    'epoch': actual_epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'gold_score': gold_score,
                    'rollout_acc_h5': gold_metrics.rollout_acc_h5,
                    'rollout_acc_h10': gold_metrics.rollout_acc_h10,
                    'rollout_acc_h20': gold_metrics.rollout_acc_h20,
                    'important_token_acc': gold_metrics.important_token_accuracy,
                    'n_actions': n_actions,
                    'n_vocab': n_embeddings,
                    'token_h': token_h,
                    'token_w': token_w,
                    'n_layers': 10,
                }, f"{run_dir}/atari_world_model_best_full.pt")
                print(f"  * New BEST_GOLD! score={gold_score:.1f}% (1-step={val_acc:.1f}%, rollout_avg={rollout_avg:.1f}%)")
        else:
            val_acc = last_val_acc
            print(f"\n  Epoch {actual_epoch+1}: train={train_acc:.1f}%, fast_val={fast_val_acc:.1f}% | {epoch_its:.1f} it/s, {gpu_mem_gb:.1f}GB")
            print(f"  Softmax stats: entropy={avg_entropy:.3f}, max_prob={avg_max_prob:.3f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['fast_val_loss'].append(fast_val_loss)
        history['fast_val_acc'].append(fast_val_acc)
        history['entropy'].append(avg_entropy)
        history['max_prob'].append(avg_max_prob)
        history['its'].append(epoch_its)
        history['gpu_mem'].append(gpu_mem_gb)
        
        # === Checkpoint strategy ===
        # 1. Always save 'latest' every epoch (for resuming)
        torch.save({
            'epoch': actual_epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss if do_full_val else 0,
            'val_acc': val_acc,
            'fast_val_acc': fast_val_acc,
            'n_actions': n_actions,
            'n_vocab': n_embeddings,
            'token_h': token_h,
            'token_w': token_w,
            'n_layers': 10,
        }, f"{run_dir}/atari_world_model_hires.pt")
        
        # 2. Periodic safety checkpoint every 5 epochs
        if (actual_epoch + 1) % 5 == 0:
            torch.save({
                'epoch': actual_epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss if do_full_val else 0,
                'val_acc': val_acc,
                'fast_val_acc': fast_val_acc,
                'n_actions': n_actions,
                'n_vocab': n_embeddings,
                'token_h': token_h,
                'token_w': token_w,
                'n_layers': 10,
            }, f"{run_dir}/atari_world_model_hires_epoch{actual_epoch+1}.pt")
            print(f"  Periodic checkpoint saved (epoch {actual_epoch+1})")
    
    # Plot training
    _plot_training(history, run_dir)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best GOLD score: {best_val_acc:.1f}%")
    print(f"  (GOLD = 50% 1-step acc + 50% rollout avg)")
    print(f"  Run directory:   {run_dir}")
    print(f"  Latest:          {run_dir}/atari_world_model_hires.pt")
    print(f"  Best:            {run_dir}/atari_world_model_best_full.pt")
    print("=" * 60)


def _plot_training(history, save_dir):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # X-axis: epoch numbers (1-indexed)
    epochs = list(range(1, len(history['train_loss']) + 1))
    xlim = (0, len(epochs) + 1)
    
    # Detect if epoch 1 train is an outlier (fresh training from random init)
    # Train metrics are averaged DURING epoch (including random init batches),
    # while FastVal runs AFTER epoch completes. Skip epoch 1 train if outlier.
    skip_first_train = False
    if len(history['train_acc']) > 2 and history.get('fast_val_acc'):
        train_e1 = history['train_acc'][0]
        val_e1 = history['fast_val_acc'][0]
        # If train is <50% of val at epoch 1, it's the random init outlier
        if train_e1 < val_e1 * 0.6:
            skip_first_train = True
    
    train_epochs = epochs[1:] if skip_first_train else epochs
    train_loss = history['train_loss'][1:] if skip_first_train else history['train_loss']
    train_acc = history['train_acc'][1:] if skip_first_train else history['train_acc']
    
    # Loss plot
    axes[0, 0].plot(train_epochs, train_loss, 'b-o', label='Train' + (' (from e2)' if skip_first_train else ''), markersize=4)
    if history.get('fast_val_loss'):
        axes[0, 0].plot(epochs, history['fast_val_loss'], '-o', color='orange', label='FastVal', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(xlim)
    
    # Accuracy plot
    axes[0, 1].plot(train_epochs, train_acc, 'b-o', label='Train' + (' (from e2)' if skip_first_train else ''), markersize=4)
    if history.get('fast_val_acc'):
        axes[0, 1].plot(epochs, history['fast_val_acc'], '-o', color='orange', label='FastVal', markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy %')
    axes[0, 1].set_title('Token Prediction Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(xlim)
    
    # Entropy plot
    if history.get('entropy'):
        axes[1, 0].plot(epochs, history['entropy'], 'g-o', markersize=4)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Softmax Entropy (lower = confident)')
    axes[1, 0].set_xlim(xlim)
    
    # Max prob plot
    if history.get('max_prob'):
        axes[1, 1].plot(epochs, history['max_prob'], 'm-o', markersize=4)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Max Probability')
    axes[1, 1].set_title('Max Token Prob (higher = confident)')
    axes[1, 1].set_xlim(xlim)
    
    # Iterations per second
    if history.get('its'):
        axes[2, 0].plot(epochs, history['its'], 'c-o', markersize=4)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('it/s')
    axes[2, 0].set_title('Training Speed (it/s)')
    axes[2, 0].set_xlim(xlim)
    
    # GPU Memory
    if history.get('gpu_mem'):
        axes[2, 1].plot(epochs, history['gpu_mem'], 'r-o', markersize=4)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('GPU Memory (GB)')
    axes[2, 1].set_title('Peak GPU Memory')
    axes[2, 1].set_xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/atari_world_model_hires_training.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--max-batches', type=int, default=100, help='Max batches per epoch (0=all)')
    parser.add_argument('--resume', action='store_true', help='Resume from best checkpoint')
    parser.add_argument('--full-val-every', type=int, default=999, help='Full validation every N epochs (999=disabled)')
    # v1.3: Multi-step rollout training
    parser.add_argument('--rollout-steps', type=int, default=3, help='K steps to unroll (0 = disabled)')
    parser.add_argument('--rollout-ratio', type=float, default=0.3, help='Fraction of batches to use multi-step')
    # v2.0: Hybrid importance weighting
    parser.add_argument('--use-hybrid', action='store_true', default=True, help='Use hybrid v2.0 weighting (default: True)')
    parser.add_argument('--no-hybrid', dest='use_hybrid', action='store_false', help='Use legacy v1.x weighting')
    parser.add_argument('--motion-scale', type=float, default=2.0, help='Embedding-distance motion boost (default: 2.0)')
    parser.add_argument('--eventness-scale', type=float, default=2.0, help='Spike detection boost for fire/spawn (default: 2.0)')
    parser.add_argument('--persistence-scale', type=float, default=1.0, help='Multi-frame consistency boost (default: 1.0)')
    parser.add_argument('--max-ratio', type=float, default=6.0, help='Safe cap: 95th/median ratio limit (default: 6.0)')
    # Legacy v1.x params (only used with --no-hybrid)
    parser.add_argument('--motion-weight', type=float, default=4.0, help='[Legacy] Weight for moving tokens')
    parser.add_argument('--continuous-bonus', type=float, default=2.0, help='[Legacy] Bonus for continuous motion')
    parser.add_argument('--max-weight', type=float, default=8.0, help='[Legacy] Maximum weight cap')
    args = parser.parse_args()
    
    train_world_model_hires(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_batches=args.max_batches,
        resume=args.resume, 
        full_val_every=args.full_val_every,
        rollout_steps=args.rollout_steps,
        rollout_ratio=args.rollout_ratio,
        # v2.0 hybrid weighting
        use_hybrid_weights=args.use_hybrid,
        motion_scale=args.motion_scale,
        eventness_scale=args.eventness_scale,
        persistence_scale=args.persistence_scale,
        max_ratio=args.max_ratio,
        # Legacy v1.x params
        motion_weight=args.motion_weight,
        continuous_bonus=args.continuous_bonus,
        max_weight=args.max_weight,
    )
