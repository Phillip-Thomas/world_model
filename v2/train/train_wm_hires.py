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

v3.0: MDP Model Training
      - Reward prediction: two-headed (sign classification + magnitude regression)
      - Done prediction: binary classification
      - Combined loss: tokens + 0.1*reward + 0.1*done
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
from utils.importance_weights import (
    compute_hybrid_importance_weights, 
    compute_token_importance_weights,
    focal_loss_per_token,
    focal_loss_with_motion_weights,
)
from config import WorldModelConfig


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
    # v3.0: Focal loss
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    # v3.0 MDP: Reward/done targets
    target_rewards: torch.Tensor = None,  # (B, K) rewards for each step
    target_dones: torch.Tensor = None,    # (B, K) done flags for each step
) -> tuple:
    """
    Compute multi-step rollout loss with discounted step weights.
    
    v1.9: Later steps weighted less (1.0, 0.7, 0.49, 0.34, ...)
    v2.1: Normalized loss, scheduled sampling, importance weighting per step.
    v3.0: MDP model - includes reward and done prediction losses.
    
    Key improvements:
    - Loss normalized by weight sum (stable across K / discount values)
    - Teacher forcing mix (prevents "ball death spiral" early in training)
    - Token importance weights applied per rollout step
    - Efficient history update using roll() instead of cat()
    - Reward/done prediction losses (v3.0)
    
    Returns:
        total_loss: normalized discounted sum of losses across all steps
        step_accuracies: list of accuracies at each step
        aux_losses: dict with reward/done loss components (if targets provided)
    """
    B, T, N = history.shape
    K = actions.shape[1]
    
    total_loss = 0.0
    total_reward_loss = 0.0
    total_done_loss = 0.0
    weight_sum = 0.0  # v2.1: track for normalization
    step_accs = []
    current_history = history.clone()
    step_weight = 1.0
    
    for k in range(K):
        # Get action and target for this step
        action_k = actions[:, k]  # (B,)
        target_k = targets[:, k].contiguous()  # (B, N)
        
        # Get reward/done targets for this step if provided
        reward_k = target_rewards[:, k] if target_rewards is not None else None
        done_k = target_dones[:, k] if target_dones is not None else None
        
        # Forward pass with reward/done if available
        logits, step_loss, aux = model(
            current_history, action_k, target_k,
            target_rewards=reward_k,
            target_dones=done_k,
        )  # (B, N, vocab)
        
        # v2.1: Compute token importance weights if enabled
        if use_importance_weights and token_embedding is not None:
            token_weights = compute_hybrid_importance_weights(
                target_k, current_history, token_embedding, device,
                motion_scale=motion_scale,
                eventness_scale=eventness_scale,
                persistence_scale=persistence_scale,
                max_ratio=max_ratio,
            )
            # v3.0: Focal loss + motion weights
            if use_focal_loss:
                token_loss = focal_loss_with_motion_weights(
                    logits, target_k, token_weights, gamma=focal_gamma
                )
            else:
                # Standard weighted CE loss
                ce_per_token = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_k.reshape(-1),
                    reduction='none'
                ).reshape(B, N)
                token_loss = (ce_per_token * token_weights).mean()
        else:
            # Standard CE loss (no weighting)
            if use_focal_loss:
                token_loss = focal_loss_per_token(logits, target_k, gamma=focal_gamma, reduction='mean')
            else:
                token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_k.reshape(-1),
                )
        
        # Combine token loss with reward/done losses from model forward
        loss = token_loss
        if aux is not None:
            if 'reward_loss' in aux:
                loss = loss + 0.1 * aux['reward_loss']
                total_reward_loss += step_weight * aux['reward_loss'].item()
            if 'done_loss' in aux:
                loss = loss + 0.1 * aux['done_loss']
                total_done_loss += step_weight * aux['done_loss'].item()
        
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
    
    # v3.0: Return auxiliary losses for monitoring
    aux_losses = {}
    if target_rewards is not None:
        aux_losses['reward_loss'] = total_reward_loss / (weight_sum + 1e-8)
    if target_dones is not None:
        aux_losses['done_loss'] = total_done_loss / (weight_sum + 1e-8)
    
    return total_loss, step_accs, aux_losses


# Default runs directory (can be overridden by base_dir parameter)
DEFAULT_RUNS_DIR = "checkpoints/v2/atari/runs"


def resolve_resume_path(from_run: str = None, from_checkpoint: str = None, 
                        runs_dir: str = None) -> str:
    """
    Resolve checkpoint path for resuming training.
    
    Args:
        from_run: Run directory name (e.g., "20260110_111307") or "latest"
        from_checkpoint: Checkpoint name (e.g., "best", "epoch240") or None for latest
        runs_dir: Directory containing run folders (default: DEFAULT_RUNS_DIR)
        
    Returns:
        Full path to checkpoint file, or None if from_run not specified
    """
    if not from_run:
        return None
    
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    
    # Find run directory
    if from_run == "latest":
        if not os.path.exists(runs_dir):
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found in: {runs_dir}")
        run_dirs.sort(reverse=True)  # Timestamp format sorts correctly
        run_name = run_dirs[0]
    else:
        run_name = from_run
    
    run_path = os.path.join(runs_dir, run_name)
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # Find checkpoint
    if from_checkpoint:
        # Try multiple naming conventions
        candidates = [
            f"world_model_{from_checkpoint}.pt",              # best checkpoint (new)
            f"world_model_hires_{from_checkpoint}.pt",        # epoch checkpoints (new)
            f"atari_world_model_{from_checkpoint}.pt",        # legacy best
            f"atari_world_model_hires_{from_checkpoint}.pt",  # legacy epoch
            f"{from_checkpoint}.pt",                           # direct name
        ]
        for ckpt_name in candidates:
            ckpt_path = os.path.join(run_path, ckpt_name)
            if os.path.exists(ckpt_path):
                return ckpt_path
        raise FileNotFoundError(f"Checkpoint not found: {from_checkpoint} in {run_path}\n  Tried: {candidates}")
    else:
        # Find latest checkpoint by epoch number
        import glob
        import re
        # Find checkpoints (new naming + legacy)
        checkpoints = glob.glob(os.path.join(run_path, "world_model*.pt"))
        checkpoints += glob.glob(os.path.join(run_path, "atari_world_model*.pt"))
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
            elif basename in ("world_model_latest.pt", "atari_world_model_hires.pt") and best_epoch < 0:
                best_ckpt = ckpt
                best_epoch = 0
        
        if best_ckpt is None:
            raise FileNotFoundError(f"No valid checkpoint found in: {run_path}")
        return best_ckpt


def train_world_model_hires(
    base_dir: str = "checkpoints/v2/atari",  # Game-specific base directory
    n_epochs: int = 30,
    batch_size: int = 8,  # Reduced for K=5 rollouts
    learning_rate: float = 5e-4,  # Increased from 3e-4 to shake things up
    resume: bool = False,
    resume_path: str = None,  # Explicit path to resume from (overrides default)
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
    # v2.1: Teacher forcing for multi-step rollouts
    teacher_forcing_prob: float = 0.1,  # Probability of using true target in rollouts
    # v3.0: Focal loss (auto-upweights hard tokens)
    use_focal_loss: bool = True,      # Enable focal loss
    focal_gamma: float = 2.0,         # Focusing parameter (higher = more focus)
    # Legacy v1.x params (only used if use_hybrid_weights=False)
    motion_weight: float = 4.0,
    continuous_bonus: float = 2.0,
    max_weight: float = 8.0,
):
    """Train world model with high-res tokenization and multi-step rollouts."""
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Ensure base_dir exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Timestamped run directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = f"{base_dir}/wm_runs"
    run_dir = f"{runs_dir}/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save run config for reproducibility (using new config system)
    run_config = WorldModelConfig()
    run_config.timestamp = timestamp
    run_config.run_dir = run_dir
    run_config.vqvae_path = f"{base_dir}/vqvae_hires.pt"
    
    # Training params
    run_config.training.n_epochs = n_epochs
    run_config.training.batch_size = batch_size
    run_config.training.learning_rate = learning_rate
    run_config.training.max_batches = max_batches
    run_config.training.full_val_every = full_val_every
    run_config.training.rollout_steps = rollout_steps
    run_config.training.rollout_ratio = rollout_ratio
    run_config.training.teacher_forcing_prob = teacher_forcing_prob
    
    # Model params (hardcoded for now, loaded from VQ-VAE later)
    run_config.model.d_model = 256
    run_config.model.n_heads = 8
    run_config.model.n_layers = 10
    run_config.model.dropout = 0.1
    run_config.model.history_len = 4
    
    # Weighting params
    run_config.weighting.use_hybrid = use_hybrid_weights
    run_config.weighting.use_focal_loss = use_focal_loss
    run_config.weighting.focal_gamma = focal_gamma
    run_config.weighting.motion_scale = motion_scale
    run_config.weighting.eventness_scale = eventness_scale
    run_config.weighting.persistence_scale = persistence_scale
    run_config.weighting.max_ratio = max_ratio
    run_config.weighting.motion_weight = motion_weight
    run_config.weighting.continuous_bonus = continuous_bonus
    run_config.weighting.max_weight = max_weight
    
    run_config.save(f"{run_dir}/config.json")
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
    # Try game-agnostic name first, fall back to legacy name
    vqvae_path = f"{base_dir}/vqvae_hires.pt"
    if not os.path.exists(vqvae_path):
        vqvae_path = f"{base_dir}/atari_vqvae_hires.pt"  # Legacy compatibility
    vqvae_ckpt = torch.load(vqvae_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    input_h = vqvae_ckpt.get('input_h', 84)
    input_w = vqvae_ckpt.get('input_w', 64)
    n_embeddings = vqvae_ckpt.get('n_embeddings', 32)  # Match checkpoint
    
    # Infer hidden_channels from layer shapes if not saved
    hidden_channels = vqvae_ckpt.get('hidden_channels', None)
    if hidden_channels is None:
        # Infer from encoder.initial.weight shape: (hidden_channels, in_channels, 3, 3)
        initial_weight = vqvae_ckpt['model_state_dict'].get('encoder.initial.weight')
        if initial_weight is not None:
            hidden_channels = initial_weight.shape[0]
        else:
            hidden_channels = 64  # fallback default
    
    vqvae = VQVAEHiRes(
        in_channels=3, hidden_channels=hidden_channels, latent_channels=256,
        n_embeddings=n_embeddings, n_residual=2,
        input_size=(input_h, input_w),
    ).to(device)
    vqvae.load_state_dict(vqvae_ckpt['model_state_dict'])
    
    # CRITICAL: Freeze VQ-VAE to prevent codebook drift during world model training!
    vqvae.freeze_for_world_model()
    print(f"  Input: {input_h}x{input_w}, hidden={hidden_channels}, codes={n_embeddings}")
    
    # === Load Data (with optional human gameplay merge) ===
    print("\nLoading game data...")
    # Try game-agnostic name first, fall back to legacy name
    data_path = f"{base_dir}/game_data.npz"
    if not os.path.exists(data_path):
        data_path = f"{base_dir}/atari_game_data.npz"  # Legacy compatibility
    data = dict(np.load(data_path, allow_pickle=True))
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
    # Try game-agnostic name first, fall back to legacy name
    tokens_path = f"{base_dir}/tokens.npz"
    legacy_tokens_path = f"{base_dir}/atari_tokens_hires.npz"
    if os.path.exists(legacy_tokens_path) and not os.path.exists(tokens_path):
        tokens_path = legacy_tokens_path
    
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
            rewards=data['rewards'],  # v3.0: Include rewards for MDP training
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
    best_fast_val_acc = 0  # Track best fast_val for checkpointing
    history = {'train_loss': [], 'train_acc': [], 'fast_val_loss': [], 'fast_val_acc': [], 'entropy': [], 'max_prob': [], 'its': [], 'gpu_mem': [], 'val_loss': [], 'val_acc': [], 'val_epochs': []}
    
    resumed_from_path = None
    if resume:
        # Use explicit resume_path if provided, otherwise fall back to base_dir
        if resume_path:
            ckpt_path = resume_path
        else:
            # Try new naming first, fall back to legacy
            ckpt_path = f"{base_dir}/world_model_best.pt"
            if not os.path.exists(ckpt_path):
                ckpt_path = f"{base_dir}/atari_world_model_hires.pt"  # Legacy
        
        if os.path.exists(ckpt_path):
            print(f"\n  Loading checkpoint from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_acc = ckpt.get('val_acc', 0)
            best_fast_val_acc = ckpt.get('fast_val_acc', 0)
            resumed_from_path = ckpt_path
            print(f"  Resuming from epoch {start_epoch}, best_fast_val_acc={best_fast_val_acc:.1f}%")
        else:
            print(f"  No checkpoint found at {ckpt_path}, starting fresh")
    
    # Update config with runtime info from VQ-VAE and model
    run_config.model.n_vocab = n_embeddings
    run_config.model.n_actions = n_actions
    run_config.model.token_h = token_h
    run_config.model.token_w = token_w
    
    # Re-save config with full info
    run_config.save(f"{run_dir}/config.json")
    
    # Write session header to stats file
    _write_session_header(
        save_dir=run_dir,
        start_epoch=start_epoch + 1,
        end_epoch=start_epoch + n_epochs,
        resumed_from=resumed_from_path,
        resumed_epoch=start_epoch if resumed_from_path else None,
        resumed_acc=best_fast_val_acc if resumed_from_path else None
    )
    
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
        for batch_data in pbar:
            # v3.0: Dataset now returns 5-tuple (history, action, target, reward, done)
            batch_hist, batch_act, batch_target, batch_reward, batch_done = batch_data
            batch_hist = batch_hist.to(device)
            batch_act = batch_act.to(device)
            batch_target = batch_target.to(device)
            batch_reward = batch_reward.to(device)
            batch_done = batch_done.to(device)
            
            # Decide: single-step or multi-step?
            use_multistep = (
                ms_iter is not None and 
                np.random.random() < rollout_ratio
            )
            
            if use_multistep:
                # === Multi-step rollout training ===
                try:
                    ms_data = next(ms_iter)
                except StopIteration:
                    ms_iter = iter(multistep_loader)
                    ms_data = next(ms_iter)
                
                # v3.0: Multi-step dataset now returns 5-tuple
                ms_hist, ms_acts, ms_targets, ms_rewards, ms_dones = ms_data
                ms_hist = ms_hist.to(device)
                ms_acts = ms_acts.to(device)
                ms_targets = ms_targets.to(device)
                ms_rewards = ms_rewards.to(device) if ms_rewards is not None else None
                ms_dones = ms_dones.to(device)
                
                with torch.amp.autocast('cuda'):
                    loss, step_accs, aux_losses = multistep_rollout_loss(
                        model, ms_hist, ms_acts, ms_targets, device,
                        teacher_forcing_prob=teacher_forcing_prob,
                        token_embedding=vqvae.quantizer.embedding,  # VQ-VAE codebook (stable, visual similarity)
                        use_importance_weights=use_hybrid_weights,
                        motion_scale=motion_scale,
                        eventness_scale=eventness_scale,
                        persistence_scale=persistence_scale,
                        max_ratio=max_ratio,
                        use_focal_loss=use_focal_loss,
                        focal_gamma=focal_gamma,
                        target_rewards=ms_rewards,  # v3.0 MDP
                        target_dones=ms_dones,      # v3.0 MDP
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
                del loss, step_accs, aux_losses, ms_hist, ms_acts, ms_targets, ms_rewards, ms_dones
            else:
                # === Standard single-step training with token importance weighting ===
                with torch.amp.autocast('cuda'):
                    # v3.0: Pass reward/done targets for MDP training
                    logits, model_loss, aux = model(
                        batch_hist, batch_act, None,
                        target_rewards=batch_reward,
                        target_dones=batch_done,
                        compute_reward_done=True,
                    )
                    
                    # v2.0: Compute hybrid token importance weights
                    token_weights = compute_token_importance_weights(
                        batch_target, batch_hist, device,
                        token_embedding=vqvae.quantizer.embedding,  # VQ-VAE codebook (stable, visual similarity)
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
                    
                    # v3.0: Focal loss + motion weights (or standard weighted CE)
                    if use_focal_loss:
                        # Focal loss automatically upweights hard tokens
                        token_loss = focal_loss_with_motion_weights(
                            logits, batch_target, token_weights, gamma=focal_gamma
                        )
                    else:
                        # Standard weighted cross-entropy
                        ce_per_token = F.cross_entropy(
                            logits.reshape(-1, model.n_vocab),
                            batch_target.reshape(-1),
                            reduction='none'
                        ).reshape_as(batch_target)  # (B, N)
                        token_loss = (ce_per_token * token_weights).mean()
                    
                    # v3.0: Combine token loss with reward/done losses from model
                    loss = token_loss
                    if aux is not None:
                        if 'reward_loss' in aux:
                            loss = loss + 0.1 * aux['reward_loss']
                        if 'done_loss' in aux:
                            loss = loss + 0.1 * aux['done_loss']
                
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
                
                # Free single-step tensors (ce_per_token only exists in non-focal path)
                del loss, logits, preds, token_weights, aux
            
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
            for batch_data in fast_val_loader:
                # v3.0: Dataset now returns 5-tuple
                batch_hist, batch_act, batch_target, batch_reward, batch_done = batch_data
                batch_hist = batch_hist.to(device)
                batch_act = batch_act.to(device)
                batch_target = batch_target.to(device)
                
                with torch.amp.autocast('cuda'):
                    logits, _, _ = model(batch_hist, batch_act, None)
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
        
        # === Save best checkpoint based on fast_val_acc ===
        if fast_val_acc > best_fast_val_acc:
            best_fast_val_acc = fast_val_acc
            torch.save({
                'epoch': actual_epoch,
                'model_state_dict': model.state_dict(),
                'fast_val_acc': fast_val_acc,
                'fast_val_loss': fast_val_loss,
                'n_actions': n_actions,
                'n_vocab': n_embeddings,
                'token_h': token_h,
                'token_w': token_w,
                'n_layers': 10,
            }, f"{run_dir}/world_model_best.pt")
            # Also copy to base_dir for easy access
            torch.save({
                'epoch': actual_epoch,
                'model_state_dict': model.state_dict(),
                'fast_val_acc': fast_val_acc,
                'fast_val_loss': fast_val_loss,
                'n_actions': n_actions,
                'n_vocab': n_embeddings,
                'token_h': token_h,
                'token_w': token_w,
                'n_layers': 10,
            }, f"{base_dir}/world_model_best.pt")
            print(f"  * New BEST! fast_val_acc={fast_val_acc:.2f}% (epoch {actual_epoch+1})")
        
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
                }, f"{run_dir}/world_model_best_gold.pt")
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
        
        # Write epoch stats to text file
        epoch_stats = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': fast_val_loss,
            'val_acc': fast_val_acc,
            'entropy': avg_entropy,
            'max_prob': avg_max_prob,
            'its': epoch_its,
            'gpu_mem': gpu_mem_gb,
        }
        is_new_best = (fast_val_acc >= best_fast_val_acc)
        _write_epoch_stats(actual_epoch + 1, epoch_stats, run_dir, is_best=is_new_best)
        
        # Update training plot every epoch (cheap - ~0.1s)
        _plot_training(history, run_dir)
        
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
        }, f"{run_dir}/world_model_latest.pt")
        
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
            }, f"{run_dir}/world_model_epoch{actual_epoch+1}.pt")
            print(f"  Periodic checkpoint saved (epoch {actual_epoch+1})")
    
    # Plot training
    _plot_training(history, run_dir)
    
    # Generate rollout visualization
    _plot_rollout_comparison(model, vqvae, val_dataset, device, run_dir, 
                            rollout_steps=min(rollout_steps, 8))
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best GOLD score: {best_val_acc:.1f}%")
    print(f"  (GOLD = 50% 1-step acc + 50% rollout avg)")
    print(f"  Run directory:   {run_dir}")
    print(f"  Latest:          {run_dir}/world_model_latest.pt")
    print(f"  Best:            {run_dir}/world_model_best_gold.pt")
    print("=" * 60)


def _write_session_header(save_dir: str, start_epoch: int, end_epoch: int, 
                          resumed_from: str = None, resumed_epoch: int = None,
                          resumed_acc: float = None):
    """Write a session header to the stats file when training starts/resumes."""
    from datetime import datetime
    stats_file = os.path.join(save_dir, "training_stats.txt")
    
    with open(stats_file, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Training Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if resumed_from:
            f.write(f"  RESUMED FROM: {resumed_from}\n")
            f.write(f"  Resume Epoch: {resumed_epoch}, Acc: {resumed_acc:.2f}%\n")
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
            f.write("Training Statistics Log\n")
            f.write("=" * 80 + "\n\n")
    
    with open(stats_file, 'a') as f:
        best_marker = " *BEST*" if is_best else ""
        f.write(f"Epoch {epoch}{best_marker}\n")
        f.write(f"  Train Loss: {stats['train_loss']:.4f}  |  Train Acc: {stats['train_acc']:.2f}%\n")
        f.write(f"  Val Loss:   {stats['val_loss']:.4f}  |  Val Acc:   {stats['val_acc']:.2f}%\n")
        f.write(f"  Entropy: {stats['entropy']:.3f}  |  Max Prob: {stats['max_prob']:.3f}\n")
        f.write(f"  Speed: {stats['its']:.1f} it/s  |  GPU Mem: {stats['gpu_mem']:.1f} GB\n")
        f.write("-" * 40 + "\n")


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
    plt.savefig(f"{save_dir}/world_model_training.png", dpi=150)
    plt.close()


def _plot_rollout_comparison(model, vqvae, val_dataset, device, save_dir, rollout_steps=5):
    """
    Generate visual comparison of ground truth vs model rollout predictions.
    Shows K-step autoregressive prediction to visualize ball/dynamics quality.
    """
    try:
        model.eval()
        vqvae.eval()
        
        # Get token grid dimensions
        token_h, token_w = 21, 16  # Default for 84x64
        
        # Handle Subset wrapper - get underlying dataset
        base_dataset = val_dataset.dataset if hasattr(val_dataset, 'dataset') else val_dataset
        
        # Find a good sample (one with motion in the sequence)
        np.random.seed(42)  # Reproducible sample selection
        sample_idx = np.random.randint(0, len(val_dataset))
        
        # Get sample: history, action, target
        history, action, target = val_dataset[sample_idx]
        history = history.unsqueeze(0).to(device)  # (1, T, N)
        
        # We need K actions and K targets - get consecutive samples
        # Access tokens/actions from base dataset (tokens are stored as 'frames' in AtariTemporalDataset)
        tokens = base_dataset.frames  # Pre-tokenized frames
        actions = base_dataset.actions
        
        # Find start index for this sample
        start_idx = sample_idx
        
        # Build rollout data
        K = min(rollout_steps, 8)
        
        # Use simpler approach: just run K steps from this history
        with torch.no_grad():
            current_history = history.clone()
            gt_frames = []
            pred_frames = []
            
            # Get ground truth frames and run predictions
            for k in range(K):
                # Get action for this step (use random if not available)
                if start_idx + k < len(actions):
                    action_k = actions[start_idx + k].unsqueeze(0).to(device)
                else:
                    action_k = torch.randint(0, 4, (1,), device=device)
                
                # Get ground truth target
                target_idx = base_dataset.frame_index[start_idx] + k + 1 if hasattr(base_dataset, 'frame_index') else start_idx + k + 1
                if target_idx < len(tokens):
                    gt_tok = tokens[target_idx]
                    # Handle both (H*W,) and (H, W) shapes
                    if gt_tok.ndim == 1:
                        gt_tokens = gt_tok.reshape(1, token_h, token_w).to(device)
                    else:
                        gt_tokens = gt_tok.unsqueeze(0).to(device)
                else:
                    gt_tok = tokens[-1]
                    if gt_tok.ndim == 1:
                        gt_tokens = gt_tok.reshape(1, token_h, token_w).to(device)
                    else:
                        gt_tokens = gt_tok.unsqueeze(0).to(device)
                
                # Decode ground truth
                gt_frame = vqvae.decode(gt_tokens)  # (1, C, H, W)
                gt_img = gt_frame.squeeze().cpu().numpy()  # (C, H, W)
                if gt_img.ndim == 3:
                    gt_img = np.transpose(gt_img, (1, 2, 0))  # (H, W, C) for matplotlib
                    gt_img = (gt_img + 1) / 2  # [-1, 1] -> [0, 1]
                    gt_img = np.clip(gt_img, 0, 1)
                gt_frames.append(gt_img)
                
                # Model prediction
                logits, _, _ = model(current_history, action_k, None)
                pred_tokens = logits.argmax(dim=-1)  # (1, N)
                
                # Decode prediction
                pred_tokens_2d = pred_tokens.reshape(1, token_h, token_w)
                pred_frame = vqvae.decode(pred_tokens_2d)  # (1, C, H, W)
                pred_img = pred_frame.squeeze().cpu().numpy()  # (C, H, W)
                if pred_img.ndim == 3:
                    pred_img = np.transpose(pred_img, (1, 2, 0))  # (H, W, C) for matplotlib
                    pred_img = (pred_img + 1) / 2  # [-1, 1] -> [0, 1]
                    pred_img = np.clip(pred_img, 0, 1)
                pred_frames.append(pred_img)
                
                # Update history with prediction (autoregressive)
                new_history = current_history.roll(shifts=-1, dims=1)
                new_history[:, -1, :] = pred_tokens
                current_history = new_history
        
        # Create visualization: 2 rows (GT, Pred) x K columns
        fig, axes = plt.subplots(2, K, figsize=(K * 2, 4))
        
        if K == 1:
            axes = axes.reshape(2, 1)
        
        for k in range(K):
            # Ground truth row
            axes[0, k].imshow(gt_frames[k])
            axes[0, k].set_title(f't+{k+1}' if k == 0 else f'+{k+1}')
            axes[0, k].axis('off')
            if k == 0:
                axes[0, k].set_ylabel('Ground Truth', fontsize=10)
            
            # Prediction row
            axes[1, k].imshow(pred_frames[k])
            axes[1, k].axis('off')
            if k == 0:
                axes[1, k].set_ylabel('Model Pred', fontsize=10)
        
        # Add row labels
        fig.text(0.02, 0.75, 'Ground\nTruth', ha='center', va='center', fontsize=10, fontweight='bold')
        fig.text(0.02, 0.25, 'Model\nPred', ha='center', va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'{K}-Step Autoregressive Rollout Comparison', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/rollout_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Rollout comparison saved to {save_dir}/rollout_comparison.png")
    except Exception as e:
        print(f"  [WARNING] Could not generate rollout comparison: {e}")


if __name__ == "__main__":
    import argparse
    from config import WorldModelConfig
    
    parser = argparse.ArgumentParser(
        description='Train World Model with configurable hyperparameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Game-specific directory
    parser.add_argument('--base-dir', type=str, default="checkpoints/v2/atari",
                        help='Base directory for game data (e.g., checkpoints/v2/mspacman)')
    
    # All args use None as default so we can detect explicit overrides
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--max-batches', type=int, default=None, help='Max batches per epoch (0=all)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--from-run', type=str, default=None, 
                        help='Run directory to resume from (e.g., "20260110_111307" or "latest")')
    parser.add_argument('--from-checkpoint', type=str, default=None,
                        help='Checkpoint within run (e.g., "best", "epoch240"). Default: latest epoch')
    parser.add_argument('--full-val-every', type=int, default=None, help='Full validation every N epochs')
    
    # Multi-step rollout training
    parser.add_argument('--rollout-steps', type=int, default=None, help='K steps to unroll (0=disabled)')
    parser.add_argument('--rollout-ratio', type=float, default=None, help='Fraction of batches for multi-step')
    parser.add_argument('--teacher-forcing', type=float, default=None, help='Teacher forcing probability')
    
    # Hybrid importance weighting
    parser.add_argument('--use-hybrid', action='store_true', default=None, help='Use hybrid v2.0 weighting')
    parser.add_argument('--no-hybrid', dest='use_hybrid', action='store_false', help='Use legacy v1.x weighting')
    parser.add_argument('--motion-scale', type=float, default=None, help='Embedding-distance motion boost')
    parser.add_argument('--eventness-scale', type=float, default=None, help='Spike detection boost')
    parser.add_argument('--persistence-scale', type=float, default=None, help='Multi-frame consistency boost')
    parser.add_argument('--max-ratio', type=float, default=None, help='Safe cap: 95th/median ratio limit')
    
    # v3.0: Focal loss
    parser.add_argument('--focal-loss', action='store_true', default=None, help='Enable focal loss (default: True)')
    parser.add_argument('--no-focal-loss', dest='focal_loss', action='store_false', help='Disable focal loss')
    parser.add_argument('--focal-gamma', type=float, default=None, help='Focal loss gamma (higher=more focus on hard)')
    
    # Legacy v1.x params
    parser.add_argument('--motion-weight', type=float, default=None, help='[Legacy] Weight for moving tokens')
    parser.add_argument('--continuous-bonus', type=float, default=None, help='[Legacy] Bonus for continuous motion')
    parser.add_argument('--max-weight', type=float, default=None, help='[Legacy] Maximum weight cap')
    
    args = parser.parse_args()
    
    # Load defaults, then override with CLI args
    config = WorldModelConfig()
    config.update_from_args(args)
    
    # Handle focal loss args
    if args.focal_loss is not None:
        config.weighting.use_focal_loss = args.focal_loss
    if args.focal_gamma is not None:
        config.weighting.focal_gamma = args.focal_gamma
    
    # Resolve resume path if --from-run specified
    resume_path = None
    runs_dir = f"{args.base_dir}/wm_runs"
    if args.from_run:
        args.resume = True  # Automatically enable resume if --from-run specified
        resume_path = resolve_resume_path(args.from_run, args.from_checkpoint, runs_dir=runs_dir)
        print(f"\n  Resume from: {resume_path}")
    
    # Print effective config
    print("\n" + "=" * 60)
    print("Effective Configuration (defaults + CLI overrides)")
    print("=" * 60)
    print(f"  Training: epochs={config.training.n_epochs}, batch={config.training.batch_size}, lr={config.training.learning_rate}")
    print(f"  Rollout: steps={config.training.rollout_steps}, ratio={config.training.rollout_ratio}, tf={config.training.teacher_forcing_prob}")
    print(f"  Weighting: hybrid={config.weighting.use_hybrid}, motion={config.weighting.motion_scale}, max_ratio={config.weighting.max_ratio}")
    print(f"  Focal Loss: enabled={config.weighting.use_focal_loss}, gamma={config.weighting.focal_gamma}")
    print("=" * 60 + "\n")
    
    train_world_model_hires(
        base_dir=args.base_dir,
        n_epochs=config.training.n_epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        max_batches=config.training.max_batches,
        resume=args.resume,
        resume_path=resume_path,
        full_val_every=config.training.full_val_every,
        rollout_steps=config.training.rollout_steps,
        rollout_ratio=config.training.rollout_ratio,
        # v2.0 hybrid weighting
        use_hybrid_weights=config.weighting.use_hybrid,
        motion_scale=config.weighting.motion_scale,
        eventness_scale=config.weighting.eventness_scale,
        persistence_scale=config.weighting.persistence_scale,
        max_ratio=config.weighting.max_ratio,
        # v2.1 teacher forcing
        teacher_forcing_prob=config.training.teacher_forcing_prob,
        # v3.0 focal loss
        use_focal_loss=config.weighting.use_focal_loss,
        focal_gamma=config.weighting.focal_gamma,
        # Legacy v1.x params
        motion_weight=config.weighting.motion_weight,
        continuous_bonus=config.weighting.continuous_bonus,
        max_weight=config.weighting.max_weight,
    )
