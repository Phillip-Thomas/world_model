"""
Action Conditioning Diagnostic
==============================
Check whether the world model is actually using action information.

If shuffling actions barely changes loss, the model is NOT learning action conditioning.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from models.vqvae_hires import VQVAEHiRes
from models.temporal_world_model import TemporalVisualWorldModel
from data.atari_dataset import AtariTemporalDataset
from torch.utils.data import DataLoader, Subset


def check_action_conditioning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # === Load VQ-VAE ===
    vqvae_path = "checkpoints/v2/atari/atari_vqvae_hires.pt"
    print(f"\nLoading VQ-VAE from {vqvae_path}...")
    vqvae_ckpt = torch.load(vqvae_path, map_location=device, weights_only=True)
    n_embeddings = vqvae_ckpt.get('n_embeddings', 512)
    # Infer from checkpoint if not saved
    if 'model_state_dict' in vqvae_ckpt:
        n_embeddings = vqvae_ckpt['model_state_dict']['quantizer.embedding.weight'].shape[0]
    vqvae = VQVAEHiRes(
        in_channels=3, hidden_channels=64, latent_channels=256,
        n_embeddings=n_embeddings, input_size=(84, 64)
    )
    vqvae.load_state_dict(vqvae_ckpt['model_state_dict'])
    vqvae = vqvae.to(device)
    vqvae.eval()
    
    # === Load World Model ===
    wm_path = "checkpoints/v2/atari/atari_world_model_hires.pt"
    print(f"Loading World Model from {wm_path}...")
    
    checkpoint = torch.load(wm_path, map_location=device, weights_only=True)
    n_actions = checkpoint['n_actions']  # Must be in checkpoint
    # Infer n_vocab from checkpoint
    n_vocab = checkpoint['model_state_dict']['token_embed.weight'].shape[0]
    
    model = TemporalVisualWorldModel(
        n_vocab=n_vocab, n_actions=n_actions, d_model=256, n_heads=8, n_layers=10,
        token_h=21, token_w=16, max_history=4
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  n_actions: {n_actions}")
    
    # === Load Data ===
    data_path = "checkpoints/v2/atari/atari_game_data.npz"
    tokens_path = "checkpoints/v2/atari/atari_tokens_hires.npz"
    
    print(f"\nLoading data...")
    data = np.load(data_path)
    
    # Load or compute tokens
    if os.path.exists(tokens_path):
        all_tokens = np.load(tokens_path)['tokens']
    else:
        print("  Computing tokens...")
        frames = torch.from_numpy(data['frames']).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        all_tokens = []
        with torch.no_grad():
            for i in range(0, len(frames), 64):
                batch = frames[i:i+64].to(device)
                tokens = vqvae.encode(batch)
                all_tokens.append(tokens.cpu().numpy())
        all_tokens = np.concatenate(all_tokens, axis=0)
    
    # Flatten tokens
    all_tokens = all_tokens.reshape(len(all_tokens), -1)
    
    token_data = {
        'frames': all_tokens,
        'actions': data['actions'],
        'rewards': data['rewards'],
        'dones': data['dones'],
        'episode_starts': data['episode_starts'],
        'sample_weights': np.ones(len(all_tokens)),
        'n_actions': n_actions,
    }
    
    dataset = AtariTemporalDataset(token_data, history_len=4, preprocessed_tokens=True)
    
    # Get a decent-sized batch
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    
    # Dataset returns (history, action, next_frame) tuple
    hist, act, tgt = batch
    hist = hist.to(device)  # (B, 4, 336)
    act = act.to(device)    # (B,)
    tgt = tgt.to(device)    # (B, 336)
    
    print(f"\nBatch shapes: hist={hist.shape}, act={act.shape}, tgt={tgt.shape}")
    print(f"Action distribution in batch: {[f'{i}:{(act==i).sum().item()}' for i in range(n_actions)]}")
    
    # === Test 1: True vs Shuffled Actions ===
    print("\n" + "="*60)
    print("TEST 1: True Actions vs Shuffled Actions")
    print("="*60)
    
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # True actions
        logits, _ = model(hist, act, None)
        loss_true = F.cross_entropy(logits.reshape(-1, model.n_vocab), tgt.reshape(-1))
        
        # Shuffled actions
        perm = torch.randperm(act.size(0), device=act.device)
        logits2, _ = model(hist, act[perm], None)
        loss_shuf = F.cross_entropy(logits2.reshape(-1, model.n_vocab), tgt.reshape(-1))
    
    gap = (loss_shuf - loss_true).item()
    
    print(f"\nloss_true:  {loss_true.item():.6f}")
    print(f"loss_shuf:  {loss_shuf.item():.6f}")
    print(f"gap:        {gap:.6f}")
    print(f"gap %:      {100*gap/loss_true.item():.2f}%")
    
    if gap > 0.01:
        print("\n[GOOD] Model IS using action information (gap > 0.01)")
    elif gap > 0.001:
        print("\n[WEAK] Model barely uses actions (0.001 < gap < 0.01)")
    else:
        print("\n[BAD] Model ignores actions (gap ~ 0)")
        print("  Possible causes:")
        print("    - Action labels misaligned with transitions")
        print("    - Frame-skip mismatch")
        print("    - Action injection too weak")
    
    # === Test 2: Per-Action Loss ===
    print("\n" + "="*60)
    print("TEST 2: Per-Action Loss (should be similar if data is balanced)")
    print("="*60)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits, _ = model(hist, act, None)
        
        for a in range(n_actions):
            mask = (act == a)
            if mask.sum() > 0:
                action_loss = F.cross_entropy(
                    logits[mask].reshape(-1, model.n_vocab),
                    tgt[mask].reshape(-1)
                )
                print(f"  Action {a}: loss={action_loss.item():.4f} (n={mask.sum().item()})")
    
    # === Test 3: Constant Action vs True ===
    print("\n" + "="*60)
    print("TEST 3: Constant Action vs True (each action separately)")
    print("="*60)
    
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for const_act in range(n_actions):
            const_actions = torch.full_like(act, const_act)
            logits_const, _ = model(hist, const_actions, None)
            loss_const = F.cross_entropy(logits_const.reshape(-1, model.n_vocab), tgt.reshape(-1))
            diff = loss_const.item() - loss_true.item()
            print(f"  All-{action_names[const_act]}: loss={loss_const.item():.4f} (diff={diff:+.4f})")
    
    # === Test 4: Action-specific subset analysis ===
    print("\n" + "="*60)
    print("TEST 4: Shuffle only LEFT/RIGHT samples (paddle movement)")
    print("="*60)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # Only shuffle among samples where true action is LEFT or RIGHT
        move_mask = (act == 2) | (act == 3)  # RIGHT=2, LEFT=3
        n_move = move_mask.sum().item()
        
        if n_move > 1:
            # Get indices of movement actions
            move_indices = torch.where(move_mask)[0]
            
            # Create shuffled actions (only swap LEFT/RIGHT samples)
            act_partial_shuf = act.clone()
            perm_move = torch.randperm(len(move_indices), device=device)
            act_partial_shuf[move_indices] = act[move_indices[perm_move]]
            
            logits_ps, _ = model(hist, act_partial_shuf, None)
            loss_partial_shuf = F.cross_entropy(logits_ps.reshape(-1, model.n_vocab), tgt.reshape(-1))
            
            gap_move = (loss_partial_shuf - loss_true).item()
            print(f"  Movement samples: {n_move}")
            print(f"  loss_true:        {loss_true.item():.6f}")
            print(f"  loss_shuf_move:   {loss_partial_shuf.item():.6f}")
            print(f"  gap:              {gap_move:.6f}")
            
            if gap_move > 0.005:
                print("  [OK] Model distinguishes LEFT from RIGHT")
            else:
                print("  [WARN] Model may not distinguish LEFT from RIGHT well")
        else:
            print(f"  Not enough movement samples ({n_move})")


def check_data_alignment():
    """
    Check if actions in the data actually correlate with frame changes.
    If LEFT/RIGHT don't produce different paddle movements, the data is misaligned.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("DATA ALIGNMENT CHECK")
    print("="*60)
    
    # Load data
    data_path = "checkpoints/v2/atari/atari_game_data.npz"
    data = np.load(data_path)
    frames = data['frames']  # (N, H, W, 3) uint8
    actions = data['actions']
    
    print(f"Frames shape: {frames.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Note: frames has {len(frames)} entries, actions has {len(actions)} entries")
    print(f"      frames[i+1] should be result of actions[i] on frames[i]")
    
    # Focus on paddle region (bottom of frame)
    # For 84x64 frames, paddle is roughly in bottom 10 rows
    H, W = frames.shape[1], frames.shape[2]
    paddle_rows = slice(H - 12, H - 2)  # Bottom region where paddle is
    
    # Compute frame differences in paddle region for each action
    print("\nAnalyzing paddle movement by action...")
    
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    
    for action_id in [2, 3]:  # RIGHT=2, LEFT=3
        mask = actions == action_id
        indices = np.where(mask)[0]
        
        if len(indices) < 100:
            print(f"  {action_names[action_id]}: Not enough samples ({len(indices)})")
            continue
        
        # Sample some transitions
        sample_idx = indices[:500]
        
        # Compute horizontal center of mass shift in paddle region
        shifts = []
        for i in sample_idx:
            if i + 1 >= len(frames):
                continue
            
            # Get paddle region before and after
            before = frames[i, paddle_rows, :, :].astype(float).mean(axis=2)  # Grayscale
            after = frames[i+1, paddle_rows, :, :].astype(float).mean(axis=2)
            
            # Find brightest column (paddle location) before and after
            col_brightness_before = before.mean(axis=0)
            col_brightness_after = after.mean(axis=0)
            
            # Center of mass
            cols = np.arange(W)
            if col_brightness_before.sum() > 0 and col_brightness_after.sum() > 0:
                com_before = (cols * col_brightness_before).sum() / col_brightness_before.sum()
                com_after = (cols * col_brightness_after).sum() / col_brightness_after.sum()
                shift = com_after - com_before
                shifts.append(shift)
        
        if shifts:
            mean_shift = np.mean(shifts)
            std_shift = np.std(shifts)
            print(f"  {action_names[action_id]}: mean_shift={mean_shift:+.3f} px, std={std_shift:.3f} (n={len(shifts)})")
        else:
            print(f"  {action_names[action_id]}: Could not compute shifts")
    
    # Compare: RIGHT should shift positive, LEFT should shift negative
    print("\nExpected: RIGHT -> positive shift, LEFT -> negative shift")
    print("If both are ~0 or same sign, there's likely an alignment issue.")


if __name__ == "__main__":
    check_action_conditioning()
    check_data_alignment()

