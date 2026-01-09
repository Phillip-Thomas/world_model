"""
Full Pipeline Test
==================
Tests all components end-to-end:
1. Data collection (84x64 aspect-preserved)
2. VQ-VAE tokenization (21x16 tokens)
3. World Model (10 layers)
4. Dataset & DataLoader
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np


def test_data_collection():
    """Test aspect-preserved data collection."""
    print("\n[1/4] Testing Data Collection (84x64 aspect-preserved)...")
    
    from v2.data.atari_dataset import AtariConfig, AtariCollector
    
    config = AtariConfig(
        game='ALE/Breakout-v5',
        n_episodes=3,
        max_steps=100,
        preserve_aspect=True,
        target_width=64,
    )
    print(f"  Target size: {config.target_size}")
    
    collector = AtariCollector(config)
    data = collector.collect(show_progress=False)
    
    print(f"  Collected: {data['frames'].shape}")
    assert data['frames'].shape[1:3] == (84, 64), "Wrong frame size!"
    print("  [OK] Data collection working!")
    
    return data


def test_vqvae(data):
    """Test VQ-VAE tokenization."""
    print("\n[2/4] Testing VQ-VAE (84x64 -> 21x16 tokens)...")
    
    from v2.models.vqvae_hires import VQVAEHiRes
    
    vqvae = VQVAEHiRes(input_size=(84, 64))
    print(f"  Token grid: {vqvae.token_h}x{vqvae.token_w} = {vqvae.n_tokens}")
    
    # Tokenize a batch
    frames_batch = torch.from_numpy(data['frames'][:8]).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    
    with torch.no_grad():
        tokens = vqvae.encode(frames_batch)
        recon = vqvae.decode(tokens)
    
    print(f"  Input: {frames_batch.shape} -> Tokens: {tokens.shape} -> Recon: {recon.shape}")
    assert tokens.shape == (8, 21, 16), f"Wrong token shape: {tokens.shape}"
    print("  [OK] VQ-VAE working!")
    
    return vqvae


def test_world_model():
    """Test world model with 10 layers."""
    print("\n[3/4] Testing World Model (10 layers, 21x16 tokens)...")
    
    from v2.models.temporal_world_model import TemporalVisualWorldModel
    
    world_model = TemporalVisualWorldModel(
        n_vocab=512,
        n_actions=4,
        d_model=256,
        n_heads=8,
        n_layers=10,
        token_h=21,
        token_w=16,
    )
    
    n_params = sum(p.numel() for p in world_model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Layers: 10")
    
    # Test forward pass
    B, T, N = 4, 4, 336
    frame_history = torch.randint(0, 512, (B, T, N))
    actions = torch.randint(0, 4, (B,))
    target = torch.randint(0, 512, (B, N))
    
    logits, loss = world_model(frame_history, actions, target)
    
    print(f"  Forward: history {frame_history.shape} -> logits {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    assert logits.shape == (B, N, 512), f"Wrong logits shape: {logits.shape}"
    print("  [OK] World Model working!")
    
    return world_model


def test_full_pipeline(data, vqvae, world_model):
    """Test full pipeline: data -> tokens -> world model."""
    print("\n[4/4] Testing Full Pipeline...")
    
    from v2.data.atari_dataset import AtariTemporalDataset
    from torch.utils.data import DataLoader
    
    # Tokenize all frames
    print("  Tokenizing frames...")
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(data['frames']), 32):
            batch = torch.from_numpy(data['frames'][i:i+32]).float()
            batch = batch.permute(0, 3, 1, 2) / 127.5 - 1.0
            toks = vqvae.encode(batch)
            all_tokens.append(toks.numpy())
    all_tokens = np.concatenate(all_tokens, axis=0)
    print(f"  Tokenized: {all_tokens.shape}")
    
    # Create dataset
    token_data = {
        'frames': all_tokens,
        'actions': data['actions'],
        'rewards': data['rewards'],
        'dones': data['dones'],
        'episode_starts': data['episode_starts'],
        'sample_weights': data['sample_weights'],
        'n_actions': 4,
    }
    dataset = AtariTemporalDataset(token_data, history_len=4, preprocessed_tokens=True)
    print(f"  Dataset size: {len(dataset)}")
    
    # Test a batch
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    hist, act, target = next(iter(loader))
    print(f"  Batch: history={hist.shape}, action={act.shape}, target={target.shape}")
    
    # Forward through world model
    logits, loss = world_model(hist, act, target)
    preds = logits.argmax(dim=-1)
    acc = (preds == target).float().mean() * 100
    
    print(f"  World model: logits={logits.shape}, loss={loss.item():.4f}, acc={acc:.1f}%")
    print("  [OK] Full pipeline working!")


def main():
    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)
    
    # Run all tests
    data = test_data_collection()
    vqvae = test_vqvae(data)
    world_model = test_world_model()
    test_full_pipeline(data, vqvae, world_model)
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  - Data: 84x64 aspect-preserved frames")
    print("  - VQ-VAE: 84x64 -> 21x16 = 336 tokens")
    print("  - World Model: 10 layers, ~8.2M parameters")
    print()
    print("Ready to collect full dataset and train!")


if __name__ == "__main__":
    main()

