"""
Test the new aspect-ratio-preserving dataloader.

Quick validation:
1. Collects small dataset with correct aspect ratio
2. Visualizes frames to verify no skewing
3. Tests dataloader performance
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from v2.data.atari_dataset import (
    AtariConfig, 
    AtariCollector, 
    AtariTemporalDataset,
    normalize_batch,
    compute_aspect_preserving_size,
    ATARI_NATIVE_SIZES
)
from torch.utils.data import DataLoader


def test_aspect_ratio():
    """Test that aspect ratio is computed correctly."""
    print("=" * 60)
    print("TEST 1: Aspect Ratio Computation")
    print("=" * 60)
    
    native = ATARI_NATIVE_SIZES['ALE/Breakout-v5']
    print(f"  Native Breakout size: {native[0]}x{native[1]} (H×W)")
    print(f"  Aspect ratio: {native[0]/native[1]:.4f}")
    
    target = compute_aspect_preserving_size(native[0], native[1], target_width=64)
    print(f"  Target size (width=64): {target[0]}x{target[1]} (H×W)")
    print(f"  Target aspect ratio: {target[0]/target[1]:.4f}")
    
    # Verify
    expected_h = int(64 * native[0] / native[1])  # 84
    expected_h = ((expected_h + 2) // 4) * 4  # Round to mult of 4
    assert target == (expected_h, 64), f"Expected ({expected_h}, 64), got {target}"
    print("  [OK] Aspect ratio preserved correctly!")
    return target


def test_small_collection(target_size):
    """Collect a small dataset and verify frame shapes."""
    print("\n" + "=" * 60)
    print("TEST 2: Small Data Collection")
    print("=" * 60)
    
    config = AtariConfig(
        game="ALE/Breakout-v5",
        n_episodes=3,
        max_steps=100,
        preserve_aspect=True,
        target_width=64,
        action_weights=[0.05, 0.25, 0.35, 0.35],  # Less NOOP
    )
    
    print(f"  Config target_size: {config.target_size}")
    assert config.target_size == target_size, f"Config size mismatch"
    
    collector = AtariCollector(config)
    
    start = time.time()
    data = collector.collect(show_progress=True)
    elapsed = time.time() - start
    
    print(f"\n  Collection time: {elapsed:.1f}s")
    print(f"  Frames shape: {data['frames'].shape}")
    print(f"  Actions shape: {data['actions'].shape}")
    print(f"  Sample weights range: [{data['sample_weights'].min():.2f}, {data['sample_weights'].max():.2f}]")
    
    # Verify shape
    assert data['frames'].shape[1:3] == target_size, f"Frame shape mismatch"
    print("  [OK] Frame shapes correct!")
    
    return data


def test_dataloader_speed(data):
    """Test dataloader iteration speed."""
    print("\n" + "=" * 60)
    print("TEST 3: DataLoader Performance")
    print("=" * 60)
    
    dataset = AtariTemporalDataset(data, history_len=4)
    print(f"  Dataset size: {len(dataset)}")
    
    loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0,  # Windows
        pin_memory=True
    )
    
    # Time a few iterations
    n_batches = min(20, len(loader))
    start = time.time()
    
    for i, (history, actions, next_frames) in enumerate(loader):
        if i >= n_batches:
            break
        # Simulate GPU transfer and normalization
        if torch.cuda.is_available():
            history = history.cuda(non_blocking=True)
            next_frames = next_frames.cuda(non_blocking=True)
        history = normalize_batch(history)
        next_frames = normalize_batch(next_frames)
    
    elapsed = time.time() - start
    batches_per_sec = n_batches / elapsed
    
    print(f"  Batch shape: history={history.shape}, next={next_frames.shape}")
    print(f"  Data type after normalize: {history.dtype}")
    print(f"  Value range: [{history.min():.2f}, {history.max():.2f}]")
    print(f"  Speed: {batches_per_sec:.1f} batches/sec ({n_batches} batches in {elapsed:.2f}s)")
    print("  [OK] DataLoader working!")
    
    return dataset


def visualize_frames(data, dataset):
    """Create visualization of frames to verify no skewing."""
    print("\n" + "=" * 60)
    print("TEST 4: Visual Verification")
    print("=" * 60)
    
    os.makedirs("checkpoints/v2/atari", exist_ok=True)
    
    # Get a sample sequence
    history, action, next_frame = dataset[len(dataset)//2]
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    
    # Top row: raw frames from data
    fig.suptitle(f'Aspect-Preserved Frames: {data["frames"].shape[1]}×{data["frames"].shape[2]} (native 210×160)', 
                 fontsize=14, fontweight='bold')
    
    # Show 5 random raw frames
    indices = np.random.choice(len(data['frames']), 5, replace=False)
    for i, idx in enumerate(indices):
        axes[0, i].imshow(data['frames'][idx])
        axes[0, i].set_title(f'Frame {idx}')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Raw Frames', fontsize=12)
    
    # Bottom row: dataset sequence
    for i in range(4):
        frame = history[i].permute(1, 2, 0).numpy()  # uint8
        axes[1, i].imshow(frame)
        axes[1, i].set_title(f't-{3-i}')
        axes[1, i].axis('off')
    
    next_f = next_frame.permute(1, 2, 0).numpy()
    axes[1, 4].imshow(next_f)
    axes[1, 4].set_title(f't+1 (a={action.item()})')
    axes[1, 4].axis('off')
    axes[1, 0].set_ylabel('Sequence', fontsize=12)
    
    # Add aspect ratio indicator
    h, w = data['frames'].shape[1:3]
    aspect_text = f"Aspect: {h/w:.3f} (native: {210/160:.3f})"
    fig.text(0.5, 0.02, aspect_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    save_path = "checkpoints/v2/atari/test_aspect_ratio.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to: {save_path}")
    print("  [OK] Check the image to verify paddle and ball are not stretched!")


def compare_old_vs_new():
    """Compare old 64x64 vs new 84x64 visually."""
    print("\n" + "=" * 60)
    print("TEST 5: Old vs New Comparison")
    print("=" * 60)
    
    old_data_path = "checkpoints/v2/atari/backup_v1_64x64/atari_game_data.npz"
    
    if not os.path.exists(old_data_path):
        print("  No old data backup found, skipping comparison")
        return
    
    old_data = np.load(old_data_path)
    old_frames = old_data['frames']
    
    print(f"  Old frame shape: {old_frames.shape[1:3]}")
    
    # Collect a few new frames for comparison
    config = AtariConfig(n_episodes=1, max_steps=50, preserve_aspect=True)
    collector = AtariCollector(config)
    new_data = collector.collect(show_progress=False)
    
    print(f"  New frame shape: {new_data['frames'].shape[1:3]}")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    
    # Old frames (64x64, skewed)
    for i in range(4):
        idx = np.random.randint(len(old_frames))
        axes[0, i].imshow(old_frames[idx])
        axes[0, i].set_title(f'Old ({old_frames.shape[1]}×{old_frames.shape[2]})')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('SKEWED (old)', fontsize=12, color='red')
    
    # New frames (84x64, correct)
    for i in range(4):
        idx = i % len(new_data['frames'])
        axes[1, i].imshow(new_data['frames'][idx])
        axes[1, i].set_title(f'New ({new_data["frames"].shape[1]}×{new_data["frames"].shape[2]})')
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('CORRECT (new)', fontsize=12, color='green')
    
    fig.suptitle('Old (64×64 Skewed) vs New (84×64 Aspect-Preserved)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = "checkpoints/v2/atari/test_old_vs_new.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison to: {save_path}")
    print("  [OK] Compare paddle shape - should look round in new, stretched in old!")


def main():
    print("\n" + "=" * 60)
    print("NEW DATALOADER TEST SUITE")
    print("=" * 60)
    
    # Test 1: Aspect ratio computation
    target_size = test_aspect_ratio()
    
    # Test 2: Small collection
    data = test_small_collection(target_size)
    
    # Test 3: DataLoader speed
    dataset = test_dataloader_speed(data)
    
    # Test 4: Visual verification
    visualize_frames(data, dataset)
    
    # Test 5: Compare old vs new
    compare_old_vs_new()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nCheck these images:")
    print("  - checkpoints/v2/atari/test_aspect_ratio.png")
    print("  - checkpoints/v2/atari/test_old_vs_new.png")
    print("\nIf frames look correct (paddle not horizontally stretched),")
    print("we're ready to collect a full dataset and retrain!")


if __name__ == "__main__":
    main()

