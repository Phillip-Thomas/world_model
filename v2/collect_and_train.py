"""
Full Training Pipeline
======================
1. Collect fresh dataset (84x64 aspect-preserved)
2. Train VQ-VAE v2.3 (with Sobel edge loss)
3. Train World Model (10 layers)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import argparse


def collect_dataset(n_episodes=50, max_steps=2000):
    """Collect fresh dataset with aspect-preserved frames."""
    from v2.data.atari_dataset import AtariConfig, AtariCollector
    
    save_dir = 'checkpoints/v2/atari'
    os.makedirs(save_dir, exist_ok=True)
    
    print('=' * 60)
    print('STEP 1: COLLECTING DATASET (84x64 aspect-preserved)')
    print('=' * 60)
    
    config = AtariConfig(
        game='ALE/Breakout-v5',
        n_episodes=n_episodes,
        max_steps=max_steps,
        preserve_aspect=True,
        target_width=64,
    )
    print(f"Target size: {config.target_size}")
    print(f"Episodes: {config.n_episodes}")
    print(f"Max steps per episode: {config.max_steps}")
    
    collector = AtariCollector(config)
    data = collector.collect(show_progress=True)
    
    # Save
    save_path = f'{save_dir}/atari_game_data.npz'
    np.savez_compressed(
        save_path,
        frames=data['frames'],
        actions=data['actions'],
        rewards=data['rewards'],
        dones=data['dones'],
        episode_starts=data['episode_starts'],
        sample_weights=data['sample_weights'],
        n_actions=data['n_actions'],
    )
    
    print(f"\nSaved {len(data['frames']):,} frames to {save_path}")
    print(f"Frame shape: {data['frames'].shape}")
    print(f"Episodes: {len(data['episode_starts'])}")
    
    return save_path


def train_vqvae(data_path, n_epochs=20, beta=1.0, edge_weight=0.05):
    """Train VQ-VAE v2.3."""
    from v2.train.train_vqvae_hires import train_vqvae_hires
    
    print('\n' + '=' * 60)
    print('STEP 2: TRAINING VQ-VAE v2.3')
    print('=' * 60)
    
    model = train_vqvae_hires(
        n_epochs=n_epochs,
        learning_rate=3e-4,
        beta=beta,
        edge_weight=edge_weight,
        data_path=data_path,
    )
    
    return model


def train_world_model(n_epochs=50):
    """Train World Model (10 layers)."""
    from v2.train.train_wm_hires import train_world_model_hires
    
    print('\n' + '=' * 60)
    print('STEP 3: TRAINING WORLD MODEL (10 layers)')
    print('=' * 60)
    
    train_world_model_hires(
        n_epochs=n_epochs,
        batch_size=48,
        learning_rate=5e-4,
    )


def main():
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument('--episodes', type=int, default=50, help="Episodes to collect")
    parser.add_argument('--max-steps', type=int, default=2000, help="Max steps per episode")
    parser.add_argument('--vqvae-epochs', type=int, default=20, help="VQ-VAE training epochs")
    parser.add_argument('--wm-epochs', type=int, default=50, help="World model training epochs")
    parser.add_argument('--beta', type=float, default=1.0, help="VQ loss weight")
    parser.add_argument('--edge-weight', type=float, default=0.05, help="Sobel edge loss weight (0 to disable)")
    parser.add_argument('--batch-size', type=int, default=64, help="VQ-VAE batch size")
    parser.add_argument('--skip-collect', action='store_true', help="Skip data collection")
    parser.add_argument('--skip-vqvae', action='store_true', help="Skip VQ-VAE training")
    parser.add_argument('--skip-wm', action='store_true', help="Skip world model training")
    args = parser.parse_args()
    
    data_path = 'checkpoints/v2/atari/atari_game_data.npz'
    
    # Step 1: Collect data
    if not args.skip_collect:
        data_path = collect_dataset(args.episodes, args.max_steps)
    else:
        print("\n[Skipping data collection]")
    
    # Step 2: Train VQ-VAE
    if not args.skip_vqvae:
        train_vqvae(data_path, args.vqvae_epochs, args.beta, args.edge_weight)
    else:
        print("\n[Skipping VQ-VAE training]")
    
    # Step 3: Train World Model
    if not args.skip_wm:
        train_world_model(args.wm_epochs)
    else:
        print("\n[Skipping world model training]")
    
    print('\n' + '=' * 60)
    print('TRAINING COMPLETE!')
    print('=' * 60)


if __name__ == "__main__":
    main()

