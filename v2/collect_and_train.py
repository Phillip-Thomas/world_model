"""
Full Training Pipeline v2
=========================
Unified command to train a world model for any ALE game.

Usage:
    python v2/collect_and_train.py --game mspacman --episodes 100
    python v2/collect_and_train.py --game breakout --skip-collect --skip-vqvae
    python v2/collect_and_train.py --game asteroids --vqvae-epochs 30 --wm-epochs 100

Pipeline:
1. Collect gameplay data (random policy with smart action selection)
2. Train VQ-VAE (tokenizer)
3. Train World Model (transformer)

Directory structure:
    checkpoints/v2/{game}/
    ├── game_data.npz           # Collected gameplay
    ├── tokens.npz              # Pre-tokenized frames
    ├── vqvae_hires.pt          # Best VQ-VAE checkpoint
    ├── world_model_best.pt     # Best world model
    ├── vqvae_runs/             # VQ-VAE experiment runs
    │   └── 20260111_143022/
    └── wm_runs/                # World model experiment runs
        └── 20260111_150000/
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import argparse
from datetime import datetime

from v2.config.defaults import VQVAEConfig, DEFAULT_VQVAE_CONFIG


# =============================================================================
# Game Name Utilities
# =============================================================================

# Map of short names to full ALE game IDs
GAME_ALIASES = {
    # Common games
    'breakout': 'ALE/Breakout-v5',
    'pong': 'ALE/Pong-v5',
    'spaceinvaders': 'ALE/SpaceInvaders-v5',
    'mspacman': 'ALE/MsPacman-v5',
    'pacman': 'ALE/MsPacman-v5',  # Alias
    'asteroids': 'ALE/Asteroids-v5',
    'qbert': 'ALE/Qbert-v5',
    'seaquest': 'ALE/Seaquest-v5',
    'enduro': 'ALE/Enduro-v5',
    'montezuma': 'ALE/MontezumaRevenge-v5',
    'frostbite': 'ALE/Frostbite-v5',
    'freeway': 'ALE/Freeway-v5',
    'beamrider': 'ALE/BeamRider-v5',
    'boxing': 'ALE/Boxing-v5',
    'bowling': 'ALE/Bowling-v5',
    'assault': 'ALE/Assault-v5',
    'atlantis': 'ALE/Atlantis-v5',
    'bankheist': 'ALE/BankHeist-v5',
    'battlezone': 'ALE/BattleZone-v5',
    'centipede': 'ALE/Centipede-v5',
    'choppercommand': 'ALE/ChopperCommand-v5',
    'crazyclimber': 'ALE/CrazyClimber-v5',
    'demonattack': 'ALE/DemonAttack-v5',
    'phoenix': 'ALE/Phoenix-v5',
    'riverraid': 'ALE/Riverraid-v5',
    'robotank': 'ALE/Robotank-v5',
    'skiing': 'ALE/Skiing-v5',
    'tennis': 'ALE/Tennis-v5',
    'tutankham': 'ALE/Tutankham-v5',
    'venture': 'ALE/Venture-v5',
    'videopinball': 'ALE/VideoPinball-v5',
    'wizard': 'ALE/WizardOfWor-v5',
    'zaxxon': 'ALE/Zaxxon-v5',
}


def resolve_game_name(game_input: str) -> str:
    """
    Resolve a game input to full ALE game ID.
    
    Accepts:
    - Short name: 'breakout', 'mspacman', 'pacman'
    - Full ALE ID: 'ALE/Breakout-v5'
    """
    # Already a full ALE ID
    if game_input.startswith('ALE/'):
        return game_input
    
    # Try alias lookup (case-insensitive)
    key = game_input.lower().replace('-', '').replace('_', '')
    if key in GAME_ALIASES:
        return GAME_ALIASES[key]
    
    # Try to construct ALE ID
    # 'Breakout' -> 'ALE/Breakout-v5'
    name = game_input.replace('-', '').replace('_', '')
    name = name[0].upper() + name[1:]  # Capitalize first letter
    return f'ALE/{name}-v5'


def game_to_slug(game_id: str) -> str:
    """
    Convert ALE game ID to directory-safe slug.
    
    'ALE/MsPacman-v5' -> 'mspacman'
    'ALE/Breakout-v5' -> 'breakout'
    """
    # Extract game name from ALE ID
    match = re.match(r'ALE/([^-]+)-v\d+', game_id)
    if match:
        name = match.group(1)
    else:
        name = game_id.replace('/', '_').replace('-', '_')
    
    return name.lower()


def get_game_dir(game_id: str) -> str:
    """Get the checkpoint directory for a game."""
    slug = game_to_slug(game_id)
    return f'checkpoints/v2/{slug}'


# =============================================================================
# Pipeline Steps
# =============================================================================

def collect_dataset(game_id: str, game_dir: str, n_episodes: int = 50, max_steps: int = 2000):
    """Collect fresh dataset with aspect-preserved frames."""
    from v2.data.atari_dataset import AtariConfig, AtariCollector
    
    os.makedirs(game_dir, exist_ok=True)
    
    print('=' * 60)
    print(f'STEP 1: COLLECTING DATASET')
    print(f'  Game: {game_id}')
    print('=' * 60)
    
    config = AtariConfig(
        game=game_id,
        n_episodes=n_episodes,
        max_steps=max_steps,
        preserve_aspect=True,
        target_width=64,
    )
    print(f"  Target size: {config.target_size}")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Max steps per episode: {config.max_steps}")
    
    collector = AtariCollector(config)
    data = collector.collect(show_progress=True)
    
    # Save
    save_path = f'{game_dir}/game_data.npz'
    np.savez_compressed(
        save_path,
        frames=data['frames'],
        actions=data['actions'],
        rewards=data['rewards'],
        dones=data['dones'],
        episode_starts=data['episode_starts'],
        sample_weights=data['sample_weights'],
        n_actions=data['n_actions'],
        game=game_id,
    )
    
    print(f"\nSaved {len(data['frames']):,} frames to {save_path}")
    print(f"  Frame shape: {data['frames'].shape}")
    print(f"  Episodes: {len(data['episode_starts'])}")
    print(f"  Actions in game: {data['n_actions']}")
    
    return save_path


def train_vqvae(game_dir: str, data_path: str, n_epochs: int = 25, 
                batch_size: int = 128, n_embeddings: int = 32):
    """Train VQ-VAE tokenizer."""
    from v2.train.train_vqvae_hires import train_vqvae_hires
    
    print('\n' + '=' * 60)
    print('STEP 2: TRAINING VQ-VAE')
    print('=' * 60)
    
    model = train_vqvae_hires(
        data_path=data_path,
        base_dir=game_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=3e-4,
        beta=0.1,
        edge_weight=0.05,
        n_embeddings=n_embeddings,
    )
    
    return model


def train_world_model(game_dir: str, n_epochs: int = 50, batch_size: int = 12,
                      learning_rate: float = 5e-4, max_batches: int = 200,
                      rollout_steps: int = 5, rollout_ratio: float = 0.3,
                      teacher_forcing: float = 0.75):
    """Train World Model transformer."""
    from v2.train.train_wm_hires import train_world_model_hires
    
    print('\n' + '=' * 60)
    print('STEP 3: TRAINING WORLD MODEL')
    print('=' * 60)
    
    train_world_model_hires(
        base_dir=game_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_batches=max_batches,
        rollout_steps=rollout_steps,
        rollout_ratio=rollout_ratio,
        teacher_forcing_prob=teacher_forcing,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full training pipeline for any ALE game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v2/collect_and_train.py --game mspacman
  python v2/collect_and_train.py --game breakout --episodes 100 --wm-epochs 100
  python v2/collect_and_train.py --game asteroids --skip-collect
  
Available game aliases:
  breakout, pong, mspacman, pacman, spaceinvaders, asteroids, qbert,
  seaquest, enduro, montezuma, frostbite, freeway, beamrider, ...
  
Or use full ALE ID: ALE/MsPacman-v5, ALE/Breakout-v5, etc.
        """
    )
    
    # Load config defaults
    cfg = VQVAEConfig()
    
    # Game selection (required)
    parser.add_argument('--game', type=str, required=True,
                        help='Game to train on (e.g., "mspacman", "breakout", or "ALE/MsPacman-v5")')
    
    # Data collection (defaults from config)
    parser.add_argument('--episodes', type=int, default=cfg.collection.n_episodes, 
                        help=f'Episodes to collect (default: {cfg.collection.n_episodes})')
    parser.add_argument('--max-steps', type=int, default=cfg.collection.max_steps, 
                        help=f'Max steps per episode (default: {cfg.collection.max_steps})')
    
    # VQ-VAE training (defaults from config)
    parser.add_argument('--vqvae-epochs', type=int, default=cfg.training.n_epochs, 
                        help=f'VQ-VAE training epochs (default: {cfg.training.n_epochs})')
    parser.add_argument('--vqvae-batch-size', type=int, default=cfg.training.batch_size,
                        help=f'VQ-VAE batch size (default: {cfg.training.batch_size})')
    parser.add_argument('--n-embeddings', type=int, default=cfg.model.n_embeddings,
                        help=f'VQ-VAE codebook size (default: {cfg.model.n_embeddings})')
    
    # World model training
    parser.add_argument('--wm-epochs', type=int, default=50, 
                        help='World model training epochs (default: 50)')
    parser.add_argument('--wm-batch-size', type=int, default=12,
                        help='World model batch size (default: 12)')
    parser.add_argument('--wm-lr', type=float, default=5e-4,
                        help='World model learning rate (default: 5e-4)')
    parser.add_argument('--max-batches', type=int, default=200,
                        help='Max batches per epoch (default: 200, 0=all)')
    parser.add_argument('--rollout-steps', type=int, default=5,
                        help='Multi-step rollout length (default: 5)')
    parser.add_argument('--rollout-ratio', type=float, default=0.3,
                        help='Fraction of batches for multi-step (default: 0.3)')
    parser.add_argument('--teacher-forcing', type=float, default=0.75,
                        help='Teacher forcing probability (default: 0.75)')
    
    # Skip steps
    parser.add_argument('--skip-collect', action='store_true', 
                        help='Skip data collection (use existing data)')
    parser.add_argument('--skip-vqvae', action='store_true', 
                        help='Skip VQ-VAE training (use existing checkpoint)')
    parser.add_argument('--skip-wm', action='store_true', 
                        help='Skip world model training')
    
    args = parser.parse_args()
    
    # Resolve game name
    game_id = resolve_game_name(args.game)
    game_dir = get_game_dir(game_id)
    game_slug = game_to_slug(game_id)
    
    print("\n" + "=" * 60)
    print("WORLD MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Game: {game_id}")
    print(f"  Slug: {game_slug}")
    print(f"  Directory: {game_dir}")
    print("=" * 60)
    
    data_path = f'{game_dir}/game_data.npz'
    
    # Step 1: Collect data
    if not args.skip_collect:
        data_path = collect_dataset(
            game_id=game_id,
            game_dir=game_dir,
            n_episodes=args.episodes, 
            max_steps=args.max_steps
        )
    else:
        if not os.path.exists(data_path):
            print(f"\n[ERROR] No data found at {data_path}")
            print("  Run without --skip-collect first!")
            return
        print(f"\n[Skipping data collection, using {data_path}]")
    
    # Step 2: Train VQ-VAE
    if not args.skip_vqvae:
        train_vqvae(
            game_dir=game_dir,
            data_path=data_path,
            n_epochs=args.vqvae_epochs,
            batch_size=args.vqvae_batch_size,
            n_embeddings=args.n_embeddings,
        )
    else:
        vqvae_path = f'{game_dir}/vqvae_hires.pt'
        if not os.path.exists(vqvae_path):
            print(f"\n[ERROR] No VQ-VAE found at {vqvae_path}")
            print("  Run without --skip-vqvae first!")
            return
        print(f"\n[Skipping VQ-VAE training, using {vqvae_path}]")
    
    # Step 3: Train World Model
    if not args.skip_wm:
        train_world_model(
            game_dir=game_dir,
            n_epochs=args.wm_epochs,
            batch_size=args.wm_batch_size,
            learning_rate=args.wm_lr,
            max_batches=args.max_batches,
            rollout_steps=args.rollout_steps,
            rollout_ratio=args.rollout_ratio,
            teacher_forcing=args.teacher_forcing,
        )
    else:
        print("\n[Skipping world model training]")
    
    print('\n' + '=' * 60)
    print('TRAINING COMPLETE!')
    print('=' * 60)
    print(f"  Game: {game_id}")
    print(f"  Directory: {game_dir}")
    print(f"  VQ-VAE: {game_dir}/vqvae_hires.pt")
    print(f"  World Model: {game_dir}/wm_runs/*/world_model_best.pt")
    print('=' * 60)


if __name__ == "__main__":
    main()
