"""
Collect additional random gameplay data and merge with existing.

Usage:
    python v2/utils/collect_more_data.py --game mspacman --episodes 200
    python v2/utils/collect_more_data.py --game breakout --episodes 100 --max-steps 5000
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.data.atari_dataset import AtariConfig, AtariCollector
from v2.collect_and_train import resolve_game_name, get_game_dir


def collect_and_merge(game: str, n_episodes: int, max_steps: int):
    """Collect new data and merge with existing."""
    
    game_id = resolve_game_name(game)
    game_dir = get_game_dir(game_id)
    data_path = f'{game_dir}/game_data.npz'
    
    print("=" * 60)
    print(f"COLLECTING ADDITIONAL DATA: {game_id}")
    print("=" * 60)
    
    # Load existing data if present
    existing_frames = 0
    existing_episodes = 0
    if os.path.exists(data_path):
        existing = dict(np.load(data_path, allow_pickle=True))
        existing_frames = len(existing['frames'])
        existing_episodes = len(existing['episode_starts'])
        print(f"\nExisting data: {existing_frames:,} frames, {existing_episodes} episodes")
    else:
        existing = None
        print("\nNo existing data found, starting fresh.")
    
    # Collect new data
    print(f"\nCollecting {n_episodes} new episodes...")
    config = AtariConfig(
        game=game_id,
        n_episodes=n_episodes,
        max_steps=max_steps,
        preserve_aspect=True,
        target_width=64,
    )
    
    collector = AtariCollector(config)
    new_data = collector.collect(show_progress=True)
    
    print(f"\nCollected: {len(new_data['frames']):,} frames, {len(new_data['episode_starts'])} episodes")
    
    # Merge with existing
    if existing is not None:
        print("\nMerging with existing data...")
        
        # Offset episode starts for new data
        offset = len(existing['frames'])
        new_episode_starts = new_data['episode_starts'] + offset
        
        merged = {
            'frames': np.concatenate([existing['frames'], new_data['frames']]),
            'actions': np.concatenate([existing['actions'], new_data['actions']]),
            'rewards': np.concatenate([existing['rewards'], new_data['rewards']]),
            'dones': np.concatenate([existing['dones'], new_data['dones']]),
            'episode_starts': np.concatenate([existing['episode_starts'], new_episode_starts]),
            'sample_weights': np.concatenate([
                existing.get('sample_weights', np.ones(len(existing['frames']))),
                new_data['sample_weights']
            ]),
            'n_actions': new_data['n_actions'],
            'game': game_id,
        }
    else:
        merged = {
            'frames': new_data['frames'],
            'actions': new_data['actions'],
            'rewards': new_data['rewards'],
            'dones': new_data['dones'],
            'episode_starts': new_data['episode_starts'],
            'sample_weights': new_data['sample_weights'],
            'n_actions': new_data['n_actions'],
            'game': game_id,
        }
    
    # Save merged data
    np.savez_compressed(data_path, **merged)
    
    total_frames = len(merged['frames'])
    total_episodes = len(merged['episode_starts'])
    
    print(f"\n{'=' * 60}")
    print(f"DONE! Total dataset:")
    print(f"  Frames: {total_frames:,}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Saved to: {data_path}")
    print(f"{'=' * 60}")
    
    # Remind about token cache
    tokens_path = f'{game_dir}/tokens.npz'
    if os.path.exists(tokens_path):
        print(f"\n[!] IMPORTANT: Delete stale token cache before training:")
        print(f"   del {tokens_path}")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='Collect additional random gameplay data')
    parser.add_argument('--game', type=str, required=True, help='Game to collect (e.g., mspacman, breakout)')
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to collect (default: 200)')
    parser.add_argument('--max-steps', type=int, default=2000, help='Max steps per episode (default: 2000)')
    args = parser.parse_args()
    
    collect_and_merge(args.game, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
