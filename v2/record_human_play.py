"""
Record Human Gameplay
=====================
Play Atari games and record frames for the world model.

Usage:
  python record_human_play.py --game mspacman
  python record_human_play.py --game breakout

Controls:
  Breakout:
    LEFT/RIGHT = Move paddle
    SPACE = Fire (launch ball)
  
  Ms. Pac-Man:
    Arrow Keys = Move (UP/DOWN/LEFT/RIGHT)

  Common:
    R = Start/Stop RECORDING
    S = SAVE recorded data
    Q = Quit (saves automatically)

Frames are saved in the same format as the training data.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time

sys.path.insert(0, os.path.dirname(__file__))
from data.atari_dataset import AtariConfig, compute_aspect_preserving_size

try:
    import gymnasium as gym
    import ale_py
    import pygame
    gym.register_envs(ale_py)
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install gymnasium[atari] pygame")
    sys.exit(1)


# =============================================================================
# Game Configuration
# =============================================================================

GAME_CONFIGS = {
    "breakout": {
        "env_name": "ALE/Breakout-v5",
        "base_dir": "checkpoints/v2/atari",
        "n_actions": 4,
        "controls": {
            pygame.K_LEFT: 3,   # LEFT
            pygame.K_RIGHT: 2,  # RIGHT
            pygame.K_SPACE: 1,  # FIRE
        },
        "control_help": "LEFT/RIGHT=Move, SPACE=Fire",
    },
    "mspacman": {
        "env_name": "ALE/MsPacman-v5",
        "base_dir": "checkpoints/v2/mspacman",
        "n_actions": 9,
        "controls": {
            pygame.K_UP: 1,     # UP
            pygame.K_RIGHT: 2,  # RIGHT
            pygame.K_LEFT: 3,   # LEFT
            pygame.K_DOWN: 4,   # DOWN
        },
        "control_help": "Arrow Keys=Move",
    },
}


def save_data(frames, actions, rewards, dones, episode_starts, target_size, game_config):
    """Save recorded data to file."""
    if len(actions) == 0:
        print("No frames to save.")
        return
    
    save_path = os.path.join(game_config["base_dir"], "human_gameplay.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_actions = game_config["n_actions"]
    game_name = game_config["env_name"]
    
    # Check if file exists and merge
    if os.path.exists(save_path):
        print("Merging with existing data...")
        existing = dict(np.load(save_path, allow_pickle=True))
        
        # Offset episode starts
        offset = len(existing['frames'])
        new_ep_starts = [s + offset for s in episode_starts]
        
        data = {
            'frames': np.concatenate([existing['frames'], np.array(frames, dtype=np.uint8)]),
            'actions': np.concatenate([existing['actions'], np.array(actions, dtype=np.int32)]),
            'rewards': np.concatenate([existing['rewards'], np.array(rewards, dtype=np.float32)]),
            'dones': np.concatenate([existing['dones'], np.array(dones, dtype=bool)]),
            'episode_starts': np.concatenate([existing['episode_starts'], np.array(new_ep_starts, dtype=np.int32)]),
            'n_actions': n_actions,
            'game': game_name,
            'frame_size': target_size,
        }
    else:
        data = {
            'frames': np.array(frames, dtype=np.uint8),
            'actions': np.array(actions, dtype=np.int32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=bool),
            'episode_starts': np.array(episode_starts, dtype=np.int32),
            'n_actions': n_actions,
            'game': game_name,
            'frame_size': target_size,
        }
    
    np.savez_compressed(save_path, **data)
    print(f"Saved to {save_path}! Total frames: {len(data['frames'])}")


def main():
    parser = argparse.ArgumentParser(description='Record human gameplay for world model training')
    parser.add_argument('--game', type=str, default='breakout', 
                        help=f'Game to play ({", ".join(GAME_CONFIGS.keys())})')
    args = parser.parse_args()
    
    game = args.game.lower()
    if game not in GAME_CONFIGS:
        print(f"Unknown game '{game}'. Available: {', '.join(GAME_CONFIGS.keys())}")
        sys.exit(1)
    
    game_config = GAME_CONFIGS[game]
    
    print("=" * 60)
    print(f"Human Gameplay Recorder - {game.upper()}")
    print("=" * 60)
    print(f"\nGame: {game_config['env_name']}")
    print(f"Controls: {game_config['control_help']}")
    print("\nRecording Controls:")
    print("  R = Start/Stop RECORDING")
    print("  S = SAVE recorded data")
    print("  Q = Quit (auto-saves)")
    print("=" * 60)
    
    # Setup
    config = AtariConfig(
        game=game_config["env_name"],
        preserve_aspect=True,
        target_width=64,
    )
    target_size = config.target_size  # (84, 64)
    
    # Create environment with RGB rendering
    env = gym.make(game_config["env_name"], render_mode='rgb_array', frameskip=4)
    
    # Setup pygame for keyboard input
    pygame.init()
    SCALE = 3
    DISPLAY_W = 160 * SCALE  # 480
    DISPLAY_H = 210 * SCALE  # 630
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption(f"{game.upper()} - Press R to Record")
    clock = pygame.time.Clock()
    
    # Storage
    frames = []
    actions = []
    rewards = []
    dones = []
    episode_starts = []
    
    # Preprocess function
    def preprocess(frame):
        resized = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        return resized  # (84, 64, 3) RGB
    
    # Start first episode
    obs, info = env.reset()
    
    running = True
    recording = False
    step = 0
    total_reward = 0
    episode_reward = 0
    episode_count = 0
    recorded_frames = 0
    
    controls = game_config["controls"]
    
    print(f"\nPress R to start recording...")
    
    while running:
        # Handle pygame events
        action = 0  # NOOP by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    # Toggle recording
                    recording = not recording
                    if recording:
                        # Start new recording session
                        episode_starts.append(len(frames))
                        frames.append(preprocess(obs))
                        episode_count += 1
                        print(f"RECORDING started - Episode {episode_count}")
                        pygame.display.set_caption(f"{game.upper()} - *** RECORDING *** (R to stop)")
                    else:
                        print(f"Recording PAUSED - {len(frames)} frames captured")
                        pygame.display.set_caption(f"{game.upper()} - PAUSED (R to record, S to save)")
                elif event.key == pygame.K_s:
                    # Save data
                    print("Saving...")
                    save_data(frames, actions, rewards, dones, episode_starts, target_size, game_config)
        
        # Check held keys for movement
        keys = pygame.key.get_pressed()
        for key, act in controls.items():
            if keys[key]:
                action = act
                break  # Use first matching key
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition ONLY if recording
        if recording:
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            frames.append(preprocess(obs))
            recorded_frames += 1
        
        episode_reward += reward
        total_reward += reward
        step += 1
        
        # Display - cv2.resize takes (width, height)
        display_frame = cv2.resize(obs, (DISPLAY_W, DISPLAY_H))
        display_frame = np.transpose(display_frame, (1, 0, 2))
        surface = pygame.surfarray.make_surface(display_frame)
        screen.blit(surface, (0, 0))
        
        # Show stats
        font = pygame.font.Font(None, 36)
        
        # Recording indicator
        if recording:
            rec_text = font.render("*** RECORDING ***", True, (255, 0, 0))
            screen.blit(rec_text, (10, 10))
        else:
            rec_text = font.render("PAUSED (R=record, S=save, Q=quit)", True, (255, 255, 0))
            screen.blit(rec_text, (10, 10))
        
        text = font.render(f"Score: {int(episode_reward)}  Recorded: {recorded_frames} frames", True, (255, 255, 255))
        screen.blit(text, (10, 45))
        
        pygame.display.flip()
        
        # Handle episode end
        if done:
            if recording:
                print(f"  Episode {episode_count} ended: score={int(episode_reward)}")
                episode_count += 1
                episode_starts.append(len(frames))
            episode_reward = 0
            obs, info = env.reset()
            if recording:
                frames.append(preprocess(obs))
                print(f"Recording... Episode {episode_count}")
        
        clock.tick(15)  # 15 FPS for playable speed
    
    pygame.quit()
    env.close()
    
    # Auto-save on quit if there's unsaved data
    if len(actions) > 0:
        print(f"\nAuto-saving {len(frames)} frames...")
        save_data(frames, actions, rewards, dones, episode_starts, target_size, game_config)
    else:
        print("No frames recorded.")


if __name__ == "__main__":
    main()
