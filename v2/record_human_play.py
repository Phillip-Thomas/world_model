"""
Record Human Gameplay
=====================
Play Breakout yourself and record frames for the world model.

Controls:
  LEFT ARROW  = Move paddle left
  RIGHT ARROW = Move paddle right  
  SPACE       = Fire (launch ball)
  R           = Start/Stop RECORDING
  S           = SAVE recorded data
  Q           = Quit (saves automatically)

Frames are saved in the same format as the training data.
"""

import os
import sys
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


def save_data(frames, actions, rewards, dones, episode_starts, target_size):
    """Save recorded data to file."""
    if len(actions) == 0:
        print("No frames to save.")
        return
    
    save_path = "checkpoints/v2/atari/human_gameplay.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
            'n_actions': 4,
            'game': 'ALE/Breakout-v5',
            'frame_size': target_size,
        }
    else:
        data = {
            'frames': np.array(frames, dtype=np.uint8),
            'actions': np.array(actions, dtype=np.int32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=bool),
            'episode_starts': np.array(episode_starts, dtype=np.int32),
            'n_actions': 4,
            'game': 'ALE/Breakout-v5',
            'frame_size': target_size,
        }
    
    np.savez_compressed(save_path, **data)
    print(f"Saved! Total frames: {len(data['frames'])}")


def main():
    print("=" * 60)
    print("Human Gameplay Recorder")
    print("=" * 60)
    print("\nControls:")
    print("  LEFT/RIGHT arrows = Move paddle")
    print("  SPACE = Fire (launch ball)")
    print("  R = Start/Stop RECORDING")
    print("  S = SAVE recorded data")
    print("  Q = Quit")
    print("=" * 60)
    
    # Setup
    config = AtariConfig(
        game='ALE/Breakout-v5',
        preserve_aspect=True,
        target_width=64,
    )
    target_size = config.target_size  # (84, 64)
    
    # Create environment with RGB rendering
    # frameskip=4 means each action repeats for 4 frames (standard Atari)
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', frameskip=4)
    
    # Setup pygame for keyboard input
    pygame.init()
    # Atari obs is (210, 160, 3) = H x W x C
    # Scale 3x: display is 480 wide x 630 tall
    SCALE = 3
    DISPLAY_W = 160 * SCALE  # 480
    DISPLAY_H = 210 * SCALE  # 630
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Breakout - Press R to Record")
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
                        pygame.display.set_caption("Breakout - *** RECORDING *** (R to stop)")
                    else:
                        print(f"Recording PAUSED - {len(frames)} frames captured")
                        pygame.display.set_caption("Breakout - PAUSED (R to record, S to save)")
                elif event.key == pygame.K_s:
                    # Save data
                    print("Saving...")
                    save_data(frames, actions, rewards, dones, episode_starts, target_size)
        
        # Check held keys for movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 3  # LEFT
        elif keys[pygame.K_RIGHT]:
            action = 2  # RIGHT
        elif keys[pygame.K_SPACE]:
            action = 1  # FIRE
        
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
        display_frame = cv2.resize(obs, (DISPLAY_W, DISPLAY_H))  # Output: (DISPLAY_H, DISPLAY_W, 3)
        # Pygame surfarray expects (width, height, 3), numpy gives (height, width, 3)
        display_frame = np.transpose(display_frame, (1, 0, 2))  # Now (DISPLAY_W, DISPLAY_H, 3)
        surface = pygame.surfarray.make_surface(display_frame)
        screen.blit(surface, (0, 0))
        
        # Show stats
        font = pygame.font.Font(None, 36)
        
        # Recording indicator
        if recording:
            rec_text = font.render("*** RECORDING ***", True, (255, 0, 0))
            screen.blit(rec_text, (10, 10))
            y_offset = 45
        else:
            rec_text = font.render("PAUSED (R=record, S=save, Q=quit)", True, (255, 255, 0))
            screen.blit(rec_text, (10, 10))
            y_offset = 45
        
        text = font.render(f"Score: {int(episode_reward)}  Recorded: {recorded_frames} frames", True, (255, 255, 255))
        screen.blit(text, (10, y_offset))
        
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
        
        clock.tick(15)  # 15 FPS for playable speed (with frameskip=4, feels like 60Hz)
    
    pygame.quit()
    env.close()
    
    # Auto-save on quit if there's unsaved data
    if len(actions) > 0:
        print(f"\nAuto-saving {len(frames)} frames...")
        save_data(frames, actions, rewards, dones, episode_starts, target_size)
    else:
        print("No frames recorded.")


if __name__ == "__main__":
    main()

