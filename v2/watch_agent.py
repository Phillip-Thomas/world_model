"""
Watch Trained Agent Play
========================
Visualize the DQN agent playing Atari games.

Usage:
    python watch_agent.py --game mspacman
    python watch_agent.py --game mspacman --policy checkpoints/v2/mspacman/policy_runs/xxx/policy_best.pt
"""

import os
import sys
import time
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend on Windows
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Error: gymnasium not installed. Run: pip install gymnasium[atari]")
    sys.exit(1)

from models.vqvae_hires import VQVAEHiRes
from agents.dqn_agent import DQNAgent


GAME_CONFIGS = {
    "breakout": {
        "env_name": "ALE/Breakout-v5",
        "base_dir": "checkpoints/v2/atari",
        "n_actions": 4,
    },
    "mspacman": {
        "env_name": "ALE/MsPacman-v5",
        "base_dir": "checkpoints/v2/mspacman",
        "n_actions": 9,
    },
}


def find_latest_policy(base_dir: str) -> str:
    """Find the most recent policy checkpoint."""
    policy_runs = os.path.join(base_dir, "policy_runs")
    if not os.path.exists(policy_runs):
        raise FileNotFoundError(f"No policy runs found in {policy_runs}")
    
    # Get all run directories sorted by name (timestamp)
    runs = sorted(glob.glob(os.path.join(policy_runs, "*")))
    if not runs:
        raise FileNotFoundError(f"No runs in {policy_runs}")
    
    latest_run = runs[-1]
    
    # Look for best policy first, then final
    best = os.path.join(latest_run, "policy_best.pt")
    if os.path.exists(best):
        return best
    
    final = os.path.join(latest_run, "policy_final.pt")
    if os.path.exists(final):
        return final
    
    raise FileNotFoundError(f"No policy checkpoint found in {latest_run}")


def preprocess_frame(frame: np.ndarray, target_size=(84, 64)) -> torch.Tensor:
    """Resize RGB frame for VQ-VAE encoding."""
    from PIL import Image
    
    # Resize RGB frame
    img = Image.fromarray(frame)
    img = img.resize(target_size, Image.BILINEAR)
    
    # To tensor (C, H, W), normalize to [0, 1]
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return tensor


def encode_frame(frame: torch.Tensor, vqvae: VQVAEHiRes, device: str) -> torch.Tensor:
    """Encode a frame to tokens using VQ-VAE."""
    with torch.no_grad():
        # Add batch dim: (C, H, W) -> (1, C, H, W)
        x = frame.unsqueeze(0).to(device)
        indices = vqvae.encode(x)  # Returns (B, H, W) or (B, N)
        return indices.flatten()  # Flatten to (N,)


class AgentVisualizer:
    """Visualize agent playing with real-time updates."""
    
    def __init__(
        self,
        game: str,
        policy_path: str = None,
        device: str = None,
        history_len: int = 4,
    ):
        self.game = game
        self.config = GAME_CONFIGS[game]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_len = history_len
        
        # Load VQ-VAE
        print("Loading VQ-VAE...")
        vqvae_path = os.path.join(self.config["base_dir"], "vqvae_hires.pt")
        checkpoint = torch.load(vqvae_path, map_location=self.device, weights_only=True)
        
        # Infer n_embeddings from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        n_embeddings = state_dict['quantizer.embedding.weight'].shape[0]
        
        self.vqvae = VQVAEHiRes(n_embeddings=n_embeddings)
        self.vqvae.load_state_dict(state_dict)
        self.vqvae = self.vqvae.to(self.device).eval()
        print(f"  Loaded from {vqvae_path} (n_embeddings={n_embeddings})")
        
        # Get token count (input is 84x64, 3 channels)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 84).to(self.device)
            indices = self.vqvae.encode(dummy)
            self.n_tokens = indices.numel()
        print(f"  Token grid: {self.n_tokens} tokens")
        
        # Load agent
        print("Loading DQN agent...")
        if policy_path is None:
            policy_path = find_latest_policy(self.config["base_dir"])
        
        # First load checkpoint to get config (for architecture parameters)
        from agents.dqn_agent import DQNConfig
        # Use weights_only=False for our own trusted checkpoints (contains reward normalizer with numpy types)
        ckpt = torch.load(policy_path, map_location=self.device, weights_only=False)
        saved_config = ckpt.get('config', None)
        
        # Create agent with matching architecture from saved config
        self.agent = DQNAgent(
            n_vocab=n_embeddings,
            n_actions=self.config["n_actions"],
            n_tokens=self.n_tokens,
            history_len=history_len,
            config=saved_config,  # Use saved config to match architecture
            device=self.device,
        )
        self.agent.load(policy_path)
        self.agent.epsilon = 0.0  # Greedy for visualization
        print(f"  Loaded from {policy_path}")
        if saved_config:
            print(f"  Architecture: d_embed={saved_config.d_embed}, hidden_dim={saved_config.hidden_dim}, n_hidden={saved_config.n_hidden}")
        
        # Create environment
        print("Creating environment...")
        self.env = gym.make(
            self.config["env_name"],
            render_mode="rgb_array",
            frameskip=1,
        )
        self.frame_skip = 4
        
        # Token history buffer
        self.token_history = deque(maxlen=history_len)
        
        # Stats
        self.episode_reward = 0
        self.total_steps = 0
        self.episode_count = 0
        
    def reset(self):
        """Reset environment and history."""
        obs, info = self.env.reset()
        self.episode_reward = 0
        self.episode_count += 1
        
        # Encode initial frame
        frame = preprocess_frame(obs)
        tokens = encode_frame(frame, self.vqvae, self.device)
        
        # Fill history with initial frame
        self.token_history.clear()
        for _ in range(self.history_len):
            self.token_history.append(tokens)
        
        return obs
    
    def get_state(self) -> torch.Tensor:
        """Get current state as token history tensor."""
        return torch.stack(list(self.token_history), dim=0)  # (T, N)
    
    def step(self, render=True, debug_q=False):
        """Take one step using the agent's policy."""
        # Get action from agent
        state = self.get_state()
        
        # Get Q-values for diagnostics
        with torch.no_grad():
            state_input = state.unsqueeze(0).to(self.device)
            q_values = self.agent.policy_net(state_input).cpu().numpy()[0]
        
        action = self.agent.select_action(state)
        
        # Debug: print Q-values periodically
        if debug_q and self.total_steps % 50 == 0:
            q_str = " ".join([f"{q:.2f}" for q in q_values])
            adv = q_values - q_values.mean()
            adv_str = " ".join([f"{a:.2f}" for a in adv])
            print(f"  Step {self.total_steps}: Q=[{q_str}] Adv=[{adv_str}] -> Action {action}")
        
        # Store Q-value stats
        self.last_q_values = q_values
        
        # Execute action with frame skip
        total_reward = 0
        obs = None
        done = False
        
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        self.episode_reward += total_reward
        self.total_steps += 1
        
        # Encode new frame
        frame = preprocess_frame(obs)
        tokens = encode_frame(frame, self.vqvae, self.device)
        self.token_history.append(tokens)
        
        return obs, total_reward, done, action
    
    def run_episode(self, max_steps=10000, delay=0.02):
        """Run a single episode with visualization."""
        obs = self.reset()
        
        # Setup matplotlib
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 10))
        img = ax.imshow(obs)
        ax.axis('off')
        title = ax.set_title("", fontsize=14)
        plt.tight_layout()
        
        steps = 0
        done = False
        
        try:
            while not done and steps < max_steps:
                obs, reward, done, action = self.step(debug_q=True)
                steps += 1
                
                # Update display
                img.set_data(obs)
                title.set_text(
                    f"Episode {self.episode_count} | "
                    f"Step {steps} | "
                    f"Reward: {self.episode_reward:.0f} | "
                    f"Action: {action}"
                )
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                if delay > 0:
                    time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        plt.ioff()
        print(f"Episode {self.episode_count} finished: {self.episode_reward:.0f} points in {steps} steps")
        return self.episode_reward
    
    def run_episodes(self, n_episodes=5, max_steps=10000, delay=0.02):
        """Run multiple episodes."""
        rewards = []
        
        for ep in range(n_episodes):
            print(f"\n{'='*50}")
            print(f"Starting Episode {ep + 1}/{n_episodes}")
            print('='*50)
            
            reward = self.run_episode(max_steps=max_steps, delay=delay)
            rewards.append(reward)
        
        print(f"\n{'='*50}")
        print(f"Results over {n_episodes} episodes:")
        print(f"  Average: {np.mean(rewards):.1f}")
        print(f"  Min: {np.min(rewards):.1f}")
        print(f"  Max: {np.max(rewards):.1f}")
        print('='*50)
        
        return rewards
    
    def close(self):
        """Clean up."""
        self.env.close()
        plt.close('all')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Watch trained agent play")
    parser.add_argument("--game", type=str, default="mspacman",
                        choices=list(GAME_CONFIGS.keys()),
                        help="Game to play")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to policy checkpoint (default: latest)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Delay between frames (seconds)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Watch Trained Agent Play")
    print("="*60)
    print(f"Game: {args.game}")
    print(f"Episodes: {args.episodes}")
    print()
    
    viz = AgentVisualizer(
        game=args.game,
        policy_path=args.policy,
        device=args.device,
    )
    
    try:
        viz.run_episodes(n_episodes=args.episodes, delay=args.delay)
    finally:
        viz.close()


if __name__ == "__main__":
    main()
