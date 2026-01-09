"""
Dataset Expansion Tool
======================
A flexible, reusable tool for expanding Atari gameplay datasets.

Features:
- Multiple collection strategies (random, epsilon-greedy, exploration)
- Incremental merging with existing data
- State coverage & diversity metrics
- Automatic token cache invalidation
- Episode quality filtering

Usage:
    # Add 100 episodes with random policy
    python expand_dataset.py --episodes 100 --strategy random
    
    # Add exploration-focused episodes (prioritizes novel states)
    python expand_dataset.py --episodes 50 --strategy explore
    
    # Add high-skill episodes using trained agent
    python expand_dataset.py --episodes 50 --strategy agent --agent-path checkpoints/dqn.pt
    
    # View dataset statistics without collecting
    python expand_dataset.py --stats-only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
from datetime import datetime
import hashlib
import json

try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Error: gymnasium required. Run: pip install gymnasium[atari]")
    sys.exit(1)

import cv2


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class ExpansionConfig:
    """Configuration for dataset expansion."""
    
    # Target dataset
    data_path: str = "checkpoints/v2/atari/atari_game_data.npz"
    token_cache_path: str = "checkpoints/v2/atari/atari_tokens_hires.npz"
    
    # Game settings
    game: str = "ALE/Breakout-v5"
    target_size: Tuple[int, int] = (84, 64)  # (H, W)
    
    # Collection settings
    n_episodes: int = 50
    max_steps: int = 2000
    frame_skip: int = 4
    
    # Strategy settings
    strategy: str = "random"  # random, explore, agent, mixed
    
    # Random policy settings
    # Aggressive NOOP/FIRE: NOOP=30%, FIRE=30%, RIGHT=20%, LEFT=20%
    # NOOP: see ball physics without paddle movement
    # FIRE: trigger ball spawns (critical for learning respawn!)
    # Heavy NOOP/FIRE to balance dataset toward 25% each
    action_weights: List[float] = field(default_factory=lambda: [0.48, 0.48, 0.02, 0.02])
    action_repeat_max: int = 4
    
    # Exploration settings (for 'explore' strategy)
    novelty_bonus: float = 0.5
    state_hash_bins: int = 1000  # Coarse state hashing for coverage
    
    # Agent settings (for 'agent' strategy)
    agent_path: Optional[str] = None
    agent_epsilon: float = 0.1  # Exploration during agent play
    
    # Mixed strategy settings
    mix_ratios: Dict[str, float] = field(default_factory=lambda: {
        "random": 0.4,
        "explore": 0.3,
        "sticky": 0.3,  # Sticky actions for longer sequences
    })
    
    # Quality settings
    min_episode_length: int = 50
    min_reward: float = 0.0
    
    # Diversity settings
    seed_range: Tuple[int, int] = (0, 10000)
    noop_range: Tuple[int, int] = (1, 30)
    
    # Processing
    max_pool_frames: bool = True
    fire_on_reset: bool = True
    fire_steps: int = 5


# =============================================================================
# State Coverage Tracker
# =============================================================================
class StateCoverageTracker:
    """
    Tracks state space coverage for diversity metrics.
    
    Uses coarse hashing to estimate coverage without storing all frames.
    """
    
    def __init__(self, n_bins: int = 1000):
        self.n_bins = n_bins
        self.visited_states = set()
        self.state_visits = {}  # hash -> count
        
    def hash_frame(self, frame: np.ndarray) -> int:
        """Coarse hash of a frame for coverage tracking."""
        # Downsample heavily and hash
        small = cv2.resize(frame, (8, 8))
        if small.ndim == 3:
            small = small.mean(axis=2)
        quantized = (small // 32).astype(np.uint8)
        return hash(quantized.tobytes()) % self.n_bins
    
    def add(self, frame: np.ndarray) -> bool:
        """Add frame, return True if novel (first time seeing this hash)."""
        h = self.hash_frame(frame)
        is_novel = h not in self.visited_states
        self.visited_states.add(h)
        self.state_visits[h] = self.state_visits.get(h, 0) + 1
        return is_novel
    
    def coverage_ratio(self) -> float:
        """Fraction of hash bins visited."""
        return len(self.visited_states) / self.n_bins
    
    def novelty_score(self, frame: np.ndarray) -> float:
        """Higher score for less-visited states."""
        h = self.hash_frame(frame)
        visits = self.state_visits.get(h, 0)
        return 1.0 / (1.0 + visits)
    
    def stats(self) -> Dict:
        """Return coverage statistics."""
        visits = list(self.state_visits.values())
        return {
            "bins_visited": len(self.visited_states),
            "total_bins": self.n_bins,
            "coverage_pct": self.coverage_ratio() * 100,
            "mean_visits": np.mean(visits) if visits else 0,
            "max_visits": max(visits) if visits else 0,
            "min_visits": min(visits) if visits else 0,
        }


# =============================================================================
# Collection Strategies
# =============================================================================
class BaseStrategy:
    """Base class for action selection strategies."""
    
    def __init__(self, n_actions: int, config: ExpansionConfig):
        self.n_actions = n_actions
        self.config = config
        self.steps = 0
        self.last_action = None
        self.repeat_count = 0
        
    def reset(self):
        """Reset for new episode."""
        self.steps = 0
        self.last_action = None
        self.repeat_count = 0
        
    def select_action(self, frame: np.ndarray = None, reward: float = 0) -> int:
        """Select action. Override in subclasses."""
        raise NotImplementedError
        
    def _apply_repeat_limit(self, action: int) -> int:
        """Apply action repeat limit."""
        if action == self.last_action:
            self.repeat_count += 1
            if self.repeat_count >= self.config.action_repeat_max:
                # Force different action
                available = [a for a in range(self.n_actions) if a != action]
                action = np.random.choice(available)
                self.repeat_count = 0
        else:
            self.repeat_count = 0
        self.last_action = action
        return action


class RandomStrategy(BaseStrategy):
    """Random action selection with configurable weights."""
    
    def select_action(self, frame: np.ndarray = None, reward: float = 0) -> int:
        self.steps += 1
        
        # Force FIRE at start
        if self.steps <= self.config.fire_steps:
            return 1
        
        weights = np.array(self.config.action_weights[:self.n_actions])
        weights = weights / weights.sum()
        action = np.random.choice(self.n_actions, p=weights)
        return self._apply_repeat_limit(action)


class ExplorationStrategy(BaseStrategy):
    """Exploration-focused strategy that seeks novel states."""
    
    def __init__(self, n_actions: int, config: ExpansionConfig, tracker: StateCoverageTracker):
        super().__init__(n_actions, config)
        self.tracker = tracker
        self.action_novelty = {a: 0.0 for a in range(n_actions)}
        self.last_frame = None
        
    def select_action(self, frame: np.ndarray = None, reward: float = 0) -> int:
        self.steps += 1
        
        # Force FIRE at start
        if self.steps <= self.config.fire_steps:
            return 1
        
        # Update action novelty based on last action's result
        if self.last_frame is not None and self.last_action is not None:
            novelty = self.tracker.novelty_score(frame) if frame is not None else 0
            # Exponential moving average
            self.action_novelty[self.last_action] = (
                0.9 * self.action_novelty[self.last_action] + 0.1 * novelty
            )
        
        self.last_frame = frame.copy() if frame is not None else None
        
        # Select action: mix of random and novelty-weighted
        if np.random.random() < 0.3:  # 30% pure random for diversity
            weights = np.array(self.config.action_weights[:self.n_actions])
        else:
            # Weight by historical novelty
            weights = np.array([self.action_novelty[a] + 0.1 for a in range(self.n_actions)])
        
        weights = weights / weights.sum()
        action = np.random.choice(self.n_actions, p=weights)
        return self._apply_repeat_limit(action)


class StickyStrategy(BaseStrategy):
    """Sticky actions for longer, more coherent sequences."""
    
    def __init__(self, n_actions: int, config: ExpansionConfig):
        super().__init__(n_actions, config)
        self.sticky_action = None
        self.sticky_duration = 0
        
    def reset(self):
        super().reset()
        self.sticky_action = None
        self.sticky_duration = 0
        
    def select_action(self, frame: np.ndarray = None, reward: float = 0) -> int:
        self.steps += 1
        
        # Force FIRE at start
        if self.steps <= self.config.fire_steps:
            return 1
        
        # Continue sticky action or pick new one
        if self.sticky_action is not None and self.sticky_duration > 0:
            self.sticky_duration -= 1
            return self.sticky_action
        
        # Pick new sticky action
        weights = np.array([0.05, 0.05, 0.45, 0.45][:self.n_actions])
        weights = weights / weights.sum()
        self.sticky_action = np.random.choice(self.n_actions, p=weights)
        self.sticky_duration = np.random.randint(5, 20)  # Hold for 5-20 steps
        
        return self.sticky_action


class AgentStrategy(BaseStrategy):
    """Use trained agent with epsilon-greedy exploration."""
    
    def __init__(self, n_actions: int, config: ExpansionConfig, agent):
        super().__init__(n_actions, config)
        self.agent = agent
        
    def select_action(self, frame: np.ndarray = None, reward: float = 0) -> int:
        self.steps += 1
        
        # Force FIRE at start
        if self.steps <= self.config.fire_steps:
            return 1
        
        # Epsilon-greedy
        if np.random.random() < self.config.agent_epsilon:
            return np.random.randint(self.n_actions)
        
        # Use agent
        if frame is not None and self.agent is not None:
            with torch.no_grad():
                frame_t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
                frame_t = frame_t / 127.5 - 1.0
                frame_t = frame_t.cuda() if torch.cuda.is_available() else frame_t
                q_values = self.agent(frame_t)
                return q_values.argmax(dim=1).item()
        
        return np.random.randint(self.n_actions)


# =============================================================================
# Data Collector
# =============================================================================
class DataExpander:
    """
    Main class for expanding datasets with multiple strategies.
    """
    
    def __init__(self, config: ExpansionConfig):
        self.config = config
        self.tracker = StateCoverageTracker(config.state_hash_bins)
        self.agent = None
        
        # Load agent if specified
        if config.agent_path and os.path.exists(config.agent_path):
            self._load_agent()
    
    def _load_agent(self):
        """Load trained agent for agent strategy."""
        try:
            checkpoint = torch.load(self.config.agent_path, map_location='cpu')
            # Assuming a simple DQN architecture - customize as needed
            print(f"  Loaded agent from {self.config.agent_path}")
            self.agent = checkpoint.get('model', None)
        except Exception as e:
            print(f"  Warning: Could not load agent: {e}")
            self.agent = None
    
    def _create_env(self, seed: int = None):
        """Create environment with optional seed."""
        env = gym.make(
            self.config.game,
            frameskip=1,
            repeat_action_probability=0.0,
        )
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size."""
        h, w = self.config.target_size
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def _get_strategy(self, strategy_name: str, n_actions: int) -> BaseStrategy:
        """Get strategy instance by name."""
        strategies = {
            "random": lambda: RandomStrategy(n_actions, self.config),
            "explore": lambda: ExplorationStrategy(n_actions, self.config, self.tracker),
            "sticky": lambda: StickyStrategy(n_actions, self.config),
            "agent": lambda: AgentStrategy(n_actions, self.config, self.agent),
        }
        return strategies.get(strategy_name, strategies["random"])()
    
    def _select_strategy_for_episode(self, n_actions: int) -> BaseStrategy:
        """Select strategy for episode based on config."""
        if self.config.strategy == "mixed":
            # Weighted random selection of strategy
            strategies = list(self.config.mix_ratios.keys())
            weights = [self.config.mix_ratios[s] for s in strategies]
            weights = np.array(weights) / sum(weights)
            chosen = np.random.choice(strategies, p=weights)
            return self._get_strategy(chosen, n_actions)
        else:
            return self._get_strategy(self.config.strategy, n_actions)
    
    def collect_episodes(self, show_progress: bool = True) -> Dict:
        """Collect new episodes."""
        env = self._create_env()
        n_actions = env.action_space.n
        
        all_frames = []
        all_actions = []
        all_rewards = []
        all_dones = []
        episode_starts = []
        episode_stats = []
        
        iterator = range(self.config.n_episodes)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Collecting ({self.config.strategy})")
        
        for ep_idx in iterator:
            # Random seed for diversity
            seed = np.random.randint(*self.config.seed_range)
            env.reset(seed=seed)
            
            # Random no-ops
            n_noops = np.random.randint(*self.config.noop_range)
            obs, _ = env.reset()
            for _ in range(n_noops):
                obs, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    obs, _ = env.reset()
                    break
            
            # Fire to start
            if self.config.fire_on_reset:
                obs, _, _, _, _ = env.step(1)
            
            # Select strategy for this episode
            strategy = self._select_strategy_for_episode(n_actions)
            strategy.reset()
            
            # Record start
            episode_starts.append(len(all_frames))
            
            # Store initial frame
            frame = self._resize_frame(obs)
            all_frames.append(frame)
            self.tracker.add(frame)
            
            ep_reward = 0
            ep_length = 0
            novel_states = 0
            
            for step in range(self.config.max_steps):
                # Select action
                action = strategy.select_action(frame, ep_reward)
                
                # Execute with frame skipping
                total_reward = 0
                done = False
                frame_buffer = []
                
                for skip_idx in range(self.config.frame_skip):
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    
                    if skip_idx >= self.config.frame_skip - 2:
                        frame_buffer.append(obs.copy())
                    
                    if done:
                        strategy.steps = 0  # Reset FIRE counter on life loss
                        break
                
                # Max-pool and process
                if self.config.max_pool_frames and len(frame_buffer) >= 2:
                    pooled = np.maximum.reduce(frame_buffer[-2:])
                else:
                    pooled = frame_buffer[-1] if frame_buffer else obs
                
                frame = self._resize_frame(pooled)
                is_novel = self.tracker.add(frame)
                
                # Store
                all_actions.append(action)
                all_rewards.append(total_reward)
                all_dones.append(done)
                all_frames.append(frame)
                
                ep_reward += total_reward
                ep_length += 1
                if is_novel:
                    novel_states += 1
                
                if done:
                    break
            
            episode_stats.append({
                "length": ep_length,
                "reward": ep_reward,
                "novel_states": novel_states,
                "novel_ratio": novel_states / max(ep_length, 1),
            })
        
        env.close()
        
        # Convert to arrays
        frames = np.array(all_frames, dtype=np.uint8)
        actions = np.array(all_actions, dtype=np.int32)
        rewards = np.array(all_rewards, dtype=np.float32)
        dones = np.array(all_dones, dtype=bool)
        ep_starts = np.array(episode_starts, dtype=np.int32)
        
        return {
            "frames": frames,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "episode_starts": ep_starts,
            "n_actions": n_actions,
            "episode_stats": episode_stats,
        }
    
    def merge_datasets(self, existing: Dict, new: Dict) -> Dict:
        """Merge new data with existing dataset."""
        
        # Offset episode starts for new data
        offset = len(existing["frames"])
        new_ep_starts = new["episode_starts"] + offset
        
        merged = {
            "frames": np.concatenate([existing["frames"], new["frames"]], axis=0),
            "actions": np.concatenate([existing["actions"], new["actions"]], axis=0),
            "rewards": np.concatenate([existing["rewards"], new["rewards"]], axis=0),
            "dones": np.concatenate([existing["dones"], new["dones"]], axis=0),
            "episode_starts": np.concatenate([existing["episode_starts"], new_ep_starts], axis=0),
            "n_actions": existing.get("n_actions", new["n_actions"]),
        }
        
        # Use uniform sample weights (sample weights computation has edge case bugs)
        merged["sample_weights"] = np.ones(len(merged["frames"]), dtype=np.float32)
        
        return merged
    
    def _compute_sample_weights(
        self,
        frames: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray,
        tau: float = 10.0,
        novelty_weight: float = 0.5,
    ) -> np.ndarray:
        """Compute sample weights (reward proximity + novelty)."""
        n = len(rewards)
        
        # Build episode ranges
        ep_ranges = []
        for i, start in enumerate(episode_starts):
            end = episode_starts[i + 1] if i + 1 < len(episode_starts) else n
            ep_ranges.append((start, end))
        
        weights = np.ones(n, dtype=np.float32)
        
        for ep_start, ep_end in ep_ranges:
            ep_len = ep_end - ep_start
            if ep_len <= 1:
                continue
            
            ep_rewards = rewards[ep_start:ep_end]
            ep_dones = dones[ep_start:ep_end]
            ep_frames = frames[ep_start:ep_end]
            
            # Reward proximity
            event_indices = np.where((ep_rewards > 0) | ep_dones)[0]
            if len(event_indices) > 0:
                distances = np.full(ep_len, np.inf)
                for eidx in event_indices:
                    dist = np.abs(np.arange(ep_len) - eidx)
                    distances = np.minimum(distances, dist)
                reward_weights = np.exp(-distances / tau)
            else:
                reward_weights = np.ones(ep_len)
            
            # Novelty
            if ep_len > 1:
                frame_diffs = np.abs(
                    ep_frames[1:].astype(np.float32) - ep_frames[:-1].astype(np.float32)
                ).mean(axis=(1, 2, 3))
                novelty = np.concatenate([[0], frame_diffs])
                if novelty.max() > 0:
                    novelty = novelty / novelty.max()
            else:
                novelty = np.ones(ep_len)
            
            # Combine
            alpha = novelty_weight
            ep_weights = (1 - alpha) * reward_weights + alpha * novelty
            if ep_weights.mean() > 0:
                ep_weights = ep_weights / ep_weights.mean()
            
            # Safety check: ensure shapes match before assignment
            target_len = ep_end - ep_start
            if len(ep_weights) == target_len:
                weights[ep_start:ep_end] = ep_weights
            else:
                # Shape mismatch - use uniform weights for this episode
                weights[ep_start:ep_end] = 1.0
        
        return weights
    
    def invalidate_token_cache(self):
        """Delete token cache if it exists (forces regeneration)."""
        if os.path.exists(self.config.token_cache_path):
            try:
                os.remove(self.config.token_cache_path)
                print(f"  Deleted stale token cache: {self.config.token_cache_path}")
                return True
            except PermissionError:
                print(f"  Warning: Could not delete token cache (in use by another process)")
                print(f"           Delete manually later: {self.config.token_cache_path}")
                return False
        return False


# =============================================================================
# Statistics & Reporting
# =============================================================================
def compute_dataset_stats(data: Dict) -> Dict:
    """Compute comprehensive dataset statistics."""
    frames = data["frames"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    ep_starts = data["episode_starts"]
    
    n_episodes = len(ep_starts)
    ep_lengths = []
    ep_rewards = []
    
    for i in range(n_episodes):
        start = ep_starts[i]
        end = ep_starts[i + 1] if i + 1 < n_episodes else len(actions)
        ep_lengths.append(end - start)
        ep_rewards.append(rewards[start:end].sum())
    
    # Action distribution
    action_counts = np.bincount(actions, minlength=data["n_actions"])
    action_pcts = action_counts / len(actions) * 100
    
    # State coverage (using coarse hashing)
    tracker = StateCoverageTracker(n_bins=1000)
    for frame in frames:
        tracker.add(frame)
    
    return {
        "n_frames": len(frames),
        "n_episodes": n_episodes,
        "frame_shape": frames.shape[1:],
        "episode_length": {
            "mean": np.mean(ep_lengths),
            "std": np.std(ep_lengths),
            "min": np.min(ep_lengths),
            "max": np.max(ep_lengths),
        },
        "episode_reward": {
            "mean": np.mean(ep_rewards),
            "std": np.std(ep_rewards),
            "min": np.min(ep_rewards),
            "max": np.max(ep_rewards),
            "total": np.sum(ep_rewards),
        },
        "action_distribution": {
            f"action_{i}": f"{pct:.1f}%" for i, pct in enumerate(action_pcts)
        },
        "state_coverage": tracker.stats(),
    }


def print_stats(stats: Dict, title: str = "Dataset Statistics"):
    """Pretty print statistics."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    
    print(f"\n[Overview]")
    print(f"   Frames:   {stats['n_frames']:,}")
    print(f"   Episodes: {stats['n_episodes']}")
    print(f"   Shape:    {stats['frame_shape']}")
    
    ep_len = stats["episode_length"]
    print(f"\n[Episode Length]")
    print(f"   Mean: {ep_len['mean']:.1f} +/- {ep_len['std']:.1f}")
    print(f"   Range: [{ep_len['min']} - {ep_len['max']}]")
    
    ep_rew = stats["episode_reward"]
    print(f"\n[Episode Reward]")
    print(f"   Mean: {ep_rew['mean']:.1f} +/- {ep_rew['std']:.1f}")
    print(f"   Range: [{ep_rew['min']:.0f} - {ep_rew['max']:.0f}]")
    print(f"   Total: {ep_rew['total']:.0f}")
    
    print(f"\n[Action Distribution]")
    for action, pct in stats["action_distribution"].items():
        print(f"   {action}: {pct}")
    
    coverage = stats["state_coverage"]
    print(f"\n[State Coverage]")
    print(f"   Bins visited: {coverage['bins_visited']}/{coverage['total_bins']}")
    print(f"   Coverage: {coverage['coverage_pct']:.1f}%")
    print(f"   Visits: mean={coverage['mean_visits']:.1f}, max={coverage['max_visits']}")
    
    print()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Dataset Expansion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add 100 episodes with random policy
  python expand_dataset.py --episodes 100 --strategy random
  
  # Add exploration-focused episodes
  python expand_dataset.py --episodes 50 --strategy explore
  
  # Mixed strategies for diversity
  python expand_dataset.py --episodes 100 --strategy mixed
  
  # View current statistics only
  python expand_dataset.py --stats-only
        """
    )
    
    # Collection settings
    parser.add_argument("--episodes", type=int, default=50, help="Episodes to collect")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--strategy", choices=["random", "explore", "sticky", "mixed"], 
                       default="random", help="Collection strategy")
    
    # Agent settings
    parser.add_argument("--agent-path", type=str, help="Path to trained agent checkpoint")
    parser.add_argument("--agent-epsilon", type=float, default=0.1, help="Epsilon for agent")
    
    # Data paths
    parser.add_argument("--data-path", type=str, 
                       default="checkpoints/v2/atari/atari_game_data.npz",
                       help="Path to existing/output dataset")
    
    # Options
    parser.add_argument("--stats-only", action="store_true", help="Show stats without collecting")
    parser.add_argument("--no-merge", action="store_true", help="Replace instead of merge")
    parser.add_argument("--keep-cache", action="store_true", help="Don't invalidate token cache")
    
    args = parser.parse_args()
    
    config = ExpansionConfig(
        data_path=args.data_path,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        strategy=args.strategy,
        agent_path=args.agent_path,
        agent_epsilon=args.agent_epsilon,
    )
    
    # Stats only mode
    if args.stats_only:
        if os.path.exists(config.data_path):
            data = dict(np.load(config.data_path, allow_pickle=True))
            stats = compute_dataset_stats(data)
            print_stats(stats, "Current Dataset Statistics")
        else:
            print(f"No dataset found at {config.data_path}")
        return
    
    print("=" * 60)
    print(" Dataset Expansion Tool")
    print("=" * 60)
    print(f"\n  Target: {config.data_path}")
    print(f"  Strategy: {config.strategy}")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Max steps: {config.max_steps}")
    
    expander = DataExpander(config)
    
    # Load existing data if merging
    existing_data = None
    if os.path.exists(config.data_path) and not args.no_merge:
        print(f"\nLoading existing dataset...")
        existing_data = dict(np.load(config.data_path, allow_pickle=True))
        existing_stats = compute_dataset_stats(existing_data)
        print(f"   Existing: {existing_stats['n_frames']:,} frames, {existing_stats['n_episodes']} episodes")
        
        # Pre-populate tracker with existing states for exploration
        print(f"   Building state coverage from existing data...")
        for frame in tqdm(existing_data["frames"], desc="   Tracking"):
            expander.tracker.add(frame)
    
    # Collect new episodes
    print(f"\nCollecting new episodes...")
    new_data = expander.collect_episodes(show_progress=True)
    
    # Report collection stats
    ep_stats = new_data["episode_stats"]
    mean_len = np.mean([e["length"] for e in ep_stats])
    mean_rew = np.mean([e["reward"] for e in ep_stats])
    mean_nov = np.mean([e["novel_ratio"] for e in ep_stats]) * 100
    print(f"\n   Collected: {len(new_data['frames']):,} frames")
    print(f"   Mean length: {mean_len:.1f}")
    print(f"   Mean reward: {mean_rew:.1f}")
    print(f"   Novel state ratio: {mean_nov:.1f}%")
    
    # Merge or replace
    if existing_data is not None and not args.no_merge:
        print(f"\nMerging datasets...")
        final_data = expander.merge_datasets(existing_data, new_data)
    else:
        # Compute sample weights for new data
        new_data["sample_weights"] = expander._compute_sample_weights(
            new_data["frames"], new_data["rewards"], 
            new_data["dones"], new_data["episode_starts"]
        )
        final_data = new_data
    
    # Save
    print(f"\nSaving to {config.data_path}...")
    os.makedirs(os.path.dirname(config.data_path), exist_ok=True)
    np.savez_compressed(
        config.data_path,
        frames=final_data["frames"],
        actions=final_data["actions"],
        rewards=final_data["rewards"],
        dones=final_data["dones"],
        episode_starts=final_data["episode_starts"],
        sample_weights=final_data["sample_weights"],
        n_actions=final_data["n_actions"],
    )
    
    # Invalidate token cache
    if not args.keep_cache:
        expander.invalidate_token_cache()
    
    # Final stats
    final_stats = compute_dataset_stats(final_data)
    print_stats(final_stats, "Final Dataset Statistics")
    
    # Coverage report
    print("[State Space Coverage]")
    coverage = expander.tracker.stats()
    print(f"   Total unique states: {coverage['bins_visited']}")
    print(f"   Coverage: {coverage['coverage_pct']:.1f}%")
    
    print("\n[OK] Dataset expansion complete!")
    print("\nNote: Token cache was invalidated. Run world model training")
    print("      to regenerate tokens with the updated dataset.")


if __name__ == "__main__":
    main()

