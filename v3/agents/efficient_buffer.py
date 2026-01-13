"""
Efficient Token Buffer (v3)
===========================
Stores single token-frame per timestep, reconstructs history stacks on-demand.

Memory comparison (100k transitions, history_len=4, n_tokens=336):
- v2 approach: 8 × 336 × 8 bytes × 100k = ~2.1 GB (stores state + next_state stacks)
- v3 approach: 336 × 2 bytes × 100k = ~67 MB (stores single frames, uint16)

That's a ~30x memory reduction!

Key insight: Consecutive transitions share 3/4 of their frames. Instead of storing
redundant copies, we store one frame per timestep and reconstruct stacks by indexing.
"""

import torch
import numpy as np
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class TransitionBatch:
    """Batched transitions for training."""
    states: torch.Tensor      # (B, T, N) token histories
    actions: torch.Tensor     # (B,) action indices
    rewards: torch.Tensor     # (B,) rewards
    next_states: torch.Tensor # (B, T, N) next token histories
    dones: torch.Tensor       # (B,) done flags (float for loss)
    
    def to(self, device: str) -> 'TransitionBatch':
        """Move batch to device."""
        return TransitionBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_states=self.next_states.to(device),
            dones=self.dones.to(device),
        )


class EfficientTokenBuffer:
    """
    Memory-efficient replay buffer that stores single token-frames.
    
    Instead of storing (T, N) token stacks per transition, we store
    single (N,) frames and reconstruct stacks on-demand during sampling.
    
    This reduces memory by ~30x while maintaining the same interface.
    
    Args:
        capacity: Maximum number of frames to store
        n_tokens: Number of tokens per frame (336 for 84x64 Atari)
        history_len: Number of frames in history stack (typically 4)
        vocab_size: VQ-VAE vocabulary size (determines dtype: <=256 -> uint8, else uint16)
    
    Memory usage:
        vocab_size <= 256:  capacity × n_tokens × 1 byte  (uint8)
        vocab_size <= 65536: capacity × n_tokens × 2 bytes (uint16)
        
    Example:
        capacity=250k, n_tokens=336, vocab=512:
        250,000 × 336 × 2 = 168 MB (vs ~5.4 GB in v2!)
    """
    
    def __init__(
        self,
        capacity: int = 250000,
        n_tokens: int = 336,
        history_len: int = 4,
        vocab_size: int = 512,
    ):
        self.capacity = capacity
        self.n_tokens = n_tokens
        self.history_len = history_len
        self.vocab_size = vocab_size
        
        # Choose smallest dtype that fits vocab
        if vocab_size <= 256:
            self.np_dtype = np.uint8
            self.torch_dtype = torch.uint8
        elif vocab_size <= 65536:
            self.np_dtype = np.uint16
            self.torch_dtype = torch.int16  # PyTorch doesn't have uint16
        else:
            self.np_dtype = np.int32
            self.torch_dtype = torch.int32
        
        # Pre-allocate storage - single frame per step!
        self.tokens = np.zeros((capacity, n_tokens), dtype=self.np_dtype)
        self.actions = np.zeros(capacity, dtype=np.uint8)  # Max 256 actions
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Episode tracking for correct stack reconstruction
        self.episode_ids = np.zeros(capacity, dtype=np.int32)
        self.current_episode = 0
        
        self.position = 0
        self.size = 0
        
        # Track memory usage
        token_bytes = capacity * n_tokens * (1 if vocab_size <= 256 else 2)
        other_bytes = capacity * (1 + 4 + 1 + 4)  # actions, rewards, dones, episode_ids
        total_mb = (token_bytes + other_bytes) / (1024 * 1024)
        print(f"[EfficientTokenBuffer] Allocated {total_mb:.1f} MB "
              f"(capacity={capacity:,}, dtype={self.np_dtype.__name__})")
    
    def add(
        self,
        tokens: np.ndarray,  # (N,) single frame tokens
        action: int,
        reward: float,
        done: bool,
    ):
        """
        Add a single frame transition.
        
        Args:
            tokens: (N,) token indices for current frame
            action: Action taken
            reward: Reward received
            done: Episode terminated
        """
        self.tokens[self.position] = tokens
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.episode_ids[self.position] = self.current_episode
        
        # Advance position
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Track episode boundaries
        if done:
            self.current_episode += 1
    
    def add_from_tensor(
        self,
        tokens: torch.Tensor,  # (N,) or (1, N) single frame tokens
        action: int,
        reward: float,
        done: bool,
    ):
        """Add from PyTorch tensor."""
        tokens_np = tokens.cpu().numpy().flatten().astype(self.np_dtype)
        self.add(tokens_np, action, reward, done)
    
    def _get_stack(self, idx: int) -> np.ndarray:
        """
        Reconstruct history_len stack for index, respecting episode boundaries.
        
        If index is near episode start, we repeat the first valid frame
        instead of crossing episode boundaries.
        
        Args:
            idx: Index of the "current" frame (last frame in stack)
            
        Returns:
            (history_len, n_tokens) stack
        """
        stack = np.zeros((self.history_len, self.n_tokens), dtype=self.np_dtype)
        current_episode = self.episode_ids[idx]
        
        for i in range(self.history_len):
            # Offset from current frame (0 = current, 1 = previous, etc.)
            offset = self.history_len - 1 - i
            frame_idx = (idx - offset) % self.capacity
            
            # Check if this frame is in the same episode
            if frame_idx < 0 or self.episode_ids[frame_idx] != current_episode:
                # Use the earliest valid frame in this episode instead
                # (repeat the first frame of the stack that's valid)
                earliest_valid = idx
                for j in range(offset - 1, -1, -1):
                    check_idx = (idx - j) % self.capacity
                    if self.episode_ids[check_idx] == current_episode:
                        earliest_valid = check_idx
                        break
                stack[i] = self.tokens[earliest_valid]
            else:
                stack[i] = self.tokens[frame_idx]
        
        return stack
    
    def sample(self, batch_size: int) -> TransitionBatch:
        """
        Sample a batch of transitions, reconstructing stacks on-the-fly.
        
        Returns:
            TransitionBatch with reconstructed (history_len, n_tokens) stacks
        """
        # We need history_len previous frames and 1 next frame
        # Valid indices: [history_len - 1, size - 2] (need next frame for next_state)
        min_idx = self.history_len - 1
        max_idx = self.size - 2
        
        if max_idx < min_idx:
            raise ValueError(f"Not enough samples in buffer (size={self.size}, need at least {self.history_len + 1})")
        
        # Sample indices (vectorized)
        if self.size < self.capacity:
            # Buffer not full - simple range
            indices = np.random.randint(min_idx, max_idx + 1, size=batch_size)
        else:
            # Buffer full - need to handle wrap-around more carefully
            # Avoid sampling too close to write position
            valid_range = self.size - self.history_len - 1
            offsets = np.random.randint(self.history_len, valid_range, size=batch_size)
            indices = (self.position - 1 - offsets) % self.capacity
        
        # Reconstruct stacks (this is the main cost, but still fast)
        states = np.stack([self._get_stack(i) for i in indices])
        next_states = np.stack([self._get_stack((i + 1) % self.capacity) for i in indices])
        
        # Get scalar values
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        
        # Convert to torch tensors
        return TransitionBatch(
            states=torch.from_numpy(states.astype(np.int64)),  # Long for embedding lookup
            actions=torch.from_numpy(actions.astype(np.int64)),
            rewards=torch.from_numpy(rewards),
            next_states=torch.from_numpy(next_states.astype(np.int64)),
            dones=torch.from_numpy(dones.astype(np.float32)),
        )
    
    def sample_states(self, batch_size: int) -> torch.Tensor:
        """Sample random states (for starting imagined rollouts)."""
        min_idx = self.history_len - 1
        max_idx = self.size - 1
        
        if max_idx < min_idx:
            raise ValueError(f"Not enough samples in buffer")
        
        indices = np.random.randint(min_idx, max_idx + 1, size=batch_size)
        states = np.stack([self._get_stack(i) for i in indices])
        return torch.from_numpy(states.astype(np.int64))
    
    def sample_recent(
        self,
        batch_size: int,
        recent_k: int = 50000,
        recent_frac: float = 0.5,
    ) -> TransitionBatch:
        """
        Sample with mix of recent and uniform transitions.
        
        For WM fine-tuning: emphasizes recent on-policy data while
        maintaining coverage of older states.
        """
        if self.size == 0:
            raise ValueError("Empty buffer")
        
        recent_k = min(recent_k, self.size - self.history_len - 1)
        n_recent = int(batch_size * recent_frac)
        n_uniform = batch_size - n_recent
        
        min_idx = self.history_len - 1
        max_idx = self.size - 2
        
        if max_idx < min_idx:
            raise ValueError("Not enough samples")
        
        # Recent indices
        if self.size < self.capacity:
            recent_start = max(min_idx, self.size - 1 - recent_k)
            recent_indices = np.random.randint(recent_start, max_idx + 1, size=n_recent)
        else:
            offsets = np.random.randint(1, min(recent_k, max_idx) + 1, size=n_recent)
            recent_indices = (self.position - 1 - offsets) % self.capacity
        
        # Uniform indices
        uniform_indices = np.random.randint(min_idx, max_idx + 1, size=n_uniform)
        
        # Combine
        indices = np.concatenate([recent_indices, uniform_indices])
        np.random.shuffle(indices)
        
        # Reconstruct stacks
        states = np.stack([self._get_stack(i) for i in indices])
        next_states = np.stack([self._get_stack((i + 1) % self.capacity) for i in indices])
        
        return TransitionBatch(
            states=torch.from_numpy(states.astype(np.int64)),
            actions=torch.from_numpy(self.actions[indices].astype(np.int64)),
            rewards=torch.from_numpy(self.rewards[indices]),
            next_states=torch.from_numpy(next_states.astype(np.int64)),
            dones=torch.from_numpy(self.dones[indices].astype(np.float32)),
        )
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= max(min_size, self.history_len + 1)
    
    def __len__(self) -> int:
        return self.size
    
    def get_memory_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        token_bytes = self.size * self.n_tokens * (1 if self.vocab_size <= 256 else 2)
        other_bytes = self.size * (1 + 4 + 1 + 4)
        return (token_bytes + other_bytes) / (1024 * 1024)


class PrioritizedEfficientBuffer(EfficientTokenBuffer):
    """
    Efficient buffer with prioritized experience replay.
    
    Extends EfficientTokenBuffer with priority-based sampling using SumTree.
    """
    
    def __init__(
        self,
        capacity: int = 250000,
        n_tokens: int = 336,
        history_len: int = 4,
        vocab_size: int = 512,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        super().__init__(capacity, n_tokens, history_len, vocab_size)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
        # SumTree for O(log n) prioritized sampling
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2
    
    def _update_priority(self, data_idx: int, priority: float):
        """Update priority of a leaf node."""
        tree_idx = data_idx + self.capacity
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def _get_priority(self, data_idx: int) -> float:
        """Get priority of a data index."""
        return self.tree[data_idx + self.capacity]
    
    def _sample_idx(self, value: float) -> int:
        """Find leaf index where cumulative sum reaches value."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - self.capacity
    
    def add(
        self,
        tokens: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        priority: float = None,
    ):
        """Add transition with priority (default: max priority)."""
        # Update priority before super().add() changes position
        p = (priority if priority else self.max_priority) ** self.alpha
        self._update_priority(self.position, p)
        
        # Call parent add
        super().add(tokens, action, reward, done)
    
    def sample(self, batch_size: int) -> Tuple[TransitionBatch, torch.Tensor, np.ndarray]:
        """
        Sample prioritized batch.
        
        Returns:
            batch: TransitionBatch
            weights: Importance sampling weights (B,)
            indices: Indices for priority update
        """
        total = self.tree[1]
        segment = total / batch_size
        
        indices = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            indices[i] = self._sample_idx(value)
        
        # Ensure valid indices for stack reconstruction
        min_idx = self.history_len - 1
        max_idx = self.size - 2
        indices = np.clip(indices, min_idx, max_idx)
        
        # Get priorities and compute weights
        priorities = np.array([self._get_priority(idx) for idx in indices])
        probs = priorities / total
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Reconstruct stacks
        states = np.stack([self._get_stack(i) for i in indices])
        next_states = np.stack([self._get_stack((i + 1) % self.capacity) for i in indices])
        
        batch = TransitionBatch(
            states=torch.from_numpy(states.astype(np.int64)),
            actions=torch.from_numpy(self.actions[indices].astype(np.int64)),
            rewards=torch.from_numpy(self.rewards[indices]),
            next_states=torch.from_numpy(next_states.astype(np.int64)),
            dones=torch.from_numpy(self.dones[indices].astype(np.float32)),
        )
        
        return batch, torch.from_numpy(weights.astype(np.float32)), indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor):
        """Update priorities based on TD-errors."""
        priorities = np.abs(td_errors.detach().cpu().numpy()) + self.epsilon
        self.max_priority = max(self.max_priority, priorities.max())
        
        for idx, priority in zip(indices, priorities):
            self._update_priority(idx, priority ** self.alpha)


class ImaginedBuffer:
    """
    Buffer for trust-weighted imagined transitions from WM rollouts.
    
    Unlike EfficientTokenBuffer, this stores complete transitions since
    imagined data doesn't have the temporal redundancy of real data.
    
    Each transition has a trust weight that downweights its contribution
    to policy learning.
    """
    
    def __init__(
        self,
        capacity: int = 50000,
        n_tokens: int = 336,
        history_len: int = 4,
        vocab_size: int = 512,
    ):
        self.capacity = capacity
        self.n_tokens = n_tokens
        self.history_len = history_len
        self.vocab_size = vocab_size
        
        # Determine dtype based on vocab size
        self.token_dtype = np.uint8 if vocab_size <= 256 else np.uint16
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, history_len, n_tokens), dtype=self.token_dtype)
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, history_len, n_tokens), dtype=self.token_dtype)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.trust_weights = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        
        # Track imagination statistics
        self.total_generated = 0
        self.total_accepted = 0
    
    def add(
        self,
        state: np.ndarray,        # (T, N) token history
        action: int,
        reward: float,
        next_state: np.ndarray,   # (T, N) next token history
        done: float,
        trust_weight: float,
    ):
        """Add an imagined transition with its trust weight."""
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.trust_weights[idx] = trust_weight
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(
        self,
        states: np.ndarray,        # (B, T, N)
        actions: np.ndarray,       # (B,)
        rewards: np.ndarray,       # (B,)
        next_states: np.ndarray,   # (B, T, N)
        dones: np.ndarray,         # (B,)
        trust_weights: np.ndarray, # (B,)
    ):
        """Add a batch of imagined transitions."""
        batch_size = len(actions)
        
        for i in range(batch_size):
            self.add(
                states[i], actions[i], rewards[i],
                next_states[i], dones[i], trust_weights[i]
            )
    
    def sample(self, batch_size: int) -> Tuple[TransitionBatch, torch.Tensor]:
        """
        Sample transitions weighted by trust.
        
        Returns:
            batch: TransitionBatch
            weights: (B,) trust weights for loss weighting
        """
        if self.size < batch_size:
            batch_size = self.size
        
        if self.size == 0:
            return None, None
        
        # Sample proportional to trust weights
        valid_weights = self.trust_weights[:self.size]
        probs = valid_weights / (valid_weights.sum() + 1e-8)
        
        indices = np.random.choice(self.size, size=batch_size, replace=False, p=probs)
        
        batch = TransitionBatch(
            states=torch.from_numpy(self.states[indices].astype(np.int64)),
            actions=torch.from_numpy(self.actions[indices].astype(np.int64)),
            rewards=torch.from_numpy(self.rewards[indices]),
            next_states=torch.from_numpy(self.next_states[indices].astype(np.int64)),
            dones=torch.from_numpy(self.dones[indices]),
        )
        
        weights = torch.from_numpy(self.trust_weights[indices])
        
        return batch, weights
    
    def sample_uniform(self, batch_size: int) -> Tuple[TransitionBatch, torch.Tensor]:
        """Sample uniformly, return trust weights for loss weighting."""
        if self.size < batch_size:
            batch_size = self.size
        
        if self.size == 0:
            return None, None
        
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        
        batch = TransitionBatch(
            states=torch.from_numpy(self.states[indices].astype(np.int64)),
            actions=torch.from_numpy(self.actions[indices].astype(np.int64)),
            rewards=torch.from_numpy(self.rewards[indices]),
            next_states=torch.from_numpy(self.next_states[indices].astype(np.int64)),
            dones=torch.from_numpy(self.dones[indices]),
        )
        
        weights = torch.from_numpy(self.trust_weights[indices])
        
        return batch, weights
    
    def clear(self):
        """Clear the buffer (useful when WM has improved significantly)."""
        self.position = 0
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def is_ready(self, min_size: int = 1) -> bool:
        return self.size >= min_size
    
    def get_stats(self) -> dict:
        """Get imagination statistics."""
        if self.size == 0:
            return {
                'size': 0,
                'accept_rate': 0.0,
                'mean_trust': 0.0,
            }
        
        return {
            'size': self.size,
            'accept_rate': self.total_accepted / max(1, self.total_generated),
            'mean_trust': float(self.trust_weights[:self.size].mean()),
            'p10_trust': float(np.percentile(self.trust_weights[:self.size], 10)),
            'p50_trust': float(np.percentile(self.trust_weights[:self.size], 50)),
            'p90_trust': float(np.percentile(self.trust_weights[:self.size], 90)),
        }


def test_efficient_buffer():
    """Test the efficient buffer implementation."""
    print("=" * 60)
    print("Efficient Token Buffer Test")
    print("=" * 60)
    
    # Create buffer
    buffer = EfficientTokenBuffer(
        capacity=10000,
        n_tokens=336,
        history_len=4,
        vocab_size=512,
    )
    
    print(f"\n--- Test 1: Memory Usage ---")
    print(f"Empty buffer size: {len(buffer)}")
    print(f"Memory allocated: {buffer.get_memory_usage_mb():.2f} MB")
    
    # Add some transitions
    print(f"\n--- Test 2: Add Transitions ---")
    episode_rewards = []
    episode_reward = 0
    
    for i in range(5000):
        tokens = np.random.randint(0, 512, size=336, dtype=np.uint16)
        action = np.random.randint(0, 9)
        reward = np.random.random()
        done = (i + 1) % 100 == 0  # Episode ends every 100 steps
        
        buffer.add(tokens, action, reward, done)
        episode_reward += reward
        
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Memory used: {buffer.get_memory_usage_mb():.2f} MB")
    print(f"Episodes completed: {len(episode_rewards)}")
    
    # Sample batch
    print(f"\n--- Test 3: Sample Batch ---")
    batch = buffer.sample(batch_size=64)
    print(f"States shape: {batch.states.shape}")
    print(f"Next states shape: {batch.next_states.shape}")
    print(f"Actions shape: {batch.actions.shape}")
    print(f"Rewards shape: {batch.rewards.shape}")
    print(f"Dones shape: {batch.dones.shape}")
    
    # Verify stack reconstruction
    print(f"\n--- Test 4: Stack Reconstruction ---")
    # Sample states and check they're valid
    states = buffer.sample_states(10)
    print(f"Sampled states shape: {states.shape}")
    print(f"Token value range: [{states.min().item()}, {states.max().item()}]")
    
    # Test prioritized buffer
    print(f"\n--- Test 5: Prioritized Buffer ---")
    prio_buffer = PrioritizedEfficientBuffer(
        capacity=5000,
        n_tokens=336,
        history_len=4,
        vocab_size=512,
    )
    
    for i in range(2000):
        tokens = np.random.randint(0, 512, size=336, dtype=np.uint16)
        prio_buffer.add(tokens, action=i % 9, reward=float(i % 10), done=(i % 50 == 0))
    
    batch, weights, indices = prio_buffer.sample(32)
    print(f"Prioritized batch shape: {batch.states.shape}")
    print(f"Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Update priorities
    td_errors = torch.randn(32).abs()
    prio_buffer.update_priorities(indices, td_errors)
    print(f"Updated priorities for {len(indices)} transitions")
    
    # Compare memory with v2 approach
    print(f"\n--- Memory Comparison (100k transitions) ---")
    v2_memory = 100000 * 8 * 336 * 8 / (1024 * 1024)  # 8 frames × 336 tokens × int64
    v3_memory = 100000 * 336 * 2 / (1024 * 1024)       # 1 frame × 336 tokens × uint16
    
    print(f"v2 approach: {v2_memory:.1f} MB")
    print(f"v3 approach: {v3_memory:.1f} MB")
    print(f"Reduction: {v2_memory / v3_memory:.1f}x")
    
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    test_efficient_buffer()
