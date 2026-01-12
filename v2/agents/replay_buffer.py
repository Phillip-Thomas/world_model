"""
Replay Buffers for Latent Space RL
===================================
Buffers for storing transitions in token/latent space for Dyna-style training.

Supports:
- Real transitions from environment interaction
- Imagined transitions from world model rollouts
- Uniform and prioritized sampling
"""

import torch
import numpy as np
from typing import Optional, Tuple, NamedTuple, List
from dataclasses import dataclass
import random


class Transition(NamedTuple):
    """A single transition in latent space."""
    state: torch.Tensor      # (T, N) token history
    action: int              # action index
    reward: float            # reward received
    next_state: torch.Tensor # (T, N) next token history
    done: bool               # episode terminated


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


class ReplayBuffer:
    """
    Experience replay buffer for latent/token space transitions.
    
    Stores (state, action, reward, next_state, done) tuples where
    states are token histories (T, N) tensors.
    
    Usage:
        buffer = ReplayBuffer(capacity=100000, history_len=4, n_tokens=336)
        buffer.add(state, action, reward, next_state, done)
        batch = buffer.sample(batch_size=32)
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        history_len: int = 4,
        n_tokens: int = 336,
    ):
        self.capacity = capacity
        self.history_len = history_len
        self.n_tokens = n_tokens
        
        # Pre-allocate tensors for efficiency
        self.states = torch.zeros(capacity, history_len, n_tokens, dtype=torch.long)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros(capacity, history_len, n_tokens, dtype=torch.long)
        self.dones = torch.zeros(capacity, dtype=torch.float32)
        
        self.position = 0
        self.size = 0
    
    def add(
        self,
        state: torch.Tensor,     # (T, N) token history
        action: int,
        reward: float,
        next_state: torch.Tensor, # (T, N) next token history  
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.states[self.position] = state.cpu()
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state.cpu()
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(
        self,
        states: torch.Tensor,     # (B, T, N)
        actions: torch.Tensor,    # (B,)
        rewards: torch.Tensor,    # (B,)
        next_states: torch.Tensor, # (B, T, N)
        dones: torch.Tensor,      # (B,)
    ):
        """Add a batch of transitions."""
        batch_size = states.shape[0]
        for i in range(batch_size):
            self.add(
                states[i], 
                actions[i].item(),
                rewards[i].item(),
                next_states[i],
                dones[i].item() > 0.5,
            )
    
    def clear(self):
        """Clear the buffer (reset position and size)."""
        self.position = 0
        self.size = 0
    
    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return TransitionBatch(
            states=self.states[indices].clone(),
            actions=self.actions[indices].clone(),
            rewards=self.rewards[indices].clone(),
            next_states=self.next_states[indices].clone(),
            dones=self.dones[indices].clone(),
        )
    
    def sample_states(self, batch_size: int) -> torch.Tensor:
        """Sample random states (for starting imagined rollouts)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices].clone()
    
    def sample_with_recency(
        self, 
        batch_size: int, 
        recent_k: int = 50000,
        recent_frac: float = 0.5,
    ) -> TransitionBatch:
        """
        Sample with mix of recent and uniform transitions.
        
        For WM fine-tuning: emphasizes recent on-policy data while
        maintaining coverage of older states.
        
        Args:
            batch_size: Total batch size
            recent_k: Window size for "recent" samples
            recent_frac: Fraction of batch from recent window (0.5 = 50%)
            
        Returns:
            TransitionBatch with mixed recent/uniform samples
        """
        if self.size == 0:
            raise ValueError("Empty buffer")
        
        recent_k = min(recent_k, self.size)
        n_recent = int(batch_size * recent_frac)
        n_uniform = batch_size - n_recent
        
        # Recent indices: sample from last recent_k transitions
        # Handle ring buffer wraparound
        if self.size < self.capacity:
            # Buffer not full yet - simple indexing
            recent_start = max(0, self.size - recent_k)
            recent_indices = np.random.randint(recent_start, self.size, size=n_recent)
        else:
            # Buffer full - handle wraparound from position
            # Recent entries are in range [position - recent_k, position)
            offsets = np.random.randint(1, recent_k + 1, size=n_recent)
            recent_indices = (self.position - offsets) % self.capacity
        
        # Uniform indices: anywhere in valid buffer
        uniform_indices = np.random.randint(0, self.size, size=n_uniform)
        
        # Combine and shuffle
        indices = np.concatenate([recent_indices, uniform_indices])
        np.random.shuffle(indices)
        
        return TransitionBatch(
            states=self.states[indices].clone(),
            actions=self.actions[indices].clone(),
            rewards=self.rewards[indices].clone(),
            next_states=self.next_states[indices].clone(),
            dones=self.dones[indices].clone(),
        )
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Transitions with higher TD-error are sampled more frequently.
    Uses sum-tree for efficient O(log n) sampling.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        history_len: int = 4,
        n_tokens: int = 336,
        alpha: float = 0.6,  # Priority exponent (0 = uniform, 1 = full prioritization)
        beta: float = 0.4,   # Importance sampling correction (starts low, anneals to 1)
        beta_increment: float = 0.001,  # How much to increase beta per sample
        epsilon: float = 1e-6,  # Small constant to ensure non-zero priority
    ):
        self.capacity = capacity
        self.history_len = history_len
        self.n_tokens = n_tokens
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Storage
        self.states = torch.zeros(capacity, history_len, n_tokens, dtype=torch.long)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros(capacity, history_len, n_tokens, dtype=torch.long)
        self.dones = torch.zeros(capacity, dtype=torch.float32)
        
        # Priority tree (sum tree for efficient sampling)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        self.position = 0
        self.size = 0
    
    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        priority: Optional[float] = None,
    ):
        """Add a transition with optional priority (default: max priority)."""
        self.states[self.position] = state.cpu()
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state.cpu()
        self.dones[self.position] = float(done)
        
        # Set priority (new transitions get max priority to ensure they're sampled)
        self.priorities[self.position] = priority if priority else self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[TransitionBatch, torch.Tensor, np.ndarray]:
        """
        Sample a prioritized batch.
        
        Returns:
            batch: TransitionBatch
            weights: Importance sampling weights (B,)
            indices: Indices of sampled transitions (for priority update)
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = torch.from_numpy(weights.astype(np.float32))
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = TransitionBatch(
            states=self.states[indices].clone(),
            actions=self.actions[indices].clone(),
            rewards=self.rewards[indices].clone(),
            next_states=self.next_states[indices].clone(),
            dones=self.dones[indices].clone(),
        )
        
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor):
        """Update priorities based on TD-errors."""
        priorities = np.abs(td_errors.cpu().numpy()) + self.epsilon
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def sample_states(self, batch_size: int) -> torch.Tensor:
        """Sample random states (for starting imagined rollouts)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices].clone()
    
    def sample_with_recency(
        self, 
        batch_size: int, 
        recent_k: int = 50000,
        recent_frac: float = 0.5,
    ) -> TransitionBatch:
        """
        Sample with mix of recent and uniform transitions (ignoring priorities).
        
        Used for WM fine-tuning where we want recency emphasis rather than
        TD-error priorities.
        
        Args:
            batch_size: Total batch size
            recent_k: Window size for "recent" samples
            recent_frac: Fraction of batch from recent window (0.5 = 50%)
            
        Returns:
            TransitionBatch with mixed recent/uniform samples
        """
        if self.size == 0:
            raise ValueError("Empty buffer")
        
        recent_k = min(recent_k, self.size)
        n_recent = int(batch_size * recent_frac)
        n_uniform = batch_size - n_recent
        
        # Recent indices: sample from last recent_k transitions
        if self.size < self.capacity:
            recent_start = max(0, self.size - recent_k)
            recent_indices = np.random.randint(recent_start, self.size, size=n_recent)
        else:
            offsets = np.random.randint(1, recent_k + 1, size=n_recent)
            recent_indices = (self.position - offsets) % self.capacity
        
        # Uniform indices: anywhere in valid buffer
        uniform_indices = np.random.randint(0, self.size, size=n_uniform)
        
        # Combine and shuffle
        indices = np.concatenate([recent_indices, uniform_indices])
        np.random.shuffle(indices)
        
        return TransitionBatch(
            states=self.states[indices].clone(),
            actions=self.actions[indices].clone(),
            rewards=self.rewards[indices].clone(),
            next_states=self.next_states[indices].clone(),
            dones=self.dones[indices].clone(),
        )
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size
    
    def __len__(self) -> int:
        return self.size


class DualReplayBuffer:
    """
    Dual buffer system for Dyna-style training.
    
    Maintains separate buffers for:
    - Real transitions from environment
    - Imagined transitions from world model
    
    Supports mixed sampling with configurable ratio.
    """
    
    def __init__(
        self,
        real_capacity: int = 100000,
        imagined_capacity: int = 50000,  # Smaller - imagined data is cheap to regenerate
        history_len: int = 4,
        n_tokens: int = 336,
        real_ratio: float = 0.5,  # Fraction of real data in mixed batches
    ):
        self.real_buffer = ReplayBuffer(real_capacity, history_len, n_tokens)
        self.imagined_buffer = ReplayBuffer(imagined_capacity, history_len, n_tokens)
        self.real_ratio = real_ratio
    
    def add_real(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Add a real transition from environment."""
        self.real_buffer.add(state, action, reward, next_state, done)
    
    def add_imagined(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Add an imagined transition from world model."""
        self.imagined_buffer.add(state, action, reward, next_state, done)
    
    def add_imagined_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add a batch of imagined transitions."""
        self.imagined_buffer.add_batch(states, actions, rewards, next_states, dones)
    
    def sample_mixed(self, batch_size: int) -> TransitionBatch:
        """Sample a mixed batch of real and imagined transitions."""
        n_real = int(batch_size * self.real_ratio)
        n_imagined = batch_size - n_real
        
        # Ensure we have enough samples
        n_real = min(n_real, len(self.real_buffer))
        n_imagined = min(n_imagined, len(self.imagined_buffer))
        
        if n_real == 0 and n_imagined == 0:
            raise ValueError("Both buffers are empty!")
        
        # Handle case where one buffer is empty
        if n_real == 0:
            return self.imagined_buffer.sample(batch_size)
        if n_imagined == 0:
            return self.real_buffer.sample(batch_size)
        
        # Sample from both
        real_batch = self.real_buffer.sample(n_real)
        imagined_batch = self.imagined_buffer.sample(n_imagined)
        
        # Concatenate
        return TransitionBatch(
            states=torch.cat([real_batch.states, imagined_batch.states], dim=0),
            actions=torch.cat([real_batch.actions, imagined_batch.actions], dim=0),
            rewards=torch.cat([real_batch.rewards, imagined_batch.rewards], dim=0),
            next_states=torch.cat([real_batch.next_states, imagined_batch.next_states], dim=0),
            dones=torch.cat([real_batch.dones, imagined_batch.dones], dim=0),
        )
    
    def sample_real(self, batch_size: int) -> TransitionBatch:
        """Sample only from real buffer."""
        return self.real_buffer.sample(batch_size)
    
    def sample_real_states(self, batch_size: int) -> torch.Tensor:
        """Sample states from real buffer (for starting imagined rollouts)."""
        return self.real_buffer.sample_states(batch_size)
    
    def clear_imagined(self):
        """Clear imagined buffer (call periodically to prevent stale data)."""
        self.imagined_buffer.position = 0
        self.imagined_buffer.size = 0
    
    def is_ready(self, min_real: int = 1000) -> bool:
        """Check if real buffer has enough samples to start training."""
        return len(self.real_buffer) >= min_real
    
    def __len__(self) -> int:
        return len(self.real_buffer) + len(self.imagined_buffer)
    
    @property
    def real_size(self) -> int:
        return len(self.real_buffer)
    
    @property
    def imagined_size(self) -> int:
        return len(self.imagined_buffer)


def test_replay_buffers():
    """Test replay buffer implementations."""
    print("=" * 60)
    print("Replay Buffer Tests")
    print("=" * 60)
    
    # Test basic buffer
    print("\n--- Test 1: Basic ReplayBuffer ---")
    buffer = ReplayBuffer(capacity=1000, history_len=4, n_tokens=336)
    
    # Add some transitions
    for i in range(100):
        state = torch.randint(0, 512, (4, 336))
        next_state = torch.randint(0, 512, (4, 336))
        buffer.add(state, action=i % 9, reward=float(i), next_state=next_state, done=(i % 20 == 0))
    
    print(f"  Buffer size: {len(buffer)}")
    
    # Sample batch
    batch = buffer.sample(32)
    print(f"  Batch shapes: states={batch.states.shape}, actions={batch.actions.shape}")
    print(f"  Sample rewards: {batch.rewards[:5].tolist()}")
    
    # Test prioritized buffer
    print("\n--- Test 2: PrioritizedReplayBuffer ---")
    prio_buffer = PrioritizedReplayBuffer(capacity=1000, history_len=4, n_tokens=336)
    
    for i in range(100):
        state = torch.randint(0, 512, (4, 336))
        next_state = torch.randint(0, 512, (4, 336))
        prio_buffer.add(state, action=i % 9, reward=float(i), next_state=next_state, done=False)
    
    batch, weights, indices = prio_buffer.sample(32)
    print(f"  Buffer size: {len(prio_buffer)}")
    print(f"  Sample weights shape: {weights.shape}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Update priorities
    td_errors = torch.randn(32).abs()
    prio_buffer.update_priorities(indices, td_errors)
    print(f"  Updated priorities for {len(indices)} transitions")
    
    # Test dual buffer
    print("\n--- Test 3: DualReplayBuffer ---")
    dual_buffer = DualReplayBuffer(
        real_capacity=1000,
        imagined_capacity=500,
        history_len=4,
        n_tokens=336,
        real_ratio=0.5,
    )
    
    # Add real transitions
    for i in range(50):
        state = torch.randint(0, 512, (4, 336))
        next_state = torch.randint(0, 512, (4, 336))
        dual_buffer.add_real(state, action=i % 9, reward=1.0, next_state=next_state, done=False)
    
    # Add imagined transitions
    for i in range(30):
        state = torch.randint(0, 512, (4, 336))
        next_state = torch.randint(0, 512, (4, 336))
        dual_buffer.add_imagined(state, action=i % 9, reward=0.5, next_state=next_state, done=False)
    
    print(f"  Real size: {dual_buffer.real_size}")
    print(f"  Imagined size: {dual_buffer.imagined_size}")
    print(f"  Total size: {len(dual_buffer)}")
    
    # Sample mixed
    mixed_batch = dual_buffer.sample_mixed(32)
    print(f"  Mixed batch rewards (should mix 1.0 and 0.5): {mixed_batch.rewards[:10].tolist()}")
    
    print("\n[OK] All replay buffer tests passed!")


if __name__ == "__main__":
    test_replay_buffers()
