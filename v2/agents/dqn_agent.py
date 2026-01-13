"""
Latent DQN Agent
================
Deep Q-Network that operates on token/latent representations.

Key features:
- Works directly on VQ-VAE token sequences (no image input)
- Can share token embeddings with world model (optional)
- Supports Double DQN, Dueling architecture
- Compatible with Dyna-style training (real + imagined experience)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .replay_buffer import TransitionBatch


class RunningRewardNormalizer:
    """
    Game-agnostic reward normalization using running statistics.
    Normalizes rewards to roughly [-1, 1] based on observed reward distribution.
    """
    def __init__(self, clip_range: float = 5.0, epsilon: float = 1e-8):
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.var = 1.0
        self.M2 = 0.0  # For Welford's online variance
    
    def update(self, rewards: torch.Tensor):
        """Update running statistics with batch of rewards."""
        rewards_np = rewards.detach().cpu().numpy().flatten()
        for r in rewards_np:
            self.count += 1
            delta = r - self.mean
            self.mean += delta / self.count
            delta2 = r - self.mean
            self.M2 += delta * delta2
        
        if self.count > 1:
            self.var = self.M2 / (self.count - 1)
    
    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running mean/std."""
        std = max(self.var ** 0.5, self.epsilon)
        normalized = (rewards - self.mean) / std
        # Clip to prevent extreme values
        return torch.clamp(normalized, -self.clip_range, self.clip_range)
    
    def state_dict(self):
        return {'count': self.count, 'mean': self.mean, 'var': self.var, 'M2': self.M2}
    
    def load_state_dict(self, state):
        self.count = state['count']
        self.mean = state['mean']
        self.var = state['var']
        self.M2 = state['M2']


@dataclass  
class DQNConfig:
    """Configuration for DQN agent."""
    # Architecture
    d_embed: int = 32           # Token embedding dimension (smaller for CNN)
    hidden_dim: int = 512       # Hidden layer dimension
    n_hidden: int = 2           # Number of hidden layers (for MLP after CNN)
    use_dueling: bool = True    # Dueling architecture
    use_spatial: bool = True    # Use spatial CNN instead of mean pooling
    token_grid_h: int = 21      # Token grid height
    token_grid_w: int = 16      # Token grid width
    
    # Training
    learning_rate: float = 1e-4   # Lower LR for stability
    gamma: float = 0.99           # Discount factor
    tau: float = 0.005            # Soft target update rate (if using soft)
    grad_clip: float = 10.0       # Gradient clipping
    max_q_value: float = 100.0    # Clip Q-values to prevent runaway
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01     # Lower final epsilon
    epsilon_decay_steps: int = 100000  # Longer exploration
    
    # Double DQN & Target Updates
    use_double_dqn: bool = True
    target_update_freq: int = 1000   # Steps between hard target updates
    use_soft_update: bool = False    # Use HARD updates for stability


class LatentDQN(nn.Module):
    """
    DQN that operates on token/latent representations.
    
    Architecture (spatial mode - default):
        tokens (T, H, W) -> embed -> CNN -> flatten -> MLP -> Q-values
        
    Architecture (pooling mode):
        tokens (T, N) -> embed -> mean pool -> MLP -> Q-values
    
    Supports dueling architecture:
        features -> value stream -> V(s)
                 -> advantage stream -> A(s, a)
        Q(s, a) = V(s) + A(s, a) - mean(A)
    """
    
    def __init__(
        self,
        n_vocab: int,
        n_actions: int,
        history_len: int = 4,
        n_tokens: int = 336,
        config: Optional[DQNConfig] = None,
        shared_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_actions = n_actions
        self.history_len = history_len
        self.n_tokens = n_tokens
        self.config = config or DQNConfig()
        
        cfg = self.config
        
        # Token embedding
        if shared_embedding is not None:
            self.token_embed = shared_embedding
            self.d_embed = shared_embedding.embedding_dim
        else:
            self.d_embed = cfg.d_embed
            self.token_embed = nn.Embedding(n_vocab, self.d_embed)
        
        if cfg.use_spatial:
            # Spatial CNN architecture - preserves position information!
            # Input: (B, T*d_embed, H, W) where T=history_len
            in_channels = history_len * self.d_embed
            
            self.conv_net = nn.Sequential(
                # Layer 1: 21x16 -> 10x8
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                # Layer 2: 10x8 -> 5x4
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                # Layer 3: 5x4 -> 3x2
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            
            # Calculate flattened size: 128 * 3 * 2 = 768
            conv_out_h = ((cfg.token_grid_h + 1) // 2 + 1) // 2
            conv_out_w = ((cfg.token_grid_w + 1) // 2 + 1) // 2
            conv_out_h = (conv_out_h + 1) // 2  # Third conv
            conv_out_w = (conv_out_w + 1) // 2
            conv_out_size = 128 * conv_out_h * conv_out_w
            
            self.feature_net = nn.Sequential(
                nn.Linear(conv_out_size, cfg.hidden_dim),
                nn.ReLU(),
            )
        else:
            # Original mean pooling approach (fallback)
            input_dim = history_len * self.d_embed
            layers = []
            current_dim = input_dim
            for i in range(cfg.n_hidden):
                layers.append(nn.Linear(current_dim, cfg.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(cfg.hidden_dim))
                current_dim = cfg.hidden_dim
            self.feature_net = nn.Sequential(*layers)
            self.conv_net = None
        
        if cfg.use_dueling:
            # Dueling architecture
            self.value_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim // 2, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim // 2, n_actions),
            )
        else:
            self.q_head = nn.Linear(cfg.hidden_dim, n_actions)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions.
        
        Args:
            states: (B, T, N) token histories where N = H*W
            
        Returns:
            q_values: (B, n_actions)
        """
        B, T, N = states.shape
        cfg = self.config
        
        # Embed tokens: (B, T, N) -> (B, T, N, d_embed)
        embedded = self.token_embed(states)
        
        if cfg.use_spatial and self.conv_net is not None:
            # Reshape to spatial grid: (B, T, H, W, d_embed)
            H, W = cfg.token_grid_h, cfg.token_grid_w
            embedded = embedded.view(B, T, H, W, self.d_embed)
            
            # Rearrange to (B, T*d_embed, H, W) for conv
            embedded = embedded.permute(0, 1, 4, 2, 3)  # (B, T, d_embed, H, W)
            embedded = embedded.reshape(B, T * self.d_embed, H, W)
            
            # CNN feature extraction
            conv_features = self.conv_net(embedded)  # (B, 128, h, w)
            flattened = conv_features.reshape(B, -1)
        else:
            # Original mean pooling
            pooled = embedded.mean(dim=2)  # (B, T, d_embed)
            flattened = pooled.reshape(B, -1)
        
        # Final feature extraction
        features = self.feature_net(flattened)  # (B, hidden_dim)
        
        if self.config.use_dueling:
            value = self.value_head(features)  # (B, 1)
            advantage = self.advantage_head(features)  # (B, n_actions)
            # Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(features)
        
        return q_values
    
    def select_action(
        self, 
        state: torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: (T, N) or (1, T, N) token history
            epsilon: Exploration probability
            
        Returns:
            action: Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        with torch.no_grad():
            if state.dim() == 2:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class DQNAgent:
    """
    DQN Agent with target network and exploration schedule.
    
    Usage:
        agent = DQNAgent(n_vocab=512, n_actions=9, device='cuda')
        
        # Collect experience
        action = agent.select_action(state)
        
        # Train on batch
        loss = agent.train_step(batch)
        
        # Update target network
        agent.update_target()
    """
    
    def __init__(
        self,
        n_vocab: int,
        n_actions: int,
        history_len: int = 4,
        n_tokens: int = 336,
        config: Optional[DQNConfig] = None,
        device: str = 'cuda',
        shared_embedding: Optional[nn.Embedding] = None,
    ):
        self.config = config or DQNConfig()
        self.device = device
        self.n_actions = n_actions
        
        # Create networks
        self.policy_net = LatentDQN(
            n_vocab, n_actions, history_len, n_tokens, 
            self.config, shared_embedding
        ).to(device)
        
        self.target_net = LatentDQN(
            n_vocab, n_actions, history_len, n_tokens,
            self.config, shared_embedding
        ).to(device)
        
        # Initialize target with policy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Exploration schedule
        self.epsilon = self.config.epsilon_start
        self.epsilon_step = (
            (self.config.epsilon_start - self.config.epsilon_end) 
            / self.config.epsilon_decay_steps
        )
        
        # Counters
        self.train_steps = 0
        self.update_steps = 0
        
        # Game-agnostic reward normalization
        self.reward_normalizer = RunningRewardNormalizer()
    
    def select_action(self, state: torch.Tensor) -> int:
        """Select action with current exploration rate."""
        self.policy_net.eval()
        state = state.to(self.device)
        action = self.policy_net.select_action(state, self.epsilon)
        self.policy_net.train()
        return action
    
    def select_actions_batch(self, states: torch.Tensor) -> np.ndarray:
        """
        Select actions for a batch of states using epsilon-greedy.
        
        This is much more GPU-efficient than calling select_action in a loop!
        
        Args:
            states: (B, T, N) batch of token histories
            
        Returns:
            actions: (B,) numpy array of action indices
        """
        self.policy_net.eval()
        B = states.shape[0]
        states = states.to(self.device)
        
        with torch.no_grad():
            # Single batched forward pass for all states
            q_values = self.policy_net(states)  # (B, n_actions)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
        
        # Apply epsilon-greedy exploration per action
        random_mask = np.random.random(B) < self.epsilon
        random_actions = np.random.randint(0, self.n_actions, size=B)
        
        # Use greedy where mask is False, random where mask is True
        actions = np.where(random_mask, random_actions, greedy_actions)
        
        self.policy_net.train()
        return actions
    
    def train_step(self, batch: TransitionBatch) -> float:
        """
        Train on a batch of transitions.
        
        Args:
            batch: TransitionBatch with states, actions, rewards, next_states, dones
            
        Returns:
            loss: TD loss value
        """
        self.policy_net.train()
        batch = batch.to(self.device)
        cfg = self.config
        
        # Standard Atari DQN reward clipping: sign(reward) -> {-1, 0, +1}
        rewards = torch.sign(batch.rewards)
        
        # Current Q-values: Q(s, a)
        q_values = self.policy_net(batch.states)  # (B, n_actions)
        q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)  # (B,)
        
        # Target Q-values
        with torch.no_grad():
            if cfg.use_double_dqn:
                # Double DQN: use policy net to select action, target net to evaluate
                next_actions = self.policy_net(batch.next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(batch.next_states)
                next_q_values = next_q_values.gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: use target net for both
                next_q_values = self.target_net(batch.next_states).max(dim=1).values
            
            # Clip next Q-values to prevent runaway
            next_q_values = torch.clamp(next_q_values, -cfg.max_q_value, cfg.max_q_value)
            
            # TD target: r + gamma * (1 - done) * max_a' Q(s', a')
            targets = rewards + cfg.gamma * (1 - batch.dones) * next_q_values
        
        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(q_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), cfg.grad_clip)
        self.optimizer.step()
        
        # Update exploration
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon - self.epsilon_step
        )
        
        self.train_steps += 1
        
        # Update target network
        if cfg.use_soft_update:
            self._soft_update()
        elif self.train_steps % cfg.target_update_freq == 0:
            self._hard_update()
        
        return loss.item()
    
    def train_step_prioritized(
        self, 
        batch: TransitionBatch,
        weights: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """
        Train with prioritized experience replay.
        
        Args:
            batch: TransitionBatch
            weights: Importance sampling weights (B,)
            
        Returns:
            loss: Mean loss
            td_errors: TD errors for priority update (B,)
        """
        self.policy_net.train()
        batch = batch.to(self.device)
        weights = weights.to(self.device)
        cfg = self.config
        
        # Standard Atari DQN reward clipping: sign(reward) -> {-1, 0, +1}
        rewards = torch.sign(batch.rewards)
        
        # Current Q-values
        q_values = self.policy_net(batch.states)
        q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            if cfg.use_double_dqn:
                next_actions = self.policy_net(batch.next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(batch.next_states)
                next_q_values = next_q_values.gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(batch.next_states).max(dim=1).values
            
            # Clip next Q-values to prevent runaway
            next_q_values = torch.clamp(next_q_values, -cfg.max_q_value, cfg.max_q_value)
            
            targets = rewards + cfg.gamma * (1 - batch.dones) * next_q_values
        
        # TD errors (for priority update)
        td_errors = (q_values - targets).detach()
        
        # Weighted Huber loss
        element_loss = F.smooth_l1_loss(q_values, targets, reduction='none')
        loss = (weights * element_loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), cfg.grad_clip)
        self.optimizer.step()
        
        # Update exploration and target
        self.epsilon = max(self.config.epsilon_end, self.epsilon - self.epsilon_step)
        self.train_steps += 1
        
        if cfg.use_soft_update:
            self._soft_update()
        elif self.train_steps % cfg.target_update_freq == 0:
            self._hard_update()
        
        return loss.item(), td_errors
    
    def _soft_update(self):
        """Soft update of target network weights."""
        tau = self.config.tau
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1 - tau) * target_param.data
            )
    
    def _hard_update(self):
        """Hard update (copy) target network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'config': self.config,
            'reward_normalizer': self.reward_normalizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        # Use weights_only=False for our own trusted checkpoints (contains reward normalizer with numpy types)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt['epsilon']
        self.train_steps = ckpt['train_steps']
        if 'reward_normalizer' in ckpt:
            self.reward_normalizer.load_state_dict(ckpt['reward_normalizer'])


def test_dqn_agent():
    """Test DQN agent implementation."""
    print("=" * 60)
    print("DQN Agent Tests")
    print("=" * 60)
    
    from .replay_buffer import ReplayBuffer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create agent
    print("\n--- Test 1: Create DQN Agent ---")
    config = DQNConfig(
        d_embed=64,
        hidden_dim=128,
        n_hidden=2,
        use_dueling=True,
        use_double_dqn=True,
    )
    agent = DQNAgent(
        n_vocab=512,
        n_actions=9,
        history_len=4,
        n_tokens=336,
        config=config,
        device=device,
    )
    
    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"  Policy network parameters: {total_params:,}")
    
    # Test action selection
    print("\n--- Test 2: Action Selection ---")
    state = torch.randint(0, 512, (4, 336), device=device)
    action = agent.select_action(state)
    print(f"  Selected action: {action}")
    print(f"  Current epsilon: {agent.epsilon:.3f}")
    
    # Test training
    print("\n--- Test 3: Training Step ---")
    buffer = ReplayBuffer(capacity=1000, history_len=4, n_tokens=336)
    
    # Add transitions
    for i in range(100):
        s = torch.randint(0, 512, (4, 336))
        ns = torch.randint(0, 512, (4, 336))
        buffer.add(s, action=i % 9, reward=float(i % 10), next_state=ns, done=(i % 50 == 0))
    
    # Train
    batch = buffer.sample(32)
    loss = agent.train_step(batch)
    print(f"  Training loss: {loss:.4f}")
    print(f"  Train steps: {agent.train_steps}")
    print(f"  Epsilon after train: {agent.epsilon:.4f}")
    
    # Test Q-value forward
    print("\n--- Test 4: Q-Value Computation ---")
    with torch.no_grad():
        q_values = agent.policy_net(batch.states.to(device))
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Sample Q-values: {q_values[0].tolist()}")
    
    print("\n[OK] DQN Agent tests passed!")


if __name__ == "__main__":
    test_dqn_agent()
