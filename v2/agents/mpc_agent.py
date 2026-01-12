"""
MPC Agent for Token-Space World Model
======================================
Model Predictive Control using the learned world model for action selection.

Key idea:
1. Encode current observation to tokens
2. Sample candidate action sequences
3. Rollout each sequence in the world model
4. Score by cumulative predicted reward
5. Execute first action of best sequence

Supports:
- Random shooting (basic)
- Cross-Entropy Method (CEM) for improved search
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class MPCConfig:
    """Configuration for MPC agent."""
    horizon: int = 10           # Planning horizon (H steps)
    n_candidates: int = 64      # Number of action sequences to sample
    gamma: float = 0.99         # Discount factor for rewards
    
    # CEM parameters
    use_cem: bool = False       # Use Cross-Entropy Method instead of random shooting
    cem_elite_ratio: float = 0.1  # Top fraction to keep as elite
    cem_iterations: int = 3     # Number of CEM refinement iterations
    cem_alpha: float = 0.25     # Smoothing factor for distribution update
    
    # Rollout parameters
    deterministic_rollout: bool = True  # Use argmax for token prediction (stable)
    temperature: float = 1.0            # Sampling temperature if not deterministic
    top_k: Optional[int] = None         # Top-k sampling if not deterministic
    
    # Early termination
    stop_on_done: bool = True   # Stop rollout if done predicted
    done_threshold: float = 0.5  # P(done) threshold for termination


class MPCAgent:
    """
    Model Predictive Control agent using a learned world model.
    
    Usage:
        agent = MPCAgent(world_model, vqvae, n_actions=9)
        agent.reset(initial_frame)
        
        for _ in range(1000):
            action = agent.select_action()
            next_frame, reward, done, info = env.step(action)
            agent.update(action, next_frame)
    """
    
    def __init__(
        self,
        world_model,
        vqvae,
        n_actions: int,
        config: Optional[MPCConfig] = None,
        device: str = 'cuda',
    ):
        self.world_model = world_model
        self.vqvae = vqvae
        self.n_actions = n_actions
        self.config = config or MPCConfig()
        self.device = device
        
        # Get token dimensions from world model
        self.token_h = world_model.token_h
        self.token_w = world_model.token_w
        self.n_tokens = self.token_h * self.token_w
        self.history_len = world_model.max_history
        
        # Token history buffer (T, N)
        self.token_history = None
        
        # Put models in eval mode
        self.world_model.eval()
        self.vqvae.eval()
    
    def reset(self, initial_frame: torch.Tensor):
        """
        Reset agent with initial frame.
        
        Args:
            initial_frame: (C, H, W) or (1, C, H, W) frame tensor in [-1, 1]
        """
        with torch.no_grad():
            # Ensure correct shape
            if initial_frame.dim() == 3:
                initial_frame = initial_frame.unsqueeze(0)
            initial_frame = initial_frame.to(self.device)
            
            # Encode to tokens
            initial_tokens = self.vqvae.encode(initial_frame)  # (1, H, W)
            initial_tokens = initial_tokens.flatten()  # (N,)
            
            # Initialize history with copies of initial frame
            self.token_history = initial_tokens.unsqueeze(0).repeat(self.history_len, 1)
    
    def update(self, action: int, next_frame: torch.Tensor):
        """
        Update history with new frame after taking action in real environment.
        
        Args:
            action: Action that was taken
            next_frame: (C, H, W) or (1, C, H, W) resulting frame tensor
        """
        with torch.no_grad():
            # Encode next frame
            if next_frame.dim() == 3:
                next_frame = next_frame.unsqueeze(0)
            next_frame = next_frame.to(self.device)
            
            next_tokens = self.vqvae.encode(next_frame)  # (1, H, W)
            next_tokens = next_tokens.flatten()  # (N,)
            
            # Roll history and add new frame
            self.token_history = torch.roll(self.token_history, shifts=-1, dims=0)
            self.token_history[-1] = next_tokens
    
    def select_action(self) -> int:
        """
        Select best action using MPC planning.
        
        Returns:
            Best action to take
        """
        if self.config.use_cem:
            return self._select_action_cem()
        else:
            return self._select_action_random()
    
    def _select_action_random(self) -> int:
        """Random shooting MPC - sample random action sequences and pick best."""
        with torch.no_grad():
            cfg = self.config
            
            # Sample random action sequences: (n_candidates, horizon)
            action_seqs = torch.randint(
                0, self.n_actions, 
                (cfg.n_candidates, cfg.horizon),
                device=self.device
            )
            
            # Score each sequence by rollout
            scores = self._score_action_sequences(action_seqs)
            
            # Return first action of best sequence
            best_idx = scores.argmax().item()
            return action_seqs[best_idx, 0].item()
    
    def _select_action_cem(self) -> int:
        """Cross-Entropy Method MPC - iteratively refine action distribution."""
        with torch.no_grad():
            cfg = self.config
            n_elite = max(1, int(cfg.n_candidates * cfg.cem_elite_ratio))
            
            # Initialize uniform distribution over actions for each timestep
            # Represent as categorical probabilities: (horizon, n_actions)
            action_probs = torch.ones(
                cfg.horizon, self.n_actions, 
                device=self.device
            ) / self.n_actions
            
            for iteration in range(cfg.cem_iterations):
                # Sample action sequences from current distribution
                action_seqs = torch.zeros(
                    cfg.n_candidates, cfg.horizon, 
                    dtype=torch.long, device=self.device
                )
                for t in range(cfg.horizon):
                    action_seqs[:, t] = torch.multinomial(
                        action_probs[t].unsqueeze(0).expand(cfg.n_candidates, -1),
                        num_samples=1
                    ).squeeze(-1)
                
                # Score sequences
                scores = self._score_action_sequences(action_seqs)
                
                # Get elite samples
                elite_indices = scores.topk(n_elite).indices
                elite_seqs = action_seqs[elite_indices]  # (n_elite, horizon)
                
                # Update distribution from elites
                new_probs = torch.zeros_like(action_probs)
                for t in range(cfg.horizon):
                    for a in range(self.n_actions):
                        new_probs[t, a] = (elite_seqs[:, t] == a).float().mean()
                
                # Smooth update (momentum)
                action_probs = cfg.cem_alpha * new_probs + (1 - cfg.cem_alpha) * action_probs
                
                # Ensure probabilities sum to 1 (numerical stability)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            
            # Return most likely action at t=0
            return action_probs[0].argmax().item()
    
    def _score_action_sequences(self, action_seqs: torch.Tensor) -> torch.Tensor:
        """
        Score action sequences by rolling out in the world model.
        
        Args:
            action_seqs: (n_candidates, horizon) action indices
            
        Returns:
            scores: (n_candidates,) cumulative discounted predicted rewards
        """
        cfg = self.config
        n_candidates = action_seqs.shape[0]
        horizon = action_seqs.shape[1]
        
        # Expand history for all candidates: (n_candidates, T, N)
        history = self.token_history.unsqueeze(0).expand(n_candidates, -1, -1).clone()
        
        # Track scores and active mask
        scores = torch.zeros(n_candidates, device=self.device)
        active = torch.ones(n_candidates, dtype=torch.bool, device=self.device)
        discount = 1.0
        
        for t in range(horizon):
            if not active.any():
                break
            
            # Get actions for this step
            actions = action_seqs[:, t]  # (n_candidates,)
            
            # Forward through world model
            next_tokens, reward_pred, done_pred = self.world_model.forward_with_heads(
                history,
                actions,
                deterministic=cfg.deterministic_rollout,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )
            
            # Accumulate discounted rewards (only for active sequences)
            scores[active] += discount * reward_pred[active]
            discount *= cfg.gamma
            
            # Check for termination
            if cfg.stop_on_done:
                terminated = done_pred > cfg.done_threshold
                active = active & ~terminated
            
            # Update history with predicted next tokens
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = next_tokens
        
        return scores
    
    @torch.no_grad()
    def get_rollout_visualization(
        self, 
        action_sequence: List[int],
        decode_frames: bool = True,
    ) -> Tuple[List[torch.Tensor], List[float], List[float]]:
        """
        Rollout a specific action sequence and optionally decode frames.
        
        Useful for debugging and visualization.
        
        Args:
            action_sequence: List of actions to execute
            decode_frames: Whether to decode tokens to frames
            
        Returns:
            frames: List of (C, H, W) tensors if decode_frames, else token tensors
            rewards: List of predicted rewards
            dones: List of predicted P(done)
        """
        history = self.token_history.unsqueeze(0).clone()  # (1, T, N)
        
        frames = []
        rewards = []
        dones = []
        
        for action in action_sequence:
            action_tensor = torch.tensor([action], device=self.device)
            
            next_tokens, reward_pred, done_pred = self.world_model.forward_with_heads(
                history,
                action_tensor,
                deterministic=True,
            )
            
            # Store predictions
            rewards.append(reward_pred.item())
            dones.append(done_pred.item())
            
            # Decode or store tokens
            if decode_frames:
                tokens_2d = next_tokens.reshape(1, self.token_h, self.token_w)
                frame = self.vqvae.decode(tokens_2d)  # (1, C, H, W)
                frames.append(frame.squeeze(0))
            else:
                frames.append(next_tokens.squeeze(0))
            
            # Update history
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = next_tokens
        
        return frames, rewards, dones


def test_mpc_agent():
    """Quick test of MPC agent with mock models."""
    print("=" * 60)
    print("MPC Agent Test")
    print("=" * 60)
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from models.temporal_world_model import TemporalVisualWorldModel
    from models.vqvae_hires import VQVAEHiRes
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create mock models
    print("\nCreating mock models...")
    vqvae = VQVAEHiRes(
        in_channels=3,
        latent_channels=256,
        n_embeddings=512,
    ).to(device)
    
    world_model = TemporalVisualWorldModel(
        n_vocab=512,
        n_actions=9,  # Ms. Pac-Man has 9 actions
        d_model=256,
        n_heads=8,
        n_layers=4,  # Fewer layers for testing
        token_h=21,
        token_w=16,
        max_history=4,
    ).to(device)
    
    # Create agent
    print("\nCreating MPC agent...")
    config = MPCConfig(
        horizon=5,
        n_candidates=32,
        gamma=0.99,
        use_cem=False,
    )
    agent = MPCAgent(world_model, vqvae, n_actions=9, config=config, device=device)
    
    # Create random initial frame
    print("\nTesting agent...")
    initial_frame = torch.randn(1, 3, 84, 64, device=device)
    agent.reset(initial_frame)
    print(f"  Token history shape: {agent.token_history.shape}")
    
    # Test action selection (random shooting)
    action = agent.select_action()
    print(f"  Selected action (random): {action}")
    
    # Test CEM
    agent.config.use_cem = True
    action_cem = agent.select_action()
    print(f"  Selected action (CEM): {action_cem}")
    
    # Test update
    next_frame = torch.randn(1, 3, 84, 64, device=device)
    agent.update(action, next_frame)
    print(f"  Updated history with new frame")
    
    # Test rollout visualization
    frames, rewards, dones = agent.get_rollout_visualization([0, 1, 2, 3, 4])
    print(f"  Rollout visualization: {len(frames)} frames")
    print(f"    Rewards: {rewards}")
    print(f"    Dones: {dones}")
    
    print("\n[OK] MPC Agent working!")


if __name__ == "__main__":
    test_mpc_agent()
