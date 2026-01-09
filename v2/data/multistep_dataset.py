"""
Multi-Step Rollout Dataset (v4)
===============================
Dataset for K-step rollout training with correct episode boundary handling.

v4 FIX: Correct frame/action alignment across episodes.
Uses episode_id to map action indices to frame indices.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiStepDataset(Dataset):
    """
    Dataset that returns K consecutive (history, action, target) tuples.
    
    v4 FIX: Correct frame/action alignment across episodes.
    Uses episode_id to map action indices to frame indices.
    
    Used for multi-step rollout training where we:
    1. Predict step 1 from real history
    2. Feed prediction back as new history
    3. Repeat for K steps, computing loss at each
    """
    
    def __init__(
        self,
        tokens: np.ndarray,      # (N_frames,) or (N_frames, H, W) token indices
        actions: np.ndarray,     # (N_actions,) action indices  
        dones: np.ndarray,       # (N_actions,) episode terminations
        episode_starts: np.ndarray,  # Not used in v4, kept for compatibility
        history_len: int = 4,
        rollout_steps: int = 4,  # K steps to unroll
    ):
        self.tokens = torch.from_numpy(tokens).long()
        self.actions = torch.from_numpy(actions.astype(np.int64)).long()
        self.dones = np.array(dones, dtype=bool)
        self.history_len = history_len
        self.rollout_steps = rollout_steps
        self.token_shape = tokens.shape[1:] if tokens.ndim > 1 else ()
        
        n = len(self.actions)
        n_frames = len(self.tokens)
        n_episodes = int(self.dones.sum()) + 1
        
        # v2.1: Alignment assertion
        expected_frames = n + n_episodes
        if abs(n_frames - expected_frames) > n_episodes:
            print(f"  [WARNING] Frame/action alignment may be off:")
            print(f"    Frames: {n_frames}, Actions: {n}, Episodes: {n_episodes}")
            print(f"    Expected frames ≈ {expected_frames} (actions + episodes)")
            print(f"    Diff: {n_frames - expected_frames}")
        
        # v4 FIX: Compute episode_id for each action index
        self.episode_id = np.zeros(n, dtype=np.int64)
        if n > 1:
            self.episode_id[1:] = np.cumsum(self.dones[:-1].astype(np.int64))
        
        # v4 FIX: Map action index -> frame index
        self.frame_index = np.arange(n, dtype=np.int64) + self.episode_id
        
        # v4 FIX: Episode start (in action-index space) for each action
        episode_start_mask = np.concatenate([[True], self.dones[:-1]])
        episode_starts_action = np.where(episode_start_mask)[0]
        self.episode_start_action = episode_starts_action[self.episode_id]
        
        # Build valid indices
        self.valid_indices = self._build_valid_indices()
        print(f"  [MultiStepDataset v4] Valid rollout starts: {len(self.valid_indices)}")
    
    def _build_valid_indices(self) -> np.ndarray:
        """
        v4 FIX: Find valid indices with correct episode boundary handling.
        """
        n = len(self.actions)
        valid = []
        
        for i in range(n):
            if i + self.rollout_steps > n:
                continue
            
            if self.dones[i:i + self.rollout_steps - 1].any():
                continue
            
            history_start_action = i - self.history_len + 1
            if history_start_action < self.episode_start_action[i]:
                continue
            
            last_action_idx = i + self.rollout_steps - 1
            if self.episode_id[last_action_idx] != self.episode_id[i]:
                continue
            
            fi_last = self.frame_index[last_action_idx]
            if fi_last + 1 >= len(self.tokens):
                continue
            
            valid.append(i)
        
        return np.array(valid, dtype=np.int64)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        """
        v4 FIX: Use frame_index for correct token indexing.
        
        Returns:
            history: (history_len, H*W) initial history tokens
            actions: (rollout_steps,) actions for each step
            targets: (rollout_steps, H*W) target tokens for each step
        """
        i = self.valid_indices[idx]
        fi = self.frame_index[i]
        start_f = fi - self.history_len + 1
        
        # Initial history
        history = self.tokens[start_f:fi + 1]
        if history.ndim > 1:
            history = history.reshape(self.history_len, -1)
        
        # Actions for K steps
        actions = self.actions[i:i + self.rollout_steps]
        
        # Targets
        targets = self.tokens[fi + 1:fi + 1 + self.rollout_steps]
        if targets.ndim > 1:
            targets = targets.reshape(self.rollout_steps, -1)
        
        return history, actions, targets
