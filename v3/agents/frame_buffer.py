"""
Frame Buffer for Continuous VQ-VAE Training (v3)
=================================================
Stores raw frames (not tokens) to enable VQ-VAE fine-tuning.

Key components:
1. FrameBuffer - stores raw frames for on-demand encoding
2. CodebookRehearsalBuffer - stores frames indexed by code for balanced sampling

This enables the continuous training loop where VQ-VAE can be updated
without invalidating the replay buffer.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
import random


class FrameBuffer:
    """
    Raw frame buffer for continuous VQ-VAE training.
    
    Stores raw RGB frames (uint8) instead of tokens. This enables:
    1. VQ-VAE fine-tuning on recent frames
    2. On-demand encoding with current VQ-VAE
    3. No token staleness issues
    
    Memory usage (100k frames at 84×64×3):
    - 100,000 × 84 × 64 × 3 = 1.6 GB
    
    For grayscale (84×64×1):
    - 100,000 × 84 × 64 × 1 = 0.54 GB
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        frame_height: int = 84,
        frame_width: int = 64,
        channels: int = 3,
    ):
        self.capacity = capacity
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.channels = channels
        
        # Pre-allocate frame storage (uint8 for memory efficiency)
        self.frames = np.zeros(
            (capacity, channels, frame_height, frame_width),
            dtype=np.uint8
        )
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.position = 0
        self.size = 0
        
        # Track memory
        frame_bytes = capacity * channels * frame_height * frame_width
        total_mb = frame_bytes / (1024 * 1024)
        print(f"[FrameBuffer] Allocated {total_mb:.1f} MB for {capacity:,} frames")
    
    def add(
        self,
        frame: np.ndarray,  # (C, H, W) or (H, W, C) uint8
        action: int,
        reward: float,
        done: bool,
    ):
        """Add a raw frame."""
        # Handle different input formats
        if frame.shape[-1] == self.channels:  # (H, W, C) format
            frame = np.transpose(frame, (2, 0, 1))
        
        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        self.frames[self.position] = frame
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_from_tensor(
        self,
        frame: torch.Tensor,  # (C, H, W) normalized [-1, 1] or [0, 1]
        action: int,
        reward: float,
        done: bool,
    ):
        """Add frame from PyTorch tensor (handles normalization)."""
        frame_np = frame.detach().cpu().numpy()
        
        # Convert from [-1, 1] or [0, 1] to [0, 255]
        if frame_np.min() < 0:
            # [-1, 1] range
            frame_np = ((frame_np + 1) * 127.5).clip(0, 255)
        elif frame_np.max() <= 1.0:
            # [0, 1] range
            frame_np = (frame_np * 255).clip(0, 255)
        
        self.add(frame_np.astype(np.uint8), action, reward, done)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch of frames.
        
        Returns:
            frames: (B, C, H, W) normalized to [-1, 1] for VQ-VAE
            actions: (B,) action indices
            rewards: (B,) rewards
            dones: (B,) done flags
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        frames = self.frames[indices].astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
        
        return (
            torch.from_numpy(frames),
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
        )
    
    def sample_recent(
        self,
        batch_size: int,
        recent_k: int = 10000,
    ) -> torch.Tensor:
        """Sample recent frames for VQ-VAE fine-tuning."""
        recent_k = min(recent_k, self.size)
        
        if self.size < self.capacity:
            start = max(0, self.size - recent_k)
            indices = np.random.randint(start, self.size, size=batch_size)
        else:
            offsets = np.random.randint(0, recent_k, size=batch_size)
            indices = (self.position - 1 - offsets) % self.capacity
        
        frames = self.frames[indices].astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(frames)
    
    def sample_all_recent(self, recent_k: int = 10000) -> torch.Tensor:
        """Get all recent frames (for VQ-VAE training batch)."""
        recent_k = min(recent_k, self.size)
        
        if self.size < self.capacity:
            start = max(0, self.size - recent_k)
            indices = np.arange(start, self.size)
        else:
            indices = np.array([(self.position - 1 - i) % self.capacity for i in range(recent_k)])
        
        frames = self.frames[indices].astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(frames)
    
    def encode_batch(
        self,
        vqvae,
        batch_size: int,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Sample and encode frames with current VQ-VAE.
        
        This is the key function for on-demand encoding.
        """
        frames, _, _, _ = self.sample(batch_size)
        frames = frames.to(device)
        
        with torch.no_grad():
            tokens = vqvae.encode(frames)
        
        return tokens
    
    def __len__(self) -> int:
        return self.size


class CodebookRehearsalBuffer:
    """
    Codebook Rehearsal Buffer for preventing VQ-VAE catastrophic forgetting.
    
    Inspired by CUCL (arXiv:2311.14911), this buffer stores frames indexed
    by their primary VQ code, enabling balanced sampling across all codes.
    
    This prevents the VQ-VAE from forgetting early-game states when fine-tuned
    on late-game data, by ensuring all codes are represented in training.
    
    Usage:
        rehearsal = CodebookRehearsalBuffer(vqvae, n_codes=512)
        
        # Add frames during training
        for frame in frames:
            rehearsal.add(frame)
        
        # Sample balanced batch for VQ-VAE fine-tuning
        mixed = rehearsal.sample_balanced(batch_size=128)
    """
    
    def __init__(
        self,
        n_codes: int = 512,
        frames_per_code: int = 100,
        frame_height: int = 84,
        frame_width: int = 64,
        channels: int = 3,
    ):
        self.n_codes = n_codes
        self.frames_per_code = frames_per_code
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.channels = channels
        
        # Storage indexed by code
        # Each code has a ring buffer of frames
        self.code_frames: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.code_positions: Dict[int, int] = defaultdict(int)
        
        # Track which codes have data
        self.active_codes = set()
        
        # VQ-VAE reference for encoding
        self.vqvae = None
        self.device = "cuda"
    
    def set_vqvae(self, vqvae, device: str = "cuda"):
        """Set the VQ-VAE for encoding frames."""
        self.vqvae = vqvae
        self.device = device
    
    def add(self, frame: np.ndarray, primary_code: int = None):
        """
        Add frame to the buffer, indexed by its primary VQ code.
        
        Args:
            frame: (C, H, W) or (H, W, C) uint8 frame
            primary_code: Optional pre-computed code. If None, requires vqvae to be set.
        """
        # Normalize frame format
        if frame.shape[-1] == self.channels:
            frame = np.transpose(frame, (2, 0, 1))
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Get primary code if not provided
        if primary_code is None:
            if self.vqvae is None:
                raise ValueError("Must set vqvae with set_vqvae() or provide primary_code")
            
            with torch.no_grad():
                frame_tensor = torch.from_numpy(frame.astype(np.float32) / 127.5 - 1.0)
                frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                tokens = self.vqvae.encode(frame_tensor)
                primary_code = tokens[0, 0, 0].item()  # First token as primary
        
        # Add to code's ring buffer
        code_list = self.code_frames[primary_code]
        if len(code_list) < self.frames_per_code:
            code_list.append(frame.copy())
        else:
            pos = self.code_positions[primary_code]
            code_list[pos] = frame.copy()
            self.code_positions[primary_code] = (pos + 1) % self.frames_per_code
        
        self.active_codes.add(primary_code)
    
    def add_batch(self, frames: torch.Tensor, tokens: torch.Tensor = None):
        """
        Add batch of frames with their tokens.
        
        Args:
            frames: (B, C, H, W) normalized tensor
            tokens: (B, H, W) token indices (optional, will encode if not provided)
        """
        if tokens is None:
            if self.vqvae is None:
                raise ValueError("Must set vqvae or provide tokens")
            with torch.no_grad():
                tokens = self.vqvae.encode(frames.to(self.device))
        
        frames_np = frames.detach().cpu().numpy()
        tokens_np = tokens.detach().cpu().numpy()
        
        for i in range(len(frames)):
            frame_uint8 = ((frames_np[i] + 1) * 127.5).clip(0, 255).astype(np.uint8)
            primary_code = int(tokens_np[i, 0, 0])  # First token
            self.add(frame_uint8, primary_code=primary_code)
    
    def sample_balanced(self, batch_size: int) -> torch.Tensor:
        """
        Sample uniformly across all active codes.
        
        This ensures the VQ-VAE sees examples from all codes during fine-tuning,
        preventing catastrophic forgetting of rarely-used codes.
        """
        if not self.active_codes:
            raise ValueError("No frames in buffer")
        
        frames = []
        codes = list(self.active_codes)
        
        for _ in range(batch_size):
            # Sample random code
            code = random.choice(codes)
            # Sample random frame from that code
            code_frames = self.code_frames[code]
            frame = random.choice(code_frames)
            frames.append(frame)
        
        # Stack and normalize
        frames_np = np.stack(frames).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(frames_np)
    
    def sample_mixed(
        self,
        batch_size: int,
        recent_frames: torch.Tensor,
        rehearsal_frac: float = 0.5,
    ) -> torch.Tensor:
        """
        Mix recent frames with rehearsal frames.
        
        Args:
            batch_size: Total batch size
            recent_frames: (N, C, H, W) recent frames for VQ-VAE training
            rehearsal_frac: Fraction of batch from rehearsal buffer
        
        Returns:
            Mixed batch of frames
        """
        n_rehearsal = int(batch_size * rehearsal_frac)
        n_recent = batch_size - n_rehearsal
        
        # Sample from recent
        if n_recent > 0 and len(recent_frames) > 0:
            indices = np.random.randint(0, len(recent_frames), size=n_recent)
            recent_batch = recent_frames[indices]
        else:
            recent_batch = torch.empty(0, self.channels, self.frame_height, self.frame_width)
        
        # Sample from rehearsal
        if n_rehearsal > 0 and self.active_codes:
            rehearsal_batch = self.sample_balanced(n_rehearsal)
        else:
            rehearsal_batch = torch.empty(0, self.channels, self.frame_height, self.frame_width)
        
        # Concatenate and shuffle
        if len(recent_batch) > 0 and len(rehearsal_batch) > 0:
            mixed = torch.cat([recent_batch, rehearsal_batch], dim=0)
            perm = torch.randperm(len(mixed))
            return mixed[perm]
        elif len(recent_batch) > 0:
            return recent_batch
        else:
            return rehearsal_batch
    
    def get_code_coverage(self) -> dict:
        """Get statistics about code coverage."""
        n_active = len(self.active_codes)
        coverage_pct = n_active / self.n_codes * 100
        
        frames_per_code = {
            code: len(frames) for code, frames in self.code_frames.items()
        }
        
        return {
            "active_codes": n_active,
            "total_codes": self.n_codes,
            "coverage_pct": coverage_pct,
            "min_frames_per_code": min(frames_per_code.values()) if frames_per_code else 0,
            "max_frames_per_code": max(frames_per_code.values()) if frames_per_code else 0,
            "total_frames": sum(frames_per_code.values()),
        }
    
    def __len__(self) -> int:
        return sum(len(frames) for frames in self.code_frames.values())


def test_frame_buffers():
    """Test the frame buffer implementations."""
    print("=" * 60)
    print("Frame Buffer Tests")
    print("=" * 60)
    
    # Test FrameBuffer
    print(f"\n--- Test 1: FrameBuffer ---")
    buffer = FrameBuffer(
        capacity=1000,
        frame_height=84,
        frame_width=64,
        channels=3,
    )
    
    for i in range(500):
        frame = np.random.randint(0, 256, (3, 84, 64), dtype=np.uint8)
        buffer.add(frame, action=i % 9, reward=float(i), done=(i % 50 == 0))
    
    print(f"Buffer size: {len(buffer)}")
    
    frames, actions, rewards, dones = buffer.sample(32)
    print(f"Sampled frames shape: {frames.shape}")
    print(f"Frame value range: [{frames.min():.2f}, {frames.max():.2f}]")
    
    recent = buffer.sample_recent(64, recent_k=100)
    print(f"Recent frames shape: {recent.shape}")
    
    # Test CodebookRehearsalBuffer
    print(f"\n--- Test 2: CodebookRehearsalBuffer ---")
    rehearsal = CodebookRehearsalBuffer(
        n_codes=512,
        frames_per_code=50,
    )
    
    # Simulate adding frames with random codes
    for i in range(1000):
        frame = np.random.randint(0, 256, (3, 84, 64), dtype=np.uint8)
        code = np.random.randint(0, 512)  # Random code assignment
        rehearsal.add(frame, primary_code=code)
    
    print(f"Total frames: {len(rehearsal)}")
    coverage = rehearsal.get_code_coverage()
    print(f"Active codes: {coverage['active_codes']}/{coverage['total_codes']}")
    print(f"Coverage: {coverage['coverage_pct']:.1f}%")
    
    balanced = rehearsal.sample_balanced(64)
    print(f"Balanced sample shape: {balanced.shape}")
    
    # Test mixed sampling
    recent_frames = torch.randn(100, 3, 84, 64)
    mixed = rehearsal.sample_mixed(64, recent_frames, rehearsal_frac=0.5)
    print(f"Mixed sample shape: {mixed.shape}")
    
    print("\n[OK] All frame buffer tests passed!")


if __name__ == "__main__":
    test_frame_buffers()
