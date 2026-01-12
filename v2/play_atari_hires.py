"""
Play Atari with High-Resolution World Model
============================================
Interactive comparison of real game vs AI world model.

Supports:
- 21x16 tokens (84x64 aspect-preserved) 
- 16x16 tokens (64x64 legacy)

Usage:
    # Use latest checkpoint from latest run:
    python play_atari_hires.py --latest
    
    # Use specific run (latest checkpoint within it):
    python play_atari_hires.py --run 20260108_164547
    
    # Use specific checkpoint:
    python play_atari_hires.py --run 20260108_164547 --checkpoint epoch20
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import re
import glob
import time
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
from models.temporal_world_model import TemporalVisualWorldModel
from config import WorldModelConfig, get_config_for_checkpoint
from PIL import Image


# =============================================================================
# Game Configuration
# =============================================================================

GAME_CONFIGS = {
    "breakout": {
        "env_name": "ALE/Breakout-v5",
        "base_dir": "checkpoints/v2/atari",
        "key_to_action": {
            ' ': 1,      # Space = Fire
            'right': 2,  # Right arrow
            'left': 3,   # Left arrow
        },
        "action_help": "SPACE=Fire  ←→=Move",
    },
    "mspacman": {
        "env_name": "ALE/MsPacman-v5",
        "base_dir": "checkpoints/v2/mspacman",
        "key_to_action": {
            'up': 1,     # Up arrow
            'right': 2,  # Right arrow  
            'left': 3,   # Left arrow
            'down': 4,   # Down arrow
        },
        "action_help": "↑↓←→=Move",
    },
}

DEFAULT_GAME = "breakout"


def get_game_config(game: str) -> dict:
    """Get configuration for a specific game."""
    game = game.lower()
    if game not in GAME_CONFIGS:
        available = ", ".join(GAME_CONFIGS.keys())
        raise ValueError(f"Unknown game '{game}'. Available: {available}")
    return GAME_CONFIGS[game]


# =============================================================================
# Checkpoint Discovery Helpers
# =============================================================================

def get_runs_dir(base_dir: str) -> str:
    """Get the wm_runs directory for a game's base directory."""
    return os.path.join(base_dir, "wm_runs")


def get_latest_run(base_dir: str) -> str:
    """Find the most recent run directory (by name, which is timestamp-based)."""
    runs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), get_runs_dir(base_dir))
    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"Runs directory not found: {runs_path}")
    
    run_dirs = [d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {runs_path}")
    
    # Sort by name (timestamp format YYYYMMDD_HHMMSS sorts correctly)
    run_dirs.sort(reverse=True)
    return run_dirs[0]


def get_latest_checkpoint(run_dir: str, base_dir: str) -> str:
    """Find the highest-epoch checkpoint in a run directory."""
    runs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), get_runs_dir(base_dir), run_dir)
    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"Run directory not found: {runs_path}")
    
    # Find all checkpoint files (game-agnostic pattern)
    checkpoints = glob.glob(os.path.join(runs_path, "*world_model*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in: {runs_path}")
    
    # Parse epoch numbers and find highest
    best_ckpt = None
    best_epoch = -1
    
    for ckpt in checkpoints:
        basename = os.path.basename(ckpt)
        # Match patterns like "*_epoch20.pt" or "*_best.pt"
        match = re.search(r'epoch(\d+)\.pt$', basename)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_ckpt = ckpt
        elif "_best.pt" in basename and best_epoch < 0:
            # Use best checkpoint as fallback
            best_ckpt = ckpt
            best_epoch = 0
        elif basename.endswith("world_model_hires.pt") and best_epoch < 0:
            # Fallback to base checkpoint
            best_ckpt = ckpt
            best_epoch = 0
    
    if best_ckpt is None:
        raise FileNotFoundError(f"No valid checkpoint found in: {runs_path}")
    
    return best_ckpt


def resolve_model_path(run: str = None, checkpoint: str = None, latest: bool = False, base_dir: str = "checkpoints/v2/atari") -> str:
    """
    Resolve the model path based on arguments.
    
    Args:
        run: Run directory name (e.g., "20260108_164547") or None for latest
        checkpoint: Checkpoint name (e.g., "epoch20", "best") or None for latest in run
        latest: If True, use latest run and latest checkpoint
        base_dir: Game-specific base directory (e.g., "checkpoints/v2/mspacman")
        
    Returns:
        Full path to checkpoint file
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    runs_dir = get_runs_dir(base_dir)
    
    # If latest flag or run specified, use runs directory
    if latest or run:
        run_name = run if run else get_latest_run(base_dir)
        run_path = os.path.join(base_path, runs_dir, run_name)
        
        if checkpoint:
            # Specific checkpoint requested - try multiple naming conventions
            candidates = [
                f"atari_world_model_hires_{checkpoint}.pt",  # legacy epoch checkpoints
                f"atari_world_model_{checkpoint}.pt",         # legacy best checkpoint
                f"world_model_{checkpoint}.pt",               # game-agnostic
                f"{checkpoint}.pt",                           # direct name
            ]
            for ckpt_name in candidates:
                ckpt_path = os.path.join(run_path, ckpt_name)
                if os.path.exists(ckpt_path):
                    return ckpt_path
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint} in {run_path}\n  Tried: {candidates}")
        else:
            # Find latest checkpoint in run
            return get_latest_checkpoint(run_name, base_dir)
    
    # Default: use the base checkpoint path (try game-agnostic then legacy)
    default_path = os.path.join(base_path, base_dir, "world_model_hires.pt")
    if os.path.exists(default_path):
        return default_path
    legacy_path = os.path.join(base_path, base_dir, "atari_world_model_hires.pt")
    if os.path.exists(legacy_path):
        return legacy_path
    raise FileNotFoundError(f"No world model checkpoint found in {base_dir}")


class AtariWorldPlayerHiRes:
    """Play Atari with high-res world model side-by-side."""
    
    def __init__(
        self,
        game_config: dict,  # From GAME_CONFIGS
        model_path: str,
        device: str = 'cuda',
        config: WorldModelConfig = None,  # Load from run dir if None
        # CLI overrides (None = use config value)
        deterministic: bool = None,
        temperature: float = None,
        top_k: int = None,
        logit_smoothing: float = None,
        n_candidates: int = None,
        adaptive_temp: bool = None,
        temp_boost: float = None,
    ):
        # Store game config
        self.game_config = game_config
        base_dir = game_config["base_dir"]
        
        # Load config from checkpoint's run directory if not provided
        if config is None:
            config = get_config_for_checkpoint(model_path)
        self.config = config
        
        # Use config values, with CLI overrides taking precedence
        self.deterministic = deterministic if deterministic is not None else config.inference.deterministic
        self.temperature = temperature if temperature is not None else config.inference.temperature
        self.top_k = top_k if top_k is not None else config.inference.top_k
        
        # Advanced inference settings (with config defaults)
        self.logit_smoothing = logit_smoothing if logit_smoothing is not None else getattr(config.inference, 'logit_smoothing', 0.0)
        self.n_candidates = n_candidates if n_candidates is not None else getattr(config.inference, 'n_candidates', 1)
        self.adaptive_temp = adaptive_temp if adaptive_temp is not None else getattr(config.inference, 'adaptive_temp', False)
        self.temp_boost = temp_boost if temp_boost is not None else getattr(config.inference, 'temp_boost', 0.3)
        
        self.device = device
        self.game = game_config["env_name"]
        self.history_len = config.model.history_len
        
        # Resolve VQ-VAE path from base_dir
        vqvae_path = self._find_vqvae(base_dir)
        
        # Load models
        print("Loading high-res models...")
        self.vqvae = self._load_vqvae(vqvae_path)
        self.world_model, self.n_actions = self._load_world_model(model_path)
        
        # Create real environment
        print(f"Creating {self.game}...")
        self.env = gym.make(self.game, frameskip=4)
        
        # Action mapping for keyboard (game-specific)
        self.key_to_action = game_config["key_to_action"]
        self.action_help = game_config["action_help"]
        
        # State
        self.real_frame = None
        self.ai_frame = None
        self.frame_history = []      # Image history (for display only)
        self.token_history = None    # Token history for world model (avoids decode->encode round-trip)
        self.divergence = deque(maxlen=500)  # Rolling window to prevent memory growth
        self.step_count = 0  # Frames since last reset (AI divergence tracking)
        self.total_frames = 0  # Total frames played (never resets)
        
        # Frame timing (Atari 60 FPS with frameskip=4 = 15 FPS effective)
        self.target_fps = 15.0
        self.frame_interval_ms = 1000.0 / self.target_fps  # ~66.67ms
        self.pending_action = 0  # NOOP by default
        self.paused = False
        self.anim = None
        
        # Logit smoothing state (reduces mode-hopping/flicker)
        self.prev_logits = None
    
    def _find_vqvae(self, base_dir: str) -> str:
        """Find VQ-VAE checkpoint in base directory."""
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        # Try modern naming first, then legacy
        candidates = [
            os.path.join(base_path, base_dir, "vqvae_hires.pt"),
            os.path.join(base_path, base_dir, "atari_vqvae_hires.pt"),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(f"No VQ-VAE checkpoint found in {base_dir}. Tried: {candidates}")
        
    def _load_vqvae(self, path: str) -> VQVAEHiRes:
        """Load VQ-VAE with flexible input size."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        # Get dimensions from checkpoint
        input_h = ckpt.get('input_h', 84)
        input_w = ckpt.get('input_w', 64)
        n_embeddings = ckpt.get('n_embeddings', 64)
        self.input_size = (input_h, input_w)
        self.n_embeddings = n_embeddings
        
        # Infer hidden_channels from layer shapes if not saved
        hidden_channels = ckpt.get('hidden_channels', None)
        if hidden_channels is None:
            initial_weight = ckpt['model_state_dict'].get('encoder.initial.weight')
            if initial_weight is not None:
                hidden_channels = initial_weight.shape[0]
            else:
                hidden_channels = 64  # fallback default
        
        model = VQVAEHiRes(
            in_channels=3, hidden_channels=hidden_channels, latent_channels=256,
            n_embeddings=n_embeddings, n_residual=2,
            input_size=self.input_size,
        ).to(self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        self.token_h = model.token_h
        self.token_w = model.token_w
        self.n_tokens = model.n_tokens
        
        # Store embeddings for candidate selection (embedding distance scoring)
        self.vqvae_embeddings = model.quantizer.embedding.weight.detach()  # (n_codes, D)
        
        print(f"  Loaded VQ-VAE ({input_h}x{input_w} -> {self.token_h}x{self.token_w} tokens, {n_embeddings} codes, hidden={hidden_channels})")
        return model
    
    def _load_world_model(self, path: str):
        """Load world model with flexible token grid."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        n_actions = ckpt['n_actions']  # Must be in checkpoint
        token_h = ckpt.get('token_h', self.token_h)
        token_w = ckpt.get('token_w', self.token_w)
        n_layers = ckpt.get('n_layers', 10)  # v1.2: 10 layers
        n_vocab = ckpt.get('n_vocab', self.n_embeddings)  # Match VQ-VAE
        
        model = TemporalVisualWorldModel(
            n_vocab=n_vocab,
            n_actions=n_actions,
            d_model=256,
            n_heads=8,
            n_layers=n_layers,
            token_h=token_h,
            token_w=token_w,
            max_history=4,
        ).to(self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        # Store checkpoint info for display
        self.wm_epoch = ckpt.get('epoch', -1) + 1  # Convert 0-indexed to 1-indexed
        self.wm_acc = ckpt.get('fast_val_acc', ckpt.get('val_acc', 0))
        
        print(f"  Loaded World Model ({token_h}x{token_w}, {n_layers} layers, {n_vocab} vocab, epoch {self.wm_epoch}, acc={self.wm_acc:.1f}%)")
        
        # v2.22: Validate vocab size matches VQ-VAE
        if n_vocab != self.n_embeddings:
            raise ValueError(
                f"VOCAB MISMATCH: World model has {n_vocab} vocab but VQ-VAE has {self.n_embeddings} codes!\n"
                f"  This will cause CUDA assertion errors.\n"
                f"  Solution: Use a world model trained with the same VQ-VAE codebook size.\n"
                f"  Current training run uses 64-code VQ-VAE - wait for it to complete."
            )
        
        return model, n_actions
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for VQ-VAE (aspect-preserved)."""
        img = Image.fromarray(frame)
        # Resize to VQ-VAE input size (e.g., 84x64 for aspect-preserved)
        img = img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)  # PIL uses (W, H)
        frame = np.array(img)
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
        tensor = tensor / 127.5 - 1.0
        return tensor
    
    def _to_display(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable image."""
        img = (tensor.cpu().permute(1, 2, 0).numpy() + 1) / 2
        return np.clip(img, 0, 1)
    
    def reset(self):
        """Reset both real game and AI state."""
        obs, info = self.env.reset()
        self.real_frame = self._preprocess(obs).to(self.device)
        self.ai_frame = self.real_frame.clone()
        
        # Initialize history with current frame (for display)
        self.frame_history = [self.real_frame.clone() for _ in range(self.history_len)]
        
        # Initialize TOKEN history (avoids decode->encode round-trip)
        with torch.no_grad():
            init_tokens = self.vqvae.encode(self.real_frame.unsqueeze(0))  # (1, H, W)
            init_tokens_flat = init_tokens.view(1, self.n_tokens)  # (1, N)
            self.token_history = init_tokens_flat.repeat(1, self.history_len, 1)  # (1, T, N)
        
        self.divergence.clear()  # Clear deque instead of replacing
        self.step_count = 0  # Frames since last reset (for AI divergence tracking)
        # Note: total_frames is NOT reset here - it tracks total real world frames
        
        # Reset logit smoothing state
        self.prev_logits = None
        
    def sync_to_real(self):
        """Sync AI frame to real frame - complete reset of AI rollout state."""
        self.ai_frame = self.real_frame.clone()
        self.frame_history = [self.real_frame.clone() for _ in range(self.history_len)]
        
        # Sync token history too (fills all 4 history slots with current frame)
        with torch.no_grad():
            real_tokens = self.vqvae.encode(self.real_frame.unsqueeze(0))  # (1, H, W)
            real_tokens_flat = real_tokens.view(1, self.n_tokens)  # (1, N)
            self.token_history = real_tokens_flat.repeat(1, self.history_len, 1)  # (1, T, N)
        
        # Reset logit smoothing state
        self.prev_logits = None
        
        # Reset divergence tracking - this is a fresh rollout
        self.divergence.clear()
        self.step_count = 0
        
        print("Reset AI rollout (synced to real, step=0)")
    
    def _sample_single(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample tokens from logits using current temperature/top-k settings."""
        if self.deterministic:
            return logits.argmax(dim=-1)
        
        # Adaptive temperature: uncertain tokens get higher temp
        if self.adaptive_temp:
            # Per-token confidence from logits
            probs = F.softmax(logits, dim=-1)  # (1, N, V)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (1, N)
            max_entropy = math.log(self.n_embeddings)
            
            # Confidence: 0 = uncertain, 1 = certain
            confidence = 1.0 - (entropy / max_entropy).clamp(0, 1)
            
            # Per-token temperature: base + boost * (1 - confidence)
            # Confident tokens get base temp, uncertain get base + boost
            token_temp = self.temperature + self.temp_boost * (1.0 - confidence)
            token_temp = token_temp.unsqueeze(-1)  # (1, N, 1) for broadcasting
            
            scaled_logits = logits / token_temp
        else:
            # Fixed temperature for all tokens
            scaled_logits = logits / self.temperature
        
        # Apply top-k filtering
        if self.top_k > 0:
            v, _ = torch.topk(scaled_logits, self.top_k, dim=-1)
            scaled_logits[scaled_logits < v[..., -1:]] = float('-inf')
        
        probs = F.softmax(scaled_logits, dim=-1)
        tokens = torch.multinomial(
            probs.view(-1, probs.shape[-1]), num_samples=1
        ).view(1, self.n_tokens)
        return tokens
    
    def _score_candidate(self, tokens: torch.Tensor, prev_tokens: torch.Tensor) -> float:
        """
        Score a candidate based on continuity and embedding distance.
        Higher score = better candidate.
        
        Game-agnostic scoring:
        - Token continuity: prefer candidates with reasonable change rate
        - Embedding distance: prefer small visual changes (smooth transitions)
        """
        # Token continuity: fraction of tokens that stayed the same
        same_mask = (tokens == prev_tokens).float()
        continuity = same_mask.mean().item()
        
        # Change rate (5-15% change per frame is typical for game dynamics)
        change_rate = 1.0 - continuity
        ideal_change = 0.08  # ~8% of tokens should change
        change_penalty = abs(change_rate - ideal_change)
        
        # Embedding distance for changed tokens (small = smooth transition)
        if change_rate > 0.001:  # Only if something changed
            changed_mask = ~same_mask.bool().squeeze()
            if changed_mask.any():
                curr_emb = self.vqvae_embeddings[tokens.squeeze()[changed_mask]]
                prev_emb = self.vqvae_embeddings[prev_tokens.squeeze()[changed_mask]]
                emb_dist = (curr_emb - prev_emb).norm(dim=-1).mean().item()
            else:
                emb_dist = 0.0
        else:
            emb_dist = 0.0
        
        # Combined score (higher = better)
        # Penalize large change rates and large embedding distances
        score = -change_penalty * 10.0 - emb_dist * 0.1
        
        return score
    
    def _sample_with_candidates(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample N candidates and pick the best one based on continuity scoring.
        
        This is game-agnostic: uses token-level continuity and embedding distance,
        not object detection or game-specific heuristics.
        """
        prev_tokens = self.token_history[:, -1, :]  # (1, N) - last frame
        
        if self.n_candidates <= 1 or self.deterministic:
            # No candidate selection - just sample once
            return self._sample_single(logits)
        
        # Sample N candidates
        candidates = []
        for _ in range(self.n_candidates):
            tokens = self._sample_single(logits)
            score = self._score_candidate(tokens, prev_tokens)
            candidates.append((score, tokens))
        
        # Pick best candidate
        best_tokens = max(candidates, key=lambda x: x[0])[1]
        return best_tokens
        
    def step(self, action: int):
        """Step both real game and AI model."""
        # Real game step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.real_frame = self._preprocess(obs).to(self.device)
        
        # AI prediction using TOKEN history (no decode->encode round-trip!)
        with torch.no_grad():
            # Predict next tokens directly from token history
            action_tensor = torch.tensor([action], device=self.device)
            logits, _, _ = self.world_model(self.token_history, action_tensor, None)
            
            # Logit smoothing: blend with previous logits to reduce mode-hopping/flicker
            if self.logit_smoothing > 0 and self.prev_logits is not None:
                alpha = self.logit_smoothing
                logits = (1 - alpha) * logits + alpha * self.prev_logits
            self.prev_logits = logits.clone()
            
            # Sample tokens (with optional candidate selection)
            pred_tokens = self._sample_with_candidates(logits)
            
            # Update token history efficiently (roll + replace last)
            self.token_history = self.token_history.roll(shifts=-1, dims=1)
            self.token_history[:, -1, :] = pred_tokens
            
            # Decode to image for display only
            pred_tokens_2d = pred_tokens.view(1, self.token_h, self.token_w)
            self.ai_frame = self.vqvae.decode(pred_tokens_2d).squeeze(0)
        
        # Update image history (for display only, not used in prediction)
        self.frame_history.append(self.ai_frame.clone())
        if len(self.frame_history) > self.history_len * 2:
            self.frame_history = self.frame_history[-self.history_len * 2:]
        
        # Calculate divergence
        div = torch.mean((self.real_frame - self.ai_frame) ** 2).item() * 1000
        self.divergence.append(div)
        
        self.step_count += 1
        self.total_frames += 1
        
        return terminated or truncated
    
    def play(self):
        """Interactive play loop with real-time frame progression."""
        os.chdir(os.path.dirname(os.path.dirname(__file__)))
        
        self.reset()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        ax_real = axes[0]
        ax_ai = axes[1]
        ax_div = axes[2]
        
        # Initial display
        im_real = ax_real.imshow(self._to_display(self.real_frame))
        ax_real.set_title("REAL ATARI", fontsize=12, fontweight='bold')
        ax_real.axis('off')
        
        im_ai = ax_ai.imshow(self._to_display(self.ai_frame))
        ax_ai.set_title(f"AI WORLD MODEL ({self.token_h}x{self.token_w})", fontsize=12, fontweight='bold')
        ax_ai.axis('off')
        
        ax_div.set_xlim(0, 100)
        ax_div.set_ylim(0, 100)
        ax_div.set_xlabel('Step')
        ax_div.set_ylabel('Divergence')
        ax_div.set_title('AI Drift from Reality')
        line_div, = ax_div.plot([], [], 'r-', linewidth=2)
        
        fig.suptitle(f"{self.game} | {self.action_help}  P=Pause  R=Sync  Q=Quit | {self.target_fps:.0f} FPS", fontsize=10)
        
        # Disable default key shortcuts
        for key in ['s', 'l', 'g', 'p']:
            if key in plt.rcParams['keymap.save']:
                plt.rcParams['keymap.save'].remove(key)
        
        # Key handling - sets pending action for next frame
        def on_key_press(event):
            if event.key == 'q':
                if self.anim:
                    self.anim.event_source.stop()
                plt.close()
                return
            elif event.key == 'r':
                self.sync_to_real()
                return
            elif event.key == 'p':
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RUNNING"
                print(f"Game {status}")
                return
            
            # Set pending action (will be used on next frame)
            self.pending_action = self.key_to_action.get(event.key, self.pending_action)
        
        def on_key_release(event):
            # Return to NOOP when key released
            if event.key in self.key_to_action:
                self.pending_action = 0
        
        # Animation update function (runs at target FPS)
        def update_frame(frame_num):
            if self.paused:
                return [im_real, im_ai, line_div]
            
            # Step with current pending action
            action = self.pending_action
            done = self.step(action)
            
            # Update displays
            im_real.set_data(self._to_display(self.real_frame))
            im_ai.set_data(self._to_display(self.ai_frame))
            
            # Update divergence plot (with rolling window)
            if len(self.divergence) > 1:
                xs = np.arange(len(self.divergence))
                line_div.set_data(xs, list(self.divergence))
                ax_div.set_xlim(0, max(100, len(self.divergence)))
                ax_div.relim()
                ax_div.autoscale_view(scalex=False, scaley=True)
            
            div_text = f" | Div: {self.divergence[-1]:.1f}" if self.divergence else ""
            ax_real.set_title(f"REAL ATARI | Frame {self.total_frames}", fontsize=12, fontweight='bold')
            ax_ai.set_title(f"AI (Epoch {self.wm_epoch}, {self.wm_acc:.1f}%) | Since Reset: {self.step_count}{div_text}", 
                           fontsize=12, fontweight='bold')
            
            if done:
                self.reset()
            
            return [im_real, im_ai, line_div]
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('key_release_event', on_key_release)
        
        # Create animation at real Atari frame rate
        self.anim = FuncAnimation(
            fig, 
            update_frame, 
            interval=self.frame_interval_ms,  # ~66.67ms = 15 FPS
            blit=False,
            cache_frame_data=False,
        )
        
        plt.tight_layout()
        plt.show()
        
        self.env.close()


def main():
    import argparse
    
    available_games = ", ".join(GAME_CONFIGS.keys())
    
    parser = argparse.ArgumentParser(
        description='Play Atari with World Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python play_atari_hires.py --game breakout --latest     # Breakout with latest checkpoint
  python play_atari_hires.py --game mspacman --latest     # Ms. Pac-Man with latest checkpoint
  python play_atari_hires.py --run 20260108_164547        # Specific run (default: breakout)
  python play_atari_hires.py --run 20260108_164547 --checkpoint best

Available games: {available_games}

Inference settings are loaded from the run's config.json by default.
Use --stochastic, --temperature, --top-k to override.
        """
    )
    
    # Game selection
    parser.add_argument('--game', type=str, default=DEFAULT_GAME,
                        help=f'Game to play ({available_games}). Default: {DEFAULT_GAME}')
    
    # Checkpoint selection
    parser.add_argument('--latest', action='store_true',
                        help='Use latest checkpoint from latest run')
    parser.add_argument('--run', type=str, default=None,
                        help='Run directory name (e.g., 20260108_164547)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint name within run (e.g., epoch20, best)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Direct path to model checkpoint (overrides --run/--latest)')
    
    # Inference settings (None = use config from run)
    parser.add_argument('--stochastic', action='store_true', default=None,
                        help='Use stochastic sampling (overrides config)')
    parser.add_argument('--deterministic', action='store_true', default=None,
                        help='Use deterministic argmax (overrides config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (overrides config)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-k sampling (overrides config)')
    # Advanced inference (None = use config from run)
    parser.add_argument('--logit-smoothing', type=float, default=None,
                        help='Blend logits with previous step (0.0-0.5, reduces flicker)')
    parser.add_argument('--n-candidates', type=int, default=None,
                        help='Sample N candidates, pick best by continuity (1=disabled, 4-8=recommended)')
    parser.add_argument('--adaptive-temp', action='store_true', default=None,
                        help='Enable confidence-adaptive temperature (uncertain tokens get more randomness)')
    parser.add_argument('--no-adaptive-temp', action='store_true',
                        help='Disable confidence-adaptive temperature')
    parser.add_argument('--temp-boost', type=float, default=None,
                        help='Extra temperature for uncertain tokens when adaptive (default: 0.3)')
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    # Get game configuration
    game_config = get_game_config(args.game)
    base_dir = game_config["base_dir"]
    
    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = resolve_model_path(
            run=args.run,
            checkpoint=args.checkpoint,
            latest=args.latest,
            base_dir=base_dir
        )
    
    # Determine deterministic override (only if explicitly set)
    deterministic_override = None
    if args.stochastic:
        deterministic_override = False
    elif args.deterministic:
        deterministic_override = True
    
    # Determine adaptive_temp override
    adaptive_temp_override = None
    if args.adaptive_temp:
        adaptive_temp_override = True
    elif args.no_adaptive_temp:
        adaptive_temp_override = False
    
    print(f"\n{'='*60}")
    print(f"Game: {args.game} ({game_config['env_name']})")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    player = AtariWorldPlayerHiRes(
        game_config=game_config,
        model_path=model_path,
        deterministic=deterministic_override,
        temperature=args.temperature,
        top_k=args.top_k,
        logit_smoothing=args.logit_smoothing,
        n_candidates=args.n_candidates,
        adaptive_temp=adaptive_temp_override,
        temp_boost=args.temp_boost,
    )
    
    # Print effective inference config
    mode = "stochastic" if not player.deterministic else "deterministic"
    extras = []
    if player.logit_smoothing > 0:
        extras.append(f"smoothing={player.logit_smoothing}")
    if player.n_candidates > 1:
        extras.append(f"candidates={player.n_candidates}")
    if player.adaptive_temp:
        extras.append(f"adaptive(boost={player.temp_boost})")
    extras_str = ", " + ", ".join(extras) if extras else ""
    print(f"\nInference: {mode} (temp={player.temperature}, top_k={player.top_k}{extras_str})")
    print(f"Config source: {player.config.run_dir or 'defaults'}")
    print("=" * 60 + "\n")
    
    player.play()


if __name__ == "__main__":
    main()



