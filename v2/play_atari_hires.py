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
from PIL import Image


# =============================================================================
# Checkpoint Discovery Helpers
# =============================================================================

RUNS_DIR = "checkpoints/v2/atari/runs"

def get_latest_run() -> str:
    """Find the most recent run directory (by name, which is timestamp-based)."""
    runs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), RUNS_DIR)
    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"Runs directory not found: {runs_path}")
    
    run_dirs = [d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {runs_path}")
    
    # Sort by name (timestamp format YYYYMMDD_HHMMSS sorts correctly)
    run_dirs.sort(reverse=True)
    return run_dirs[0]


def get_latest_checkpoint(run_dir: str) -> str:
    """Find the highest-epoch checkpoint in a run directory."""
    runs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), RUNS_DIR, run_dir)
    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"Run directory not found: {runs_path}")
    
    # Find all checkpoint files
    checkpoints = glob.glob(os.path.join(runs_path, "atari_world_model_hires*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in: {runs_path}")
    
    # Parse epoch numbers and find highest
    best_ckpt = None
    best_epoch = -1
    
    for ckpt in checkpoints:
        basename = os.path.basename(ckpt)
        # Match patterns like "atari_world_model_hires_epoch20.pt"
        match = re.search(r'epoch(\d+)\.pt$', basename)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_ckpt = ckpt
        elif basename == "atari_world_model_hires.pt" and best_epoch < 0:
            # Fallback to base checkpoint if no epoch checkpoints
            best_ckpt = ckpt
            best_epoch = 0
    
    if best_ckpt is None:
        raise FileNotFoundError(f"No valid checkpoint found in: {runs_path}")
    
    return best_ckpt


def resolve_model_path(run: str = None, checkpoint: str = None, latest: bool = False) -> str:
    """
    Resolve the model path based on arguments.
    
    Args:
        run: Run directory name (e.g., "20260108_164547") or None for latest
        checkpoint: Checkpoint name (e.g., "epoch20") or None for latest in run
        latest: If True, use latest run and latest checkpoint
        
    Returns:
        Full path to checkpoint file
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    
    # If latest flag or run specified, use runs directory
    if latest or run:
        run_name = run if run else get_latest_run()
        run_path = os.path.join(base_path, RUNS_DIR, run_name)
        
        if checkpoint:
            # Specific checkpoint requested
            ckpt_name = f"atari_world_model_hires_{checkpoint}.pt"
            ckpt_path = os.path.join(run_path, ckpt_name)
            if not os.path.exists(ckpt_path):
                # Try without prefix
                ckpt_path = os.path.join(run_path, f"{checkpoint}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_name} in {run_path}")
            return ckpt_path
        else:
            # Find latest checkpoint in run
            return get_latest_checkpoint(run_name)
    
    # Default: use the base checkpoint path
    return os.path.join(base_path, "checkpoints/v2/atari/atari_world_model_hires.pt")


class AtariWorldPlayerHiRes:
    """Play Atari with high-res world model side-by-side."""
    
    def __init__(
        self,
        game: str = "ALE/Breakout-v5",
        vqvae_path: str = "checkpoints/v2/atari/atari_vqvae_hires.pt",
        model_path: str = "checkpoints/v2/atari/atari_world_model_hires.pt",
        device: str = 'cuda',
        deterministic: bool = True,  # Use greedy argmax by default
        temperature: float = 0.8,    # Only used if deterministic=False
        top_k: int = 5,              # Only used if deterministic=False
    ):
        self.deterministic = deterministic
        self.temperature = temperature
        self.top_k = top_k
        self.device = device
        self.game = game
        self.history_len = 4
        
        # Load models
        print("Loading high-res models...")
        self.vqvae = self._load_vqvae(vqvae_path)
        self.world_model, self.n_actions = self._load_world_model(model_path)
        
        # Create real environment
        print(f"Creating {game}...")
        self.env = gym.make(game, frameskip=4)
        
        # Action mapping for keyboard
        # Breakout: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        self.key_to_action = {
            ' ': 1,      # Space = Fire
            'right': 2,  # Right arrow
            'left': 3,   # Left arrow
        }
        
        # State
        self.real_frame = None
        self.ai_frame = None
        self.frame_history = []
        self.divergence = deque(maxlen=500)  # Rolling window to prevent memory growth
        self.step_count = 0
        
        # Frame timing (Atari 60 FPS with frameskip=4 = 15 FPS effective)
        self.target_fps = 15.0
        self.frame_interval_ms = 1000.0 / self.target_fps  # ~66.67ms
        self.pending_action = 0  # NOOP by default
        self.paused = False
        self.anim = None
        
    def _load_vqvae(self, path: str) -> VQVAEHiRes:
        """Load VQ-VAE with flexible input size."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        # Get dimensions from checkpoint
        input_h = ckpt.get('input_h', 84)
        input_w = ckpt.get('input_w', 64)
        n_embeddings = ckpt.get('n_embeddings', 32)  # Load from checkpoint
        self.input_size = (input_h, input_w)
        self.n_embeddings = n_embeddings
        
        model = VQVAEHiRes(
            in_channels=3, hidden_channels=64, latent_channels=256,
            n_embeddings=n_embeddings, n_residual=2,
            input_size=self.input_size,
        ).to(self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        self.token_h = model.token_h
        self.token_w = model.token_w
        self.n_tokens = model.n_tokens
        
        print(f"  Loaded VQ-VAE ({input_h}x{input_w} -> {self.token_h}x{self.token_w} tokens, {n_embeddings} codes)")
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
        print(f"  Loaded World Model ({token_h}x{token_w}, {n_layers} layers, {n_vocab} vocab, epoch {ckpt.get('epoch', '?')}, acc={ckpt.get('val_acc', 0):.1f}%)")
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
        
        # Initialize history with current frame
        self.frame_history = [self.real_frame.clone() for _ in range(self.history_len)]
        self.divergence.clear()  # Clear deque instead of replacing
        self.step_count = 0
        
    def sync_to_real(self):
        """Sync AI frame to real frame."""
        self.ai_frame = self.real_frame.clone()
        self.frame_history = [self.real_frame.clone() for _ in range(self.history_len)]
        print("Synced AI to real game")
        
    def step(self, action: int):
        """Step both real game and AI model."""
        # Real game step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.real_frame = self._preprocess(obs).to(self.device)
        
        # AI prediction
        with torch.no_grad():
            # Build history tokens
            history = torch.stack(self.frame_history[-self.history_len:]).to(self.device)
            history = history.unsqueeze(0)  # (1, T, C, H, W)
            
            # Tokenize history
            B, T = history.shape[:2]
            hist_tokens = []
            for t in range(T):
                tok = self.vqvae.encode(history[:, t])  # (1, token_h, token_w)
                hist_tokens.append(tok.view(B, self.n_tokens))  # Flatten
            hist_tokens = torch.stack(hist_tokens, dim=1)  # (1, T, n_tokens)
            
            # Predict next tokens
            action_tensor = torch.tensor([action], device=self.device)
            logits, _ = self.world_model(hist_tokens, action_tensor, None)
            
            if self.deterministic:
                # Greedy argmax - most stable, no randomness
                pred_tokens = logits.argmax(dim=-1)  # (1, n_tokens)
            else:
                # Temperature sampling with optional top-k
                logits = logits / self.temperature
                if self.top_k > 0:
                    v, _ = torch.topk(logits, self.top_k, dim=-1)
                    logits[logits < v[..., -1:]] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                pred_tokens = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples=1
                ).view(1, self.n_tokens)
            
            # Decode to image
            pred_tokens = pred_tokens.view(1, self.token_h, self.token_w)
            self.ai_frame = self.vqvae.decode(pred_tokens).squeeze(0)
        
        # Update history with AI frame
        self.frame_history.append(self.ai_frame.clone())
        if len(self.frame_history) > self.history_len * 2:
            self.frame_history = self.frame_history[-self.history_len * 2:]
        
        # Calculate divergence
        div = torch.mean((self.real_frame - self.ai_frame) ** 2).item() * 1000
        self.divergence.append(div)
        
        self.step_count += 1
        
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
        
        fig.suptitle(f"{self.game} | SPACE=Fire  ←→=Move  P=Pause  R=Sync  Q=Quit | {self.target_fps:.0f} FPS", fontsize=10)
        
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
            ax_real.set_title(f"REAL ATARI | Step {self.step_count}", fontsize=12, fontweight='bold')
            ax_ai.set_title(f"AI WORLD MODEL ({self.token_h}x{self.token_w}){div_text}", 
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
    
    parser = argparse.ArgumentParser(
        description='Play Atari with World Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play_atari_hires.py --latest              # Latest checkpoint from latest run
  python play_atari_hires.py --run 20260108_164547 # Latest checkpoint from specific run  
  python play_atari_hires.py --run 20260108_164547 --checkpoint epoch20
        """
    )
    
    # Checkpoint selection
    parser.add_argument('--latest', action='store_true',
                        help='Use latest checkpoint from latest run')
    parser.add_argument('--run', type=str, default=None,
                        help='Run directory name (e.g., 20260108_164547)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint name within run (e.g., epoch20)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Direct path to model checkpoint (overrides --run/--latest)')
    
    # Inference settings
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic sampling instead of greedy argmax')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (only with --stochastic)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top-k sampling (only with --stochastic, 0=disabled)')
    args = parser.parse_args()
    
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = resolve_model_path(
            run=args.run,
            checkpoint=args.checkpoint,
            latest=args.latest
        )
    
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    mode = "stochastic (temp={}, top_k={})".format(args.temperature, args.top_k) if args.stochastic else "deterministic (argmax)"
    print(f"Inference: {mode}")
    print(f"{'='*60}\n")
    
    player = AtariWorldPlayerHiRes(
        model_path=model_path,
        deterministic=not args.stochastic,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    player.play()


if __name__ == "__main__":
    main()



