"""
VQ-VAE Reconstruction Analysis
==============================
In-depth analysis of VQ-VAE reconstruction quality for Atari frames.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vqvae_hires import VQVAEHiRes


def load_model_and_data(checkpoint_path: str, data_path: str):
    """Load trained VQ-VAE and game data."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model dimensions from checkpoint
    input_h = ckpt.get('input_h', 84)
    input_w = ckpt.get('input_w', 64)
    n_embeddings = ckpt.get('n_embeddings', 256)
    
    # Infer hidden_channels from layer shapes if not saved
    hidden_channels = ckpt.get('hidden_channels', None)
    if hidden_channels is None:
        # Infer from encoder.initial.weight shape: (hidden_channels, in_channels, 3, 3)
        initial_weight = ckpt['model_state_dict'].get('encoder.initial.weight')
        if initial_weight is not None:
            hidden_channels = initial_weight.shape[0]
        else:
            hidden_channels = 64  # fallback default
    
    print(f"Model: {input_h}x{input_w} input, {n_embeddings} codes, hidden={hidden_channels}")
    
    # Create model
    model = VQVAEHiRes(
        n_embeddings=n_embeddings,
        hidden_channels=hidden_channels,
        input_size=(input_h, input_w),
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load data
    data = np.load(data_path)
    frames = data['frames']
    rewards = data.get('rewards', np.zeros(len(frames)))
    episode_starts = data.get('episode_starts', np.zeros(len(frames), dtype=bool))
    
    print(f"Data: {len(frames)} frames, {episode_starts.sum()} episodes")
    
    return model, frames, rewards, episode_starts, device


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Convert numpy frame to model input tensor."""
    # (H, W, C) -> (C, H, W), normalize to [-1, 1]
    x = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1.0
    return x


def postprocess_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output back to displayable image."""
    # (C, H, W) -> (H, W, C), denormalize
    x = tensor.cpu().detach()
    x = ((x + 1.0) * 127.5).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return x


def analyze_reconstructions(model, frames, device, n_samples=16):
    """Analyze reconstruction quality on sample frames."""
    
    # Sample frames evenly
    indices = np.linspace(0, len(frames)-1, n_samples, dtype=int)
    
    originals = []
    reconstructions = []
    all_indices = []
    l1_errors = []
    
    with torch.no_grad():
        for idx in indices:
            frame = frames[idx]
            x = preprocess_frame(frame).unsqueeze(0).to(device)
            
            recon, vq_loss, tokens = model(x)
            
            originals.append(frame)
            reconstructions.append(postprocess_frame(recon[0]))
            all_indices.append(tokens[0].cpu().numpy())
            
            # Per-pixel L1 error
            l1 = (recon[0] - x[0]).abs().mean().item()
            l1_errors.append(l1)
    
    return originals, reconstructions, all_indices, l1_errors


def analyze_token_usage(model, frames, device, n_samples=1000):
    """Analyze which tokens are actually being used."""
    
    # Sample frames
    indices = np.random.choice(len(frames), min(n_samples, len(frames)), replace=False)
    
    all_tokens = []
    
    with torch.no_grad():
        for idx in indices:
            frame = frames[idx]
            x = preprocess_frame(frame).unsqueeze(0).to(device)
            tokens = model.encode(x)
            all_tokens.append(tokens[0].cpu().numpy().flatten())
    
    all_tokens = np.concatenate(all_tokens)
    
    # Compute histogram
    n_codes = model.n_embeddings
    counts = np.bincount(all_tokens, minlength=n_codes)
    
    return counts


def find_interesting_frames(frames, rewards, episode_starts, n_each=4):
    """Find frames with rewards, episode starts, and regular frames."""
    
    reward_frames = np.where(rewards != 0)[0][:n_each]
    start_frames = np.where(episode_starts)[0][:n_each]
    
    # Mid-game frames (not near starts or rewards)
    mid_game = []
    for i in range(len(frames)):
        if i > 100 and not episode_starts[max(0,i-10):i+1].any() and rewards[max(0,i-10):i+1].sum() == 0:
            mid_game.append(i)
            if len(mid_game) >= n_each:
                break
    
    return {
        'reward': reward_frames,
        'start': start_frames,
        'mid_game': np.array(mid_game[:n_each])
    }


def compute_frame_difference(frames, idx):
    """Compute difference from previous frame."""
    if idx == 0:
        return np.zeros_like(frames[0])
    return np.abs(frames[idx].astype(float) - frames[idx-1].astype(float)).astype(np.uint8)


def create_analysis_figure(model, frames, rewards, episode_starts, device, save_path):
    """Create comprehensive analysis figure."""
    
    print("Analyzing reconstructions...")
    
    # Get interesting frames
    interesting = find_interesting_frames(frames, rewards, episode_starts)
    
    # Create figure with multiple sections
    fig = plt.figure(figsize=(20, 24))
    
    # === Section 1: Random sample reconstructions ===
    print("  Section 1: Random reconstructions")
    n_rand = 8
    rand_indices = np.random.choice(len(frames), n_rand, replace=False)
    
    for i, idx in enumerate(rand_indices):
        frame = frames[idx]
        x = preprocess_frame(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            recon, _, tokens = model(x)
        
        recon_np = postprocess_frame(recon[0])
        diff = np.abs(frame.astype(float) - recon_np.astype(float))
        
        # Original
        ax = fig.add_subplot(8, 6, i*3 + 1)
        ax.imshow(frame)
        ax.set_title(f'Frame {idx}', fontsize=8)
        ax.axis('off')
        
        # Reconstruction
        ax = fig.add_subplot(8, 6, i*3 + 2)
        ax.imshow(recon_np)
        l1 = diff.mean()
        ax.set_title(f'Recon (L1={l1:.1f})', fontsize=8)
        ax.axis('off')
        
        # Difference (amplified)
        ax = fig.add_subplot(8, 6, i*3 + 3)
        ax.imshow((diff * 3).clip(0, 255).astype(np.uint8))
        ax.set_title('Diff (3x)', fontsize=8)
        ax.axis('off')
    
    # === Section 2: Token usage histogram ===
    print("  Section 2: Token usage")
    counts = analyze_token_usage(model, frames, device, n_samples=2000)
    
    ax = fig.add_subplot(8, 2, 9)
    used_codes = counts > 0
    ax.bar(range(len(counts)), counts, width=1.0, color=['blue' if u else 'red' for u in used_codes])
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Usage Count')
    ax.set_title(f'Token Usage: {used_codes.sum()}/{len(counts)} codes used ({100*used_codes.mean():.1f}%)')
    ax.set_xlim(0, len(counts))
    
    # === Section 3: Token usage heatmap ===
    ax = fig.add_subplot(8, 2, 10)
    # Show top-10 used codes
    top_k = 20
    sorted_idx = np.argsort(counts)[::-1][:top_k]
    ax.barh(range(top_k), counts[sorted_idx])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'Code {i}' for i in sorted_idx])
    ax.set_xlabel('Usage Count')
    ax.set_title(f'Top {top_k} Most Used Tokens')
    ax.invert_yaxis()
    
    # === Section 4: Reward frames (important for world model) ===
    print("  Section 3: Reward frame analysis")
    if len(interesting['reward']) > 0:
        for i, idx in enumerate(interesting['reward'][:4]):
            frame = frames[idx]
            x = preprocess_frame(frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                recon, _, _ = model(x)
            
            recon_np = postprocess_frame(recon[0])
            
            ax = fig.add_subplot(8, 4, 21 + i*2)
            ax.imshow(frame)
            ax.set_title(f'Reward Frame {idx}\n(r={rewards[idx]:.0f})', fontsize=8)
            ax.axis('off')
            
            ax = fig.add_subplot(8, 4, 22 + i*2)
            ax.imshow(recon_np)
            ax.set_title('Reconstruction', fontsize=8)
            ax.axis('off')
    
    # === Section 5: Token spatial map ===
    print("  Section 4: Token spatial distribution")
    # Show which tokens map to which spatial locations
    sample_idx = np.random.randint(len(frames))
    frame = frames[sample_idx]
    x = preprocess_frame(frame).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _, tokens = model(x)
    
    tokens_np = tokens[0].cpu().numpy()
    
    ax = fig.add_subplot(8, 3, 22)
    ax.imshow(frame)
    ax.set_title(f'Original Frame {sample_idx}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(8, 3, 23)
    im = ax.imshow(tokens_np, cmap='tab20')
    ax.set_title(f'Token Map ({tokens_np.shape[0]}x{tokens_np.shape[1]})', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(8, 3, 24)
    unique_tokens = np.unique(tokens_np)
    ax.text(0.1, 0.9, f'Unique tokens in frame: {len(unique_tokens)}', transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.7, f'Token IDs: {sorted(unique_tokens)[:10]}...', transform=ax.transAxes, fontsize=8)
    ax.text(0.1, 0.5, f'Most common: {np.bincount(tokens_np.flatten()).argmax()}', transform=ax.transAxes, fontsize=10)
    ax.axis('off')
    ax.set_title('Token Statistics', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis to {save_path}")
    plt.close()


def analyze_ball_reconstruction(model, frames, device, save_path):
    """Specific analysis of ball/paddle reconstruction quality - FULL FRAME (no cropping)."""
    
    print("\nAnalyzing ball/paddle reconstruction (full frame, no clipping)...")
    
    # Find frames with the ball visible
    # Ball in Breakout is typically a small bright/white object
    # Search the entire playfield area (below score, above paddle)
    
    ball_frames = []
    ball_scores = []
    
    for i in range(min(10000, len(frames))):
        frame = frames[i]
        
        # Playfield region (skip score area at top ~15px, paddle area at bottom ~10px)
        playfield = frame[15:74, :, :]
        
        # Ball is typically bright - look for small bright clusters
        # Convert to grayscale-ish by taking max across channels
        brightness = playfield.max(axis=2)
        
        # Find very bright pixels (ball is usually white/bright)
        # Note: Atari frames are normalized to [0, 200] range
        bright_mask = brightness > 170
        n_bright = bright_mask.sum()
        
        # Ball is small (roughly 2-4 pixels wide, so 4-16 total pixels)
        # Also check it's not just the bricks (which form horizontal lines)
        if 4 <= n_bright <= 50:
            # Check if bright pixels are clustered (not spread like bricks)
            # Simple heuristic: check variance of bright pixel positions
            bright_positions = np.argwhere(bright_mask)
            if len(bright_positions) >= 4:
                row_var = bright_positions[:, 0].var()
                col_var = bright_positions[:, 1].var()
                # Ball should be compact - low variance in both dimensions
                if row_var < 20 and col_var < 20:
                    # Additional check: ball shouldn't be at brick height (rows ~5-35 of playfield)
                    mean_row = bright_positions[:, 0].mean()
                    if mean_row > 30:  # Below brick area
                        ball_frames.append(i)
                        ball_scores.append(n_bright)
    
    print(f"  Found {len(ball_frames)} candidate ball frames")
    
    if len(ball_frames) < 4:
        print("  Not enough ball frames found, trying looser criteria...")
        # Fallback: just find frames with some bright pixels in lower playfield
        # Ball area is roughly rows 40-75 (below bricks, above paddle)
        ball_frames = []
        for i in range(min(10000, len(frames))):
            frame = frames[i]
            # Ball area (below bricks ~row 40, above paddle ~row 75)
            ball_area = frame[45:72, 5:59, :]
            brightness = ball_area.max(axis=2)
            n_bright = (brightness > 150).sum()
            if 2 <= n_bright <= 30:  # Small bright cluster = ball
                ball_frames.append(i)
        print(f"  Fallback found {len(ball_frames)} frames")
    
    if len(ball_frames) < 4:
        print("  Still not enough, using evenly spaced frames")
        ball_frames = list(np.linspace(100, len(frames)-100, 8, dtype=int))
    else:
        # Sample from found frames
        ball_frames = list(np.random.choice(ball_frames, min(8, len(ball_frames)), replace=False))
    
    fig, axes = plt.subplots(4, 5, figsize=(16, 14))
    
    for i, idx in enumerate(ball_frames[:4]):
        frame = frames[idx]
        x = preprocess_frame(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            recon, _, tokens = model(x)
        
        recon_np = postprocess_frame(recon[0])
        diff = np.abs(frame.astype(float) - recon_np.astype(float))
        
        row = i
        
        # Full original frame (NO CROPPING)
        axes[row, 0].imshow(frame)
        axes[row, 0].set_title(f'Original Frame {idx}', fontsize=9)
        axes[row, 0].axis('off')
        
        # Full reconstruction (NO CROPPING)
        axes[row, 1].imshow(recon_np)
        axes[row, 1].set_title('Reconstruction', fontsize=9)
        axes[row, 1].axis('off')
        
        # Full difference (NO CROPPING)
        axes[row, 2].imshow((diff * 5).clip(0, 255).astype(np.uint8))
        axes[row, 2].set_title(f'Diff 5x (L1={diff.mean():.2f})', fontsize=9)
        axes[row, 2].axis('off')
        
        # Token map
        tokens_np = tokens[0].cpu().numpy()
        axes[row, 3].imshow(tokens_np, cmap='tab20')
        axes[row, 3].set_title(f'Token Map ({tokens_np.shape[0]}x{tokens_np.shape[1]})', fontsize=9)
        axes[row, 3].axis('off')
        
        # Paddle region zoom (bottom of frame where paddle is)
        # Paddle is typically at rows 75-82 in 84-pixel height
        paddle_region_orig = frame[70:84, :]
        paddle_region_recon = recon_np[70:84, :]
        
        # Stack original and recon vertically for comparison
        paddle_comparison = np.vstack([paddle_region_orig, 
                                       np.ones((2, paddle_region_orig.shape[1], 3), dtype=np.uint8) * 128,
                                       paddle_region_recon])
        axes[row, 4].imshow(paddle_comparison)
        axes[row, 4].set_title('Paddle: Orig (top) vs Recon (bot)', fontsize=8)
        axes[row, 4].axis('off')
    
    plt.suptitle('Ball/Paddle Reconstruction Analysis (Full Frame - No Clipping)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ball/paddle analysis to {save_path}")
    plt.close()


def compute_metrics(model, frames, device, n_samples=1000):
    """Compute quantitative reconstruction metrics."""
    
    print("\nComputing reconstruction metrics...")
    
    indices = np.random.choice(len(frames), min(n_samples, len(frames)), replace=False)
    
    l1_errors = []
    mse_errors = []
    unique_tokens_per_frame = []
    
    with torch.no_grad():
        for idx in indices:
            frame = frames[idx]
            x = preprocess_frame(frame).unsqueeze(0).to(device)
            
            recon, _, tokens = model(x)
            
            l1 = (recon - x).abs().mean().item()
            mse = ((recon - x) ** 2).mean().item()
            unique = len(torch.unique(tokens))
            
            l1_errors.append(l1)
            mse_errors.append(mse)
            unique_tokens_per_frame.append(unique)
    
    metrics = {
        'l1_mean': np.mean(l1_errors),
        'l1_std': np.std(l1_errors),
        'mse_mean': np.mean(mse_errors),
        'mse_std': np.std(mse_errors),
        'unique_tokens_mean': np.mean(unique_tokens_per_frame),
        'unique_tokens_std': np.std(unique_tokens_per_frame),
        'unique_tokens_min': np.min(unique_tokens_per_frame),
        'unique_tokens_max': np.max(unique_tokens_per_frame),
    }
    
    print("\n" + "="*50)
    print("RECONSTRUCTION METRICS")
    print("="*50)
    print(f"L1 Error:        {metrics['l1_mean']:.4f} +/- {metrics['l1_std']:.4f}")
    print(f"MSE Error:       {metrics['mse_mean']:.6f} +/- {metrics['mse_std']:.6f}")
    print(f"PSNR (approx):   {10 * np.log10(4 / metrics['mse_mean']):.1f} dB")  # 4 = (1-(-1))^2 range
    print(f"Unique Tokens/Frame: {metrics['unique_tokens_mean']:.1f} +/- {metrics['unique_tokens_std']:.1f}")
    print(f"Token Range:     [{metrics['unique_tokens_min']}, {metrics['unique_tokens_max']}]")
    print("="*50)
    
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='VQ-VAE Reconstruction Analysis')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/v2/atari/atari_vqvae_hires.pt',
                        help='Path to VQ-VAE checkpoint')
    parser.add_argument('--data', type=str,
                        default='checkpoints/v2/atari/atari_game_data.npz',
                        help='Path to game data')
    parser.add_argument('--output', type=str,
                        default='checkpoints/v2/atari/vqvae_analysis.png',
                        help='Output path for analysis figure')
    args = parser.parse_args()
    
    # Load model and data
    model, frames, rewards, episode_starts, device = load_model_and_data(
        args.checkpoint, args.data
    )
    
    # Compute metrics
    metrics = compute_metrics(model, frames, device)
    
    # Create main analysis figure
    create_analysis_figure(model, frames, rewards, episode_starts, device, args.output)
    
    # Create ball-specific analysis
    ball_output = args.output.replace('.png', '_ball.png')
    analyze_ball_reconstruction(model, frames, device, ball_output)
    
    print(f"\nAnalysis complete!")
    print(f"  Main analysis: {args.output}")
    print(f"  Ball analysis: {ball_output}")


if __name__ == '__main__':
    main()

