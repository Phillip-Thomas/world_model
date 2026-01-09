"""
Gold Evaluation Suite for World Model
======================================
Frozen, reproducible evaluation protocol for fair comparison across runs.

Tiers:
- Canary: 500 samples, every epoch, quick regression detection
- FastVal: Frozen balanced subset, every epoch, trend line
- Gold: Full eval + rollout benchmark, every N epochs, checkpoint criterion

Gold Evals:
- Gold-Random: Held-out random episodes
- Gold-Human: Held-out human episodes  
- Gold-Rollout: Open-loop rollouts at horizons 5/10/20
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class GoldMetrics:
    """Metrics from gold evaluation."""
    # 1-step metrics
    loss: float = 0.0
    accuracy: float = 0.0
    important_token_accuracy: float = 0.0  # Bright/moving tokens only
    
    # Rollout metrics (per horizon)
    rollout_acc_h5: float = 0.0
    rollout_acc_h10: float = 0.0
    rollout_acc_h20: float = 0.0
    
    # Transfer-friendly metrics
    mirror_consistency: float = 0.0  # LEFT/RIGHT equivariance (0-100%)
    action_sensitivity: float = 0.0  # KL(LEFT||RIGHT) in paddle region
    
    def to_dict(self) -> dict:
        return asdict(self)


class GoldEvalSuite:
    """
    Frozen evaluation suite for reproducible world model evaluation.
    
    Indices are saved to disk on first run and reloaded thereafter.
    """
    
    def __init__(
        self,
        save_dir: str,
        dataset_size: int,
        n_actions: int = 4,
        canary_size: int = 500,
        fast_val_per_action: int = 1500,
        rollout_sequences: int = 50,
        rollout_horizons: List[int] = None,
        seed: int = 42,
    ):
        self.save_dir = save_dir
        self.dataset_size = dataset_size
        self.n_actions = n_actions
        self.canary_size = canary_size
        self.fast_val_per_action = fast_val_per_action
        self.rollout_sequences = rollout_sequences
        self.rollout_horizons = rollout_horizons or [5, 10, 20]
        self.seed = seed
        
        self.indices_path = os.path.join(save_dir, "gold_eval_indices.json")
        
        # Will be populated by initialize()
        self.canary_indices = None
        self.fast_val_indices = None
        self.gold_random_indices = None
        self.gold_human_indices = None
        self.rollout_start_indices = None
        
    def initialize(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        sample_actions: np.ndarray,
        human_start_idx: int = 0,
        human_frame_count: int = 0,
    ):
        """
        Initialize or load frozen evaluation indices.
        
        Args:
            train_indices: Training set indices
            val_indices: Validation set indices
            sample_actions: Action for each sample
            human_start_idx: Where human data starts in dataset
            human_frame_count: Number of human frames
        """
        if os.path.exists(self.indices_path):
            print(f"  Loading frozen gold eval indices from {self.indices_path}")
            self._load_indices()
            return
        
        print(f"  Creating frozen gold eval indices (seed={self.seed})...")
        rng = np.random.RandomState(self.seed)
        
        # === Canary: Fixed 500 samples from validation ===
        canary_size = min(self.canary_size, len(val_indices))
        self.canary_indices = rng.choice(val_indices, canary_size, replace=False).tolist()
        
        # === FastVal: Balanced by action, frozen ===
        self.fast_val_indices = []
        for action in range(self.n_actions):
            action_mask = sample_actions[val_indices] == action
            action_indices = val_indices[action_mask]
            n_samples = min(self.fast_val_per_action, len(action_indices))
            if n_samples > 0:
                selected = rng.choice(action_indices, n_samples, replace=False)
                self.fast_val_indices.extend(selected.tolist())
        rng.shuffle(self.fast_val_indices)
        
        # === Gold-Random: Random-policy validation samples ===
        if human_start_idx > 0:
            random_val_mask = val_indices < human_start_idx
            self.gold_random_indices = val_indices[random_val_mask].tolist()
        else:
            self.gold_random_indices = val_indices.tolist()
        
        # === Gold-Human: Human gameplay validation samples ===
        if human_frame_count > 0:
            human_val_mask = val_indices >= human_start_idx
            self.gold_human_indices = val_indices[human_val_mask].tolist()
        else:
            self.gold_human_indices = []
        
        # === Rollout start indices: Fixed sequences for rollout eval ===
        # Select from validation set, ensuring we have enough headroom
        valid_for_rollout = [i for i in val_indices if i + max(self.rollout_horizons) < self.dataset_size]
        n_rollout = min(self.rollout_sequences, len(valid_for_rollout))
        self.rollout_start_indices = rng.choice(valid_for_rollout, n_rollout, replace=False).tolist()
        
        self._save_indices()
        print(f"    Canary: {len(self.canary_indices)} samples")
        print(f"    FastVal: {len(self.fast_val_indices)} samples (frozen)")
        print(f"    Gold-Random: {len(self.gold_random_indices)} samples")
        print(f"    Gold-Human: {len(self.gold_human_indices)} samples")
        print(f"    Rollout starts: {len(self.rollout_start_indices)} sequences")
    
    def _save_indices(self):
        """Save frozen indices to disk."""
        data = {
            "canary_indices": self.canary_indices,
            "fast_val_indices": self.fast_val_indices,
            "gold_random_indices": self.gold_random_indices,
            "gold_human_indices": self.gold_human_indices,
            "rollout_start_indices": self.rollout_start_indices,
            "seed": self.seed,
        }
        with open(self.indices_path, 'w') as f:
            json.dump(data, f)
    
    def _load_indices(self):
        """Load frozen indices from disk."""
        with open(self.indices_path, 'r') as f:
            data = json.load(f)
        self.canary_indices = data["canary_indices"]
        self.fast_val_indices = data["fast_val_indices"]
        self.gold_random_indices = data["gold_random_indices"]
        self.gold_human_indices = data["gold_human_indices"]
        self.rollout_start_indices = data["rollout_start_indices"]
        print(f"    Canary: {len(self.canary_indices)}, FastVal: {len(self.fast_val_indices)}")
        print(f"    Gold-Random: {len(self.gold_random_indices)}, Gold-Human: {len(self.gold_human_indices)}")
        print(f"    Rollout starts: {len(self.rollout_start_indices)}")


def compute_important_token_mask(
    target_tokens: torch.Tensor,
    history_tokens: torch.Tensor,
    brightness_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Compute mask for important tokens (bright or moving).
    
    Returns: (B, N) boolean tensor where True = important token
    """
    B, N = target_tokens.shape
    
    with torch.no_grad():
        # Motion: tokens that changed from previous frame
        prev_tokens = history_tokens[:, -1, :]  # (B, N)
        is_moving = target_tokens != prev_tokens
        
        # We don't have decoded brightness here, so use motion as proxy
        # In full eval, we could decode and check brightness
        important = is_moving
    
    return important


def eval_one_step(
    model,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    indices: List[int],
    device: torch.device,
    history_len: int = 4,
    batch_size: int = 32,
) -> Tuple[float, float, float]:
    """
    Evaluate 1-step prediction on fixed indices.
    
    Returns: (loss, accuracy, important_token_accuracy)
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    important_correct = 0
    important_total = 0
    
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            
            # Build batch
            batch_hist = []
            batch_act = []
            batch_target = []
            
            for idx in batch_idx:
                start = idx - history_len + 1
                hist = tokens[start:idx+1]  # (T, N)
                act = actions[idx]
                target = tokens[idx+1]  # (N,)
                
                batch_hist.append(hist)
                batch_act.append(act)
                batch_target.append(target)
            
            batch_hist = torch.stack(batch_hist).to(device)  # (B, T, N)
            batch_act = torch.tensor(batch_act).to(device)
            batch_target = torch.stack(batch_target).to(device)  # (B, N)
            
            with torch.amp.autocast('cuda'):
                logits, loss = model(batch_hist, batch_act, batch_target)
            
            total_loss += loss.item() * len(batch_idx)
            
            preds = logits.argmax(dim=-1)
            correct = (preds == batch_target)
            total_correct += correct.sum().item()
            total_tokens += batch_target.numel()
            
            # Important tokens (moving)
            important_mask = compute_important_token_mask(batch_target, batch_hist)
            if important_mask.sum() > 0:
                important_correct += (correct & important_mask).sum().item()
                important_total += important_mask.sum().item()
    
    avg_loss = total_loss / len(indices) if indices else 0
    accuracy = total_correct / total_tokens * 100 if total_tokens > 0 else 0
    important_acc = important_correct / important_total * 100 if important_total > 0 else 0
    
    return avg_loss, accuracy, important_acc


def eval_rollout(
    model,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    start_indices: List[int],
    horizons: List[int],
    device: torch.device,
    history_len: int = 4,
) -> Dict[int, float]:
    """
    Evaluate open-loop rollouts at multiple horizons.
    
    Returns: {horizon: accuracy} for each horizon
    """
    model.eval()
    max_horizon = max(horizons)
    
    horizon_correct = {h: 0 for h in horizons}
    horizon_total = {h: 0 for h in horizons}
    
    with torch.no_grad():
        for start_idx in start_indices:
            # Initialize history with ground truth
            hist_start = start_idx - history_len + 1
            current_history = tokens[hist_start:start_idx+1].unsqueeze(0).to(device)  # (1, T, N)
            
            # Rollout
            for step in range(max_horizon):
                target_idx = start_idx + 1 + step
                if target_idx >= len(tokens):
                    break
                
                action = actions[start_idx + step]
                action_tensor = torch.tensor([action]).to(device)
                target = tokens[target_idx].unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    logits, _ = model(current_history, action_tensor, None)
                
                preds = logits.argmax(dim=-1)  # (1, N)
                
                # Check accuracy at this horizon
                correct = (preds == target).sum().item()
                total = target.numel()
                
                for h in horizons:
                    if step + 1 == h:
                        horizon_correct[h] += correct
                        horizon_total[h] += total
                
                # Update history with prediction (open-loop)
                current_history = torch.cat([
                    current_history[:, 1:, :],
                    preds.unsqueeze(1)
                ], dim=1)
    
    results = {}
    for h in horizons:
        if horizon_total[h] > 0:
            results[h] = horizon_correct[h] / horizon_total[h] * 100
        else:
            results[h] = 0.0
    
    return results


def eval_mirror_consistency(
    model,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    indices: List[int],
    device: torch.device,
    token_h: int = 21,
    token_w: int = 16,
    history_len: int = 4,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """
    Evaluate LEFT/RIGHT mirror consistency (transfer-friendly metric).
    
    Tests: model(H, LEFT) ≈ flip(model(flip(H), RIGHT))
    
    A model that passes this learns true directional control, not memorization.
    
    Returns: (mirror_consistency %, action_sensitivity KL)
    """
    model.eval()
    
    # Filter to LEFT (3) and RIGHT (2) actions only
    lr_indices = [i for i in indices if actions[i].item() in (2, 3)]
    if len(lr_indices) < 10:
        return 0.0, 0.0
    
    def flip_tokens_h(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Horizontally flip token grid: (B, N) or (B, T, N)."""
        if t.ndim == 2:
            return t.view(-1, h, w).flip(dims=[-1]).view(-1, h * w)
        else:  # (B, T, N)
            B, T, N = t.shape
            return t.view(B, T, h, w).flip(dims=[-1]).view(B, T, N)
    
    def swap_lr_action(a: torch.Tensor) -> torch.Tensor:
        """Swap LEFT (3) ↔ RIGHT (2), keep others."""
        swapped = a.clone()
        swapped[a == 2] = 3
        swapped[a == 3] = 2
        return swapped
    
    total_kl = 0.0
    total_consistency = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for i in range(0, len(lr_indices), batch_size):
            batch_idx = lr_indices[i:i+batch_size]
            
            batch_hist, batch_act = [], []
            for idx in batch_idx:
                start = idx - history_len + 1
                hist = tokens[start:idx+1]
                batch_hist.append(hist)
                batch_act.append(actions[idx])
            
            batch_hist = torch.stack(batch_hist).to(device)  # (B, T, N)
            batch_act = torch.stack(batch_act).to(device)
            
            # Original prediction
            with torch.amp.autocast('cuda'):
                logits_orig, _ = model(batch_hist, batch_act, None)  # (B, N, vocab)
            
            # Flipped prediction: flip history, swap action
            hist_flip = flip_tokens_h(batch_hist, token_h, token_w)
            act_swap = swap_lr_action(batch_act)
            
            with torch.amp.autocast('cuda'):
                logits_flip, _ = model(hist_flip, act_swap, None)
            
            # Unflip the flipped prediction to compare
            logits_flip_unflip = flip_tokens_h(
                logits_flip.view(-1, token_h * token_w, logits_flip.shape[-1]),
                token_h, token_w
            ).view_as(logits_flip)
            
            # Compare distributions via softmax
            probs_orig = F.softmax(logits_orig.float(), dim=-1)
            probs_flip = F.softmax(logits_flip_unflip.float(), dim=-1)
            
            # Consistency: 1 - mean(|p_orig - p_flip|) / 2
            # Perfect mirror = 100%, random = ~50%
            diff = (probs_orig - probs_flip).abs().sum(dim=-1).mean()  # L1 distance
            consistency = (1.0 - diff.item() / 2.0) * 100
            
            # Action sensitivity: KL(LEFT || RIGHT) for same history
            # Higher = actions matter more
            with torch.amp.autocast('cuda'):
                logits_left, _ = model(batch_hist, torch.full_like(batch_act, 3), None)
                logits_right, _ = model(batch_hist, torch.full_like(batch_act, 2), None)
            
            probs_left = F.softmax(logits_left.float(), dim=-1)
            probs_right = F.softmax(logits_right.float(), dim=-1)
            kl = F.kl_div(probs_right.log(), probs_left, reduction='batchmean').item()
            
            total_consistency += consistency * len(batch_idx)
            total_kl += kl * len(batch_idx)
            n_samples += len(batch_idx)
    
    avg_consistency = total_consistency / n_samples if n_samples > 0 else 0.0
    avg_kl = total_kl / n_samples if n_samples > 0 else 0.0
    
    return avg_consistency, avg_kl


def run_gold_eval(
    model,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    gold_suite: GoldEvalSuite,
    device: torch.device,
    tier: str = "canary",  # "canary", "fast", or "gold"
    token_h: int = 21,
    token_w: int = 16,
) -> GoldMetrics:
    """
    Run gold evaluation at specified tier.
    
    Args:
        model: World model
        tokens: All tokens (N, H, W) or (N, N_tokens)
        actions: All actions (N,)
        gold_suite: GoldEvalSuite with frozen indices
        device: torch device
        tier: "canary", "fast", or "gold"
    
    Returns: GoldMetrics
    """
    metrics = GoldMetrics()
    
    # Flatten tokens if needed
    if tokens.ndim == 3:
        tokens = tokens.reshape(len(tokens), -1)
    tokens = torch.from_numpy(tokens).long() if isinstance(tokens, np.ndarray) else tokens
    actions = torch.from_numpy(actions).long() if isinstance(actions, np.ndarray) else actions
    
    if tier == "canary":
        indices = gold_suite.canary_indices
        loss, acc, imp_acc = eval_one_step(model, tokens, actions, indices, device)
        metrics.loss = loss
        metrics.accuracy = acc
        metrics.important_token_accuracy = imp_acc
        
    elif tier == "fast":
        indices = gold_suite.fast_val_indices
        loss, acc, imp_acc = eval_one_step(model, tokens, actions, indices, device)
        metrics.loss = loss
        metrics.accuracy = acc
        metrics.important_token_accuracy = imp_acc
        
    elif tier == "gold":
        # Full gold eval: random + human + rollouts + transfer metrics
        
        # Gold-Random 1-step
        if gold_suite.gold_random_indices:
            loss, acc, imp_acc = eval_one_step(
                model, tokens, actions, gold_suite.gold_random_indices, device
            )
            metrics.loss = loss
            metrics.accuracy = acc
            metrics.important_token_accuracy = imp_acc
        
        # Rollout eval
        if gold_suite.rollout_start_indices:
            rollout_results = eval_rollout(
                model, tokens, actions,
                gold_suite.rollout_start_indices,
                gold_suite.rollout_horizons,
                device
            )
            metrics.rollout_acc_h5 = rollout_results.get(5, 0)
            metrics.rollout_acc_h10 = rollout_results.get(10, 0)
            metrics.rollout_acc_h20 = rollout_results.get(20, 0)
        
        # Mirror consistency (transfer-friendly metric)
        if gold_suite.fast_val_indices:
            mirror_cons, action_sens = eval_mirror_consistency(
                model, tokens, actions, gold_suite.fast_val_indices,
                device, token_h=token_h, token_w=token_w
            )
            metrics.mirror_consistency = mirror_cons
            metrics.action_sensitivity = action_sens
    
    return metrics


def format_gold_metrics(metrics: GoldMetrics, tier: str) -> str:
    """Format metrics for logging."""
    if tier == "canary":
        return f"canary: loss={metrics.loss:.3f}, acc={metrics.accuracy:.1f}%, imp={metrics.important_token_accuracy:.1f}%"
    elif tier == "fast":
        return f"fast_val: acc={metrics.accuracy:.1f}%, imp={metrics.important_token_accuracy:.1f}%"
    elif tier == "gold":
        base = (f"GOLD: acc={metrics.accuracy:.1f}%, imp={metrics.important_token_accuracy:.1f}%, "
                f"roll@5={metrics.rollout_acc_h5:.1f}%, @10={metrics.rollout_acc_h10:.1f}%, @20={metrics.rollout_acc_h20:.1f}%")
        if metrics.mirror_consistency > 0:
            base += f"\n  Mirror: {metrics.mirror_consistency:.1f}%, ActionKL: {metrics.action_sensitivity:.3f}"
        return base
    return str(metrics.to_dict())

