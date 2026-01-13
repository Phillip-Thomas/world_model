"""
Temporal Visual World Model (v2.0 - MDP Model)
===============================================
Extends v1.2 with reward and done prediction heads for RL integration.

Key changes from v1.2:
- Reward head: Two-headed (sign classification + magnitude regression)
- Done head: Binary classification for episode termination
- forward_with_heads(): Returns all predictions for MPC planning

Architecture:
    [frame_t-3, frame_t-2, frame_t-1, frame_t] + [action] → Transformer → [frame_t+1, reward, done]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial tokens."""
    
    def __init__(self, d_model: int, max_h: int = 21, max_w: int = 16):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)
        
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        device = x.device
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        
        row_emb = self.row_embed(rows)
        col_emb = self.col_embed(cols)
        
        pos_emb = torch.cat([
            row_emb.unsqueeze(1).expand(-1, w, -1),
            col_emb.unsqueeze(0).expand(h, -1, -1),
        ], dim=-1)
        
        pos_emb = pos_emb.view(h * w, -1).unsqueeze(0)
        return x + pos_emb


class TemporalVisualWorldModel(nn.Module):
    """
    World model with temporal context from multiple past frames.
    
    The model sees a history of frames, enabling:
    - Motion understanding (velocity, direction)
    - Object permanence (tracking through occlusion)
    - Better temporal consistency in predictions
    
    Input sequence structure:
        [frame_0_tokens, frame_1_tokens, ..., frame_T_tokens, action, query_tokens]
         └─────────────── history ──────────────────────────┘         └── predict ─┘
    """
    
    def __init__(
        self,
        n_vocab: int = 512,
        n_actions: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 10,         # v1.2: 10 layers (was 8)
        token_h: int = 21,          # v1.2: default for 84x64 aspect-preserved
        token_w: int = 16,
        dropout: float = 0.1,
        max_history: int = 4,
    ):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_actions = n_actions
        self.d_model = d_model
        self.token_h = token_h
        self.token_w = token_w
        self.n_tokens = token_h * token_w
        self.max_history = max_history
        
        # === Embeddings ===
        self.token_embed = nn.Embedding(n_vocab, d_model)
        self.action_embed = nn.Embedding(n_actions, d_model)
        
        # Spatial position (within each frame)
        self.spatial_pos = PositionalEncoding2D(d_model, token_h, token_w)
        
        # Temporal position (which frame in history)
        # 0 = oldest, max_history-1 = current, max_history = prediction target
        self.temporal_embed = nn.Embedding(max_history + 1, d_model)
        
        # === v1.6: Strong Action Conditioning ===
        # Per-action learned spatial bias (paddle region gets action-specific offset)
        # This directly encodes "LEFT means paddle moves left" into the architecture
        # Shape: (n_actions, token_h, token_w, d_model)
        self.action_spatial_bias = nn.Parameter(torch.zeros(n_actions, token_h, token_w, d_model))
        nn.init.normal_(self.action_spatial_bias, std=0.1)
        
        # Learnable action scale (starts at 1.0, can grow if actions need more influence)
        self.action_scale = nn.Parameter(torch.ones(1))
        
        # === Transformer ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # === Output ===
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, n_vocab)
        
        # === v2.0: Reward/Done Heads for MDP Model ===
        # Reward prediction: two-headed approach for raw ALE rewards
        # - Sign head: 3-class (negative=0, zero=1, positive=2)
        # - Magnitude head: regression on log(|reward| + 1) for non-zero rewards
        self.reward_sign_head = nn.Linear(d_model, 3)
        self.reward_magnitude_head = nn.Linear(d_model, 1)
        
        # Done prediction: binary classification (with sigmoid)
        self.done_head = nn.Linear(d_model, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _encode_sequence(
        self,
        frame_history: torch.Tensor,    # (batch, n_frames, n_tokens)
        actions: torch.Tensor,           # (batch,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input and run through transformer.
        
        Returns:
            output: (B, N, d_model) - query token outputs after transformer
            context: (B, d_model) - pooled context for reward/done prediction
        """
        B, T, N = frame_history.shape
        device = frame_history.device
        
        # === Embed All History Frames ===
        all_frame_tokens = []
        
        for t in range(T):
            # Token embedding
            frame_emb = self.token_embed(frame_history[:, t, :])  # (B, N, d)
            
            # Add spatial position (same for all frames)
            frame_emb = self.spatial_pos(frame_emb, self.token_h, self.token_w)
            
            # Add temporal position (different for each frame in history)
            temporal_pos = torch.full((1,), t, dtype=torch.long, device=device)
            frame_emb = frame_emb + self.temporal_embed(temporal_pos)
            
            all_frame_tokens.append(frame_emb)
        
        # Concatenate all history frames: (B, T*N, d)
        history_emb = torch.cat(all_frame_tokens, dim=1)
        
        # === Action Embedding ===
        action_emb = self.action_embed(actions).unsqueeze(1)  # (B, 1, d)
        
        # === Query Tokens for Prediction ===
        query_tokens = torch.zeros(B, N, self.d_model, device=device)
        query_tokens = self.spatial_pos(query_tokens, self.token_h, self.token_w)
        
        # Temporal position = T (the "next" frame position)
        query_temporal = torch.full((1,), T, dtype=torch.long, device=device)
        query_tokens = query_tokens + self.temporal_embed(query_temporal)
        
        # === v1.6: STRONG Action Conditioning ===
        # 1. Per-action spatial bias: directly encodes "action X affects region Y"
        #    Shape: (B, H, W, d_model) -> (B, N, d_model)
        action_spatial = self.action_spatial_bias[actions]  # (B, H, W, d_model)
        action_spatial = action_spatial.view(B, N, self.d_model)  # Flatten spatial dims
        
        # 2. Scaled action embedding added to ALL query tokens
        scaled_action = action_emb * self.action_scale  # Learnable scale
        
        # 3. Combine: spatial bias + global action + FiLM modulation
        query_tokens = query_tokens + action_spatial  # Spatial action bias
        query_tokens = query_tokens + scaled_action   # Global action signal
        query_tokens = query_tokens * (1.0 + 0.2 * torch.tanh(scaled_action))  # FiLM (bounded)
        
        # === Build Full Sequence ===
        # [history_tokens (T*N), action (1), query_tokens (N)]
        # Action appears both as sequence token (for attention) AND baked into queries
        sequence = torch.cat([history_emb, action_emb, query_tokens], dim=1)
        
        # === Transformer ===
        transformed = self.transformer(sequence)
        
        # === Extract Predictions ===
        # Last N positions are the query outputs
        output = transformed[:, -N:, :]
        output = self.output_norm(output)
        
        # Pool query tokens to get context for reward/done prediction
        context = output.mean(dim=1)  # (B, d_model)
        
        return output, context
    
    def forward(
        self,
        frame_history: torch.Tensor,    # (batch, n_frames, n_tokens)
        actions: torch.Tensor,           # (batch,)
        target_tokens: Optional[torch.Tensor] = None,
        target_rewards: Optional[torch.Tensor] = None,  # (batch,) raw rewards
        target_dones: Optional[torch.Tensor] = None,    # (batch,) 0/1 floats
        compute_reward_done: bool = False,  # Whether to compute reward/done even without targets
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Forward pass with temporal context and optional reward/done prediction.
        
        Args:
            frame_history: (B, T, N) token indices for T past frames
            actions: (B,) action indices
            target_tokens: (B, N) target tokens for next frame
            target_rewards: (B,) raw reward values (for training reward head)
            target_dones: (B,) done flags as float (for training done head)
            compute_reward_done: if True, always compute reward/done predictions
            
        Returns:
            logits: (B, N, vocab) token predictions
            loss: combined loss if any targets provided, else None
            aux: dict with reward/done predictions and per-head losses (if computed)
        """
        # Encode and run transformer
        output, context = self._encode_sequence(frame_history, actions)
        
        # Token prediction
        logits = self.output_head(output)
        
        # === Compute losses ===
        loss = None
        aux = {}
        
        # Token loss - compute per-sample for trust-weighted imagination
        if target_tokens is not None:
            B = frame_history.shape[0]
            
            # Per-token loss (no reduction)
            token_loss_per_token = F.cross_entropy(
                logits.reshape(-1, self.n_vocab),  # (B*N, vocab)
                target_tokens.reshape(-1),          # (B*N,)
                reduction='none'
            )
            
            # Mean over tokens per sample → (B,) for trust calculation
            token_loss_per_sample = token_loss_per_token.view(B, -1).mean(dim=1)
            
            # Scalar for backward
            token_loss = token_loss_per_sample.mean()
            
            loss = token_loss
            aux['token_loss'] = token_loss.detach()
            aux['token_loss_per_sample'] = token_loss_per_sample.detach()  # For trust-weighted Dyna
        
        # Reward/done predictions (if training or explicitly requested)
        if target_rewards is not None or target_dones is not None or compute_reward_done:
            # Reward prediction
            reward_sign_logits = self.reward_sign_head(context)  # (B, 3)
            reward_magnitude = self.reward_magnitude_head(context).squeeze(-1)  # (B,)
            
            aux['reward_sign_logits'] = reward_sign_logits
            aux['reward_magnitude'] = reward_magnitude
            
            # Done prediction
            done_logits = self.done_head(context).squeeze(-1)  # (B,)
            aux['done_logits'] = done_logits
            
            # Convert to scalar reward prediction for MPC
            # reward = sign * exp(magnitude) - 1
            reward_sign_pred = reward_sign_logits.argmax(dim=-1)  # 0=neg, 1=zero, 2=pos
            sign_multiplier = reward_sign_pred.float() - 1.0  # -1, 0, +1
            aux['reward_pred'] = sign_multiplier * (torch.exp(reward_magnitude) - 1.0)
            aux['done_pred'] = torch.sigmoid(done_logits)
        
        # Reward loss
        if target_rewards is not None:
            device = target_rewards.device
            
            # Convert raw rewards to sign labels: neg=0, zero=1, pos=2
            reward_sign_target = torch.ones_like(target_rewards, dtype=torch.long)
            reward_sign_target[target_rewards < 0] = 0
            reward_sign_target[target_rewards > 0] = 2
            
            reward_sign_loss = F.cross_entropy(reward_sign_logits, reward_sign_target)
            aux['reward_sign_loss'] = reward_sign_loss.detach()
            
            # Magnitude loss (only for non-zero rewards)
            non_zero_mask = target_rewards != 0
            if non_zero_mask.any():
                target_magnitude = torch.log1p(target_rewards.abs())
                reward_mag_loss = F.mse_loss(
                    reward_magnitude[non_zero_mask],
                    target_magnitude[non_zero_mask]
                )
                aux['reward_mag_loss'] = reward_mag_loss.detach()
            else:
                reward_mag_loss = torch.tensor(0.0, device=device)
                aux['reward_mag_loss'] = reward_mag_loss
            
            # Combined reward loss
            reward_loss = reward_sign_loss + 0.5 * reward_mag_loss
            aux['reward_loss'] = reward_loss.detach()
            
            if loss is None:
                loss = 0.1 * reward_loss
            else:
                loss = loss + 0.1 * reward_loss
        
        # Done loss
        if target_dones is not None:
            # Binary CE with positive weight (dones are rare)
            done_loss = F.binary_cross_entropy_with_logits(
                done_logits,
                target_dones,
                pos_weight=torch.tensor(10.0, device=target_dones.device)
            )
            aux['done_loss'] = done_loss.detach()
            
            if loss is None:
                loss = 0.1 * done_loss
            else:
                loss = loss + 0.1 * done_loss
        
        return logits, loss, aux if aux else None
    
    @torch.no_grad()
    def forward_with_heads(
        self,
        frame_history: torch.Tensor,    # (batch, n_frames, n_tokens)
        actions: torch.Tensor,           # (batch,)
        deterministic: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MPC planning - returns next tokens + reward/done predictions.
        
        Args:
            frame_history: (B, T, N) token indices for T past frames
            actions: (B,) action indices
            deterministic: use argmax for next tokens (default True for stable rollouts)
            temperature: sampling temperature if not deterministic
            top_k: top-k sampling if not deterministic
            
        Returns:
            next_tokens: (B, N) predicted next frame tokens
            reward_pred: (B,) predicted scalar reward
            done_pred: (B,) predicted P(done)
        """
        self.eval()
        
        # Encode and run transformer
        output, context = self._encode_sequence(frame_history, actions)
        
        # Token prediction
        logits = self.output_head(output)  # (B, N, vocab)
        
        if deterministic:
            next_tokens = logits.argmax(dim=-1)  # (B, N)
        else:
            logits_temp = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits_temp, top_k, dim=-1)
                logits_temp[logits_temp < v[..., -1:]] = float('-inf')
            probs = F.softmax(logits_temp, dim=-1)
            B, N, V = probs.shape
            next_tokens = torch.multinomial(
                probs.view(-1, V), num_samples=1
            ).view(B, N)
        
        # Reward prediction
        reward_sign_logits = self.reward_sign_head(context)  # (B, 3)
        reward_magnitude = self.reward_magnitude_head(context).squeeze(-1)  # (B,)
        
        # Convert to scalar: reward = sign * (exp(mag) - 1)
        reward_sign = reward_sign_logits.argmax(dim=-1)  # 0=neg, 1=zero, 2=pos
        sign_multiplier = reward_sign.float() - 1.0  # -1, 0, +1
        reward_pred = sign_multiplier * (torch.expm1(reward_magnitude))  # expm1 = exp(x) - 1
        
        # Done prediction
        done_logits = self.done_head(context).squeeze(-1)
        done_pred = torch.sigmoid(done_logits)
        
        return next_tokens, reward_pred, done_pred
    
    @torch.no_grad()
    def predict_next_frame(
        self,
        frame_history: torch.Tensor,    # (T, N) or (1, T, N)
        action: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Predict next frame tokens for inference.
        
        Args:
            frame_history: (T, N) or (1, T, N) token history
            action: action to take
            temperature: sampling temperature (ignored if deterministic=True)
            top_k: if set, only sample from top k tokens (ignored if deterministic=True)
            deterministic: if True, use greedy argmax instead of sampling
            
        Returns:
            next_tokens: (1, N) predicted next frame tokens
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Ensure correct shape (1, T, N)
        if frame_history.dim() == 2:
            frame_history = frame_history.unsqueeze(0)
        frame_history = frame_history.to(device)
        
        action_tensor = torch.tensor([action], device=device)
        
        # Forward (ignore reward/done for legacy compatibility)
        logits, _, _ = self.forward(frame_history, action_tensor)
        
        if deterministic:
            # Greedy argmax - most likely token at each position
            next_tokens = logits.argmax(dim=-1)  # (1, N)
        else:
            # Stochastic sampling with temperature and optional top-k
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[..., -1:]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(
                probs.view(-1, self.n_vocab),
                num_samples=1
            ).view(1, self.n_tokens)
        
        return next_tokens


class TemporalFrameBuffer:
    """
    Maintains a rolling buffer of recent frames for inference.
    
    Usage:
        buffer = TemporalFrameBuffer(max_history=4, n_tokens=336)  # 21x16 for 84x64
        buffer.reset(initial_tokens)
        
        for action in actions:
            frame_history = buffer.get_history()
            next_tokens = model.predict_next_frame(frame_history, action)
            buffer.push(next_tokens)
    """
    
    def __init__(self, max_history: int = 4, n_tokens: int = 336):
        self.max_history = max_history
        self.n_tokens = n_tokens
        self.buffer = None
        
    def reset(self, initial_tokens: torch.Tensor):
        """Reset buffer with initial frame (repeated to fill history)."""
        if initial_tokens.dim() == 1:
            initial_tokens = initial_tokens.unsqueeze(0)
        
        # Fill history with copies of initial frame
        self.buffer = initial_tokens.repeat(self.max_history, 1)
        
    def push(self, new_tokens: torch.Tensor):
        """Add new frame to buffer, removing oldest."""
        if new_tokens.dim() == 2:
            new_tokens = new_tokens.squeeze(0)
        
        # Shift buffer and add new frame
        self.buffer = torch.cat([
            self.buffer[1:],
            new_tokens.unsqueeze(0)
        ], dim=0)
    
    def get_history(self) -> torch.Tensor:
        """Get current frame history (T, N)."""
        return self.buffer


if __name__ == "__main__":
    print("=" * 60)
    print("Temporal Visual World Model (v2.0 - MDP Model) Test")
    print("=" * 60)
    
    # Test with aspect-preserved 84x64 input (21x16 = 336 tokens)
    print("\n--- Test 1: Aspect-Preserved (84x64 -> 21x16 tokens) ---")
    model = TemporalVisualWorldModel(
        n_vocab=512,
        n_actions=4,
        d_model=256,
        n_heads=8,
        n_layers=10,
        token_h=21,
        token_w=16,
        max_history=4,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Token grid: {model.token_h}x{model.token_w} = {model.n_tokens}")
    
    # Test forward (token loss only)
    B, T, N = 4, 4, 336  # batch, history, tokens (21x16)
    frame_history = torch.randint(0, 512, (B, T, N))
    actions = torch.randint(0, 4, (B,))
    target = torch.randint(0, 512, (B, N))
    
    logits, loss, aux = model(frame_history, actions, target)
    
    print(f"\nForward pass (token loss only):")
    print(f"  Input history: {frame_history.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test forward with reward/done targets
    print("\n--- Test 2: Full MDP Forward (tokens + reward + done) ---")
    target_rewards = torch.tensor([0.0, 10.0, -5.0, 50.0])  # Mixed rewards
    target_dones = torch.tensor([0.0, 0.0, 1.0, 0.0])  # One terminal
    
    logits, loss, aux = model(
        frame_history, actions, target,
        target_rewards=target_rewards,
        target_dones=target_dones
    )
    
    print(f"  Combined loss: {loss.item():.4f}")
    print(f"  Token loss: {aux['token_loss'].item():.4f}")
    print(f"  Reward sign loss: {aux['reward_sign_loss'].item():.4f}")
    print(f"  Reward mag loss: {aux['reward_mag_loss'].item():.4f}")
    print(f"  Done loss: {aux['done_loss'].item():.4f}")
    
    # Test forward_with_heads for MPC
    print("\n--- Test 3: forward_with_heads for MPC ---")
    next_tokens, reward_pred, done_pred = model.forward_with_heads(frame_history, actions)
    
    print(f"  Next tokens: {next_tokens.shape}")
    print(f"  Reward predictions: {reward_pred}")
    print(f"  Done predictions: {done_pred}")
    
    # Test with legacy 64x64 input (16x16 = 256 tokens)
    print("\n--- Test 4: Legacy Square (64x64 -> 16x16 tokens) ---")
    model_legacy = TemporalVisualWorldModel(
        n_vocab=512,
        n_actions=4,
        d_model=256,
        n_heads=8,
        n_layers=10,
        token_h=16,
        token_w=16,
        max_history=4,
    )
    
    N_legacy = 256
    frame_history_legacy = torch.randint(0, 512, (B, T, N_legacy))
    target_legacy = torch.randint(0, 512, (B, N_legacy))
    
    logits_legacy, loss_legacy, _ = model_legacy(frame_history_legacy, actions, target_legacy)
    print(f"  Input history: {frame_history_legacy.shape}")
    print(f"  Output logits: {logits_legacy.shape}")
    
    # Test inference with buffer
    print("\n--- Test 5: Inference with Frame Buffer ---")
    buffer = TemporalFrameBuffer(max_history=4, n_tokens=336)
    initial = torch.randint(0, 512, (336,))
    buffer.reset(initial)
    
    for step in range(3):
        history = buffer.get_history()
        next_tokens = model.predict_next_frame(history, action=1)
        buffer.push(next_tokens)
        print(f"  Step {step}: predicted {next_tokens.shape}")
    
    print("\n[OK] Temporal world model v2.0 (MDP Model) working!")



