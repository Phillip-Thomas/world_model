"""
Minimal Integration Test for RL Components
===========================================
Tests that all new RL components work together without requiring trained models.
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_world_model_mdp():
    """Test world model with reward/done heads."""
    print("\n" + "=" * 60)
    print("Test 1: World Model MDP Heads")
    print("=" * 60)
    
    from models.temporal_world_model import TemporalVisualWorldModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TemporalVisualWorldModel(
        n_vocab=512,
        n_actions=9,
        d_model=128,  # Smaller for testing
        n_heads=4,
        n_layers=2,   # Fewer layers for speed
        token_h=21,
        token_w=16,
        max_history=4,
    ).to(device)
    
    B, T, N = 4, 4, 336
    history = torch.randint(0, 512, (B, T, N), device=device)
    actions = torch.randint(0, 9, (B,), device=device)
    targets = torch.randint(0, 512, (B, N), device=device)
    rewards = torch.tensor([0.0, 10.0, 0.0, 50.0], device=device)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    
    # Test forward with all targets
    logits, loss, aux = model(history, actions, targets, rewards, dones)
    
    print(f"  Forward with targets:")
    print(f"    Logits: {logits.shape}")
    print(f"    Combined loss: {loss.item():.4f}")
    print(f"    Token loss: {aux['token_loss'].item():.4f}")
    print(f"    Reward loss: {aux['reward_loss'].item():.4f}")
    print(f"    Done loss: {aux['done_loss'].item():.4f}")
    
    # Test forward_with_heads for MPC
    next_tokens, reward_pred, done_pred = model.forward_with_heads(history, actions)
    
    print(f"  forward_with_heads:")
    print(f"    Next tokens: {next_tokens.shape}")
    print(f"    Reward pred: {reward_pred.tolist()}")
    print(f"    Done pred: {done_pred.tolist()}")
    
    print("  [PASS] World model MDP heads work!")
    return True


def test_dataset_returns():
    """Test that datasets return 5-tuples with rewards/dones."""
    print("\n" + "=" * 60)
    print("Test 2: Dataset Returns")
    print("=" * 60)
    
    from data.atari_dataset import AtariTemporalDataset
    from data.multistep_dataset import MultiStepDataset
    
    # Create mock data
    n_frames = 105  # 100 actions + 5 episodes
    n_actions = 100
    
    data = {
        'frames': np.random.randint(0, 512, (n_frames, 21, 16), dtype=np.int64),
        'actions': np.random.randint(0, 9, (n_actions,), dtype=np.int32),
        'rewards': np.random.randn(n_actions).astype(np.float32) * 10,
        'dones': np.zeros(n_actions, dtype=bool),
        'episode_starts': np.array([0, 20, 40, 60, 80]),
        'n_actions': 9,
    }
    # Mark episode ends
    data['dones'][19] = True
    data['dones'][39] = True
    data['dones'][59] = True
    data['dones'][79] = True
    data['dones'][99] = True
    
    # Test AtariTemporalDataset
    print("  AtariTemporalDataset:")
    dataset = AtariTemporalDataset(data, history_len=4, preprocessed_tokens=True)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"    Sample length: {len(sample)}")
        assert len(sample) == 5, f"Expected 5-tuple, got {len(sample)}"
        history, action, next_frame, reward, done = sample
        print(f"    History: {history.shape}")
        print(f"    Action: {action}")
        print(f"    Next frame: {next_frame.shape}")
        print(f"    Reward: {reward}")
        print(f"    Done: {done}")
        print("    [PASS] AtariTemporalDataset returns 5-tuple!")
    
    # Test MultiStepDataset
    print("\n  MultiStepDataset:")
    ms_dataset = MultiStepDataset(
        tokens=data['frames'],
        actions=data['actions'],
        dones=data['dones'],
        episode_starts=data['episode_starts'],
        history_len=4,
        rollout_steps=3,
        rewards=data['rewards'],
    )
    
    if len(ms_dataset) > 0:
        sample = ms_dataset[0]
        print(f"    Sample length: {len(sample)}")
        assert len(sample) == 5, f"Expected 5-tuple, got {len(sample)}"
        history, actions, targets, rewards, dones = sample
        print(f"    History: {history.shape}")
        print(f"    Actions: {actions.shape}")
        print(f"    Targets: {targets.shape}")
        print(f"    Rewards: {rewards.shape if rewards is not None else None}")
        print(f"    Dones: {dones.shape}")
        print("    [PASS] MultiStepDataset returns 5-tuple!")
    
    return True


def test_mpc_agent():
    """Test MPC agent with mock models."""
    print("\n" + "=" * 60)
    print("Test 3: MPC Agent")
    print("=" * 60)
    
    from models.temporal_world_model import TemporalVisualWorldModel
    from models.vqvae_hires import VQVAEHiRes
    from agents.mpc_agent import MPCAgent, MPCConfig
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create small models
    vqvae = VQVAEHiRes(
        in_channels=3,
        latent_channels=64,
        n_embeddings=512,
    ).to(device)
    
    world_model = TemporalVisualWorldModel(
        n_vocab=512,
        n_actions=9,
        d_model=64,
        n_heads=2,
        n_layers=2,
        token_h=21,
        token_w=16,
        max_history=4,
    ).to(device)
    
    # Create agent
    config = MPCConfig(
        horizon=3,
        n_candidates=8,
        gamma=0.99,
        use_cem=False,
    )
    agent = MPCAgent(world_model, vqvae, n_actions=9, config=config, device=device)
    
    # Test reset
    initial_frame = torch.randn(1, 3, 84, 64, device=device)
    agent.reset(initial_frame)
    print(f"  Reset: token_history shape = {agent.token_history.shape}")
    
    # Test action selection (random)
    action = agent.select_action()
    print(f"  Random shooting action: {action}")
    
    # Test CEM
    agent.config.use_cem = True
    action_cem = agent.select_action()
    print(f"  CEM action: {action_cem}")
    
    # Test update
    next_frame = torch.randn(1, 3, 84, 64, device=device)
    agent.update(action, next_frame)
    print(f"  Updated history")
    
    print("  [PASS] MPC Agent works!")
    return True


def test_dqn_agent():
    """Test DQN agent."""
    print("\n" + "=" * 60)
    print("Test 4: DQN Agent")
    print("=" * 60)
    
    from agents.dqn_agent import DQNAgent, DQNConfig
    from agents.replay_buffer import ReplayBuffer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = DQNConfig(
        d_embed=32,
        hidden_dim=64,
        n_hidden=1,
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
    
    params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"  Policy net params: {params:,}")
    
    # Test action selection
    state = torch.randint(0, 512, (4, 336), device=device)
    action = agent.select_action(state)
    print(f"  Selected action: {action}")
    
    # Create buffer and add transitions
    buffer = ReplayBuffer(capacity=100, history_len=4, n_tokens=336)
    for i in range(50):
        s = torch.randint(0, 512, (4, 336))
        ns = torch.randint(0, 512, (4, 336))
        buffer.add(s, i % 9, float(i), ns, i % 20 == 0)
    
    # Train step
    batch = buffer.sample(16)
    loss = agent.train_step(batch)
    print(f"  Training loss: {loss:.4f}")
    
    print("  [PASS] DQN Agent works!")
    return True


def test_replay_buffers():
    """Test replay buffer variants."""
    print("\n" + "=" * 60)
    print("Test 5: Replay Buffers")
    print("=" * 60)
    
    from agents.replay_buffer import ReplayBuffer, DualReplayBuffer, PrioritizedReplayBuffer
    
    # Basic buffer
    buffer = ReplayBuffer(capacity=100, history_len=4, n_tokens=336)
    for i in range(30):
        s = torch.randint(0, 512, (4, 336))
        ns = torch.randint(0, 512, (4, 336))
        buffer.add(s, i % 9, float(i), ns, False)
    
    batch = buffer.sample(8)
    print(f"  ReplayBuffer: {len(buffer)} samples, batch.states={batch.states.shape}")
    
    # Dual buffer
    dual = DualReplayBuffer(
        real_capacity=100,
        imagined_capacity=50,
        history_len=4,
        n_tokens=336,
    )
    for i in range(20):
        s = torch.randint(0, 512, (4, 336))
        ns = torch.randint(0, 512, (4, 336))
        dual.add_real(s, i % 9, 1.0, ns, False)
        dual.add_imagined(s, i % 9, 0.5, ns, False)
    
    mixed = dual.sample_mixed(8)
    print(f"  DualReplayBuffer: real={dual.real_size}, imagined={dual.imagined_size}")
    
    # Prioritized buffer
    prio = PrioritizedReplayBuffer(capacity=100, history_len=4, n_tokens=336)
    for i in range(30):
        s = torch.randint(0, 512, (4, 336))
        ns = torch.randint(0, 512, (4, 336))
        prio.add(s, i % 9, float(i), ns, False)
    
    batch, weights, indices = prio.sample(8)
    print(f"  PrioritizedReplayBuffer: {len(prio)} samples, weights={weights.shape}")
    
    print("  [PASS] Replay buffers work!")
    return True


def test_training_loop_components():
    """Test that training script components work."""
    print("\n" + "=" * 60)
    print("Test 6: Training Loop Components")
    print("=" * 60)
    
    from models.temporal_world_model import TemporalVisualWorldModel
    from agents.dqn_agent import DQNAgent, DQNConfig
    from agents.replay_buffer import DualReplayBuffer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create world model
    world_model = TemporalVisualWorldModel(
        n_vocab=512,
        n_actions=9,
        d_model=64,
        n_heads=2,
        n_layers=2,
        token_h=21,
        token_w=16,
        max_history=4,
    ).to(device)
    world_model.eval()
    
    # Create policy
    policy = DQNAgent(
        n_vocab=512,
        n_actions=9,
        history_len=4,
        n_tokens=336,
        config=DQNConfig(d_embed=32, hidden_dim=64, n_hidden=1),
        device=device,
    )
    
    # Create buffer
    buffer = DualReplayBuffer(
        real_capacity=100,
        imagined_capacity=50,
        history_len=4,
        n_tokens=336,
    )
    
    # Simulate environment steps
    print("  Simulating environment interaction...")
    state = torch.randint(0, 512, (4, 336), device=device)
    
    for step in range(20):
        action = policy.select_action(state)
        next_state = torch.randint(0, 512, (4, 336), device=device)
        reward = float(np.random.randn())
        done = step % 10 == 9
        
        buffer.add_real(state, action, reward, next_state, done)
        state = next_state
    
    print(f"  Real buffer size: {buffer.real_size}")
    
    # Generate imagined experience
    print("  Generating imagined rollouts...")
    start_states = buffer.sample_real_states(4).to(device)
    
    with torch.no_grad():
        for k in range(3):
            actions = torch.randint(0, 9, (4,), device=device)
            next_tokens, reward_pred, done_pred = world_model.forward_with_heads(
                start_states, actions, deterministic=True
            )
            next_states = torch.roll(start_states, -1, dims=1)
            next_states[:, -1, :] = next_tokens
            
            # Store imagined transitions
            for i in range(4):
                buffer.add_imagined(
                    start_states[i], actions[i].item(),
                    reward_pred[i].item(), next_states[i],
                    done_pred[i].item() > 0.5
                )
            
            start_states = next_states
    
    print(f"  Imagined buffer size: {buffer.imagined_size}")
    
    # Train on mixed batch
    print("  Training on mixed batch...")
    batch = buffer.sample_mixed(8)
    loss = policy.train_step(batch)
    print(f"  Training loss: {loss:.4f}")
    
    print("  [PASS] Training loop components work!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("RL Integration Test Suite")
    print("=" * 60)
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    tests = [
        ("World Model MDP", test_world_model_mdp),
        ("Dataset Returns", test_dataset_returns),
        ("MPC Agent", test_mpc_agent),
        ("DQN Agent", test_dqn_agent),
        ("Replay Buffers", test_replay_buffers),
        ("Training Loop", test_training_loop_components),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  All tests passed! RL integration is ready.")
    else:
        print("\n  Some tests failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
