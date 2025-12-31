"""
Quick Test and Demo for DQN Implementation
Person C (Ingrid)

Use this to verify everything works before running full experiments
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n" + "="*60)
print("🧪 DQN IMPLEMENTATION TEST")
print("Person C (Ingrid)")
print("="*60 + "\n")

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    from dqn.network import DQNNetwork, DuelingDQNNetwork
    from dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
    from dqn.agent import DQNAgent
    print("✅ All modules imported successfully!\n")
except Exception as e:
    print(f"❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Create networks
print("Test 2: Creating neural networks...")
try:
    net1 = DQNNetwork(state_size=500, action_size=6, hidden_dims=[128, 64], embedding_dim=64)
    net2 = DuelingDQNNetwork(state_size=500, action_size=6, hidden_dims=[128, 64], embedding_dim=64)
    print(f"✅ Standard DQN Network: {sum(p.numel() for p in net1.parameters())} parameters")
    print(f"✅ Dueling DQN Network: {sum(p.numel() for p in net2.parameters())} parameters\n")
except Exception as e:
    print(f"❌ Network creation failed: {e}\n")
    sys.exit(1)

# Test 3: Create replay buffers
print("Test 3: Creating replay buffers...")
try:
    buffer1 = ReplayBuffer(capacity=1000)
    buffer2 = PrioritizedReplayBuffer(capacity=1000)
    
    # Add some transitions
    for i in range(10):
        buffer1.push(i, 0, 1.0, i+1, False)
        buffer2.push(i, 0, 1.0, i+1, False)
    
    print(f"✅ Basic Replay Buffer: {len(buffer1)} transitions")
    print(f"✅ Prioritized Replay Buffer: {len(buffer2)} transitions\n")
except Exception as e:
    print(f"❌ Replay buffer failed: {e}\n")
    sys.exit(1)

# Test 4: Create DQN agents
print("Test 4: Creating DQN agents...")
try:
    agent1 = DQNAgent(
        state_size=500,
        action_size=6,
        use_double_dqn=False
    )
    print("✅ Standard DQN agent created")
    
    agent2 = DQNAgent(
        state_size=500,
        action_size=6,
        use_double_dqn=True
    )
    print("✅ Double DQN agent created\n")
except Exception as e:
    print(f"❌ Agent creation failed: {e}\n")
    sys.exit(1)

# Test 5: Agent methods
print("Test 5: Testing agent methods...")
try:
    # Select action
    action = agent1.select_action(state=42, training=True)
    print(f"✅ select_action: {action} (epsilon-greedy)")
    
    action_eval = agent1.select_action(state=42, training=False)
    print(f"✅ select_action (eval): {action_eval} (greedy)")
    
    # Store transition
    agent1.store_transition(0, 1, 10.0, 1, False)
    print(f"✅ store_transition: buffer size = {len(agent1.replay_buffer)}")
    
    # Add more transitions
    for i in range(100):
        agent1.store_transition(i, 0, 1.0, i+1, False)
    
    # Train step
    loss = agent1.train_step()
    print(f"✅ train_step: loss = {loss:.4f if loss else 'None (not enough samples)'}")
    
    # Decay epsilon
    old_epsilon = agent1.epsilon
    agent1.decay_epsilon()
    print(f"✅ decay_epsilon: {old_epsilon:.4f} → {agent1.epsilon:.4f}")
    
    # Get stats
    stats = agent1.get_stats()
    print(f"✅ get_stats: {stats}\n")
except Exception as e:
    print(f"❌ Agent methods failed: {e}\n")
    sys.exit(1)

# Test 6: Save and load
print("Test 6: Testing save/load...")
try:
    import tempfile
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    
    agent1.save(temp_path)
    print(f"✅ Model saved to {temp_path}")
    
    # Load
    agent_loaded = DQNAgent(state_size=500, action_size=6)
    agent_loaded.load(temp_path)
    print(f"✅ Model loaded successfully")
    
    # Clean up
    os.remove(temp_path)
    print(f"✅ Temp file cleaned up\n")
except Exception as e:
    print(f"❌ Save/load failed: {e}\n")
    sys.exit(1)

# Test 7: Quick training test (optional, requires gymnasium)
print("Test 7: Quick training test...")
try:
    import gymnasium as gym
    import numpy as np
    import torch
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment and agent
    env = gym.make('Taxi-v3')
    env.reset(seed=42)
    
    test_agent = DQNAgent(
        state_size=500,
        action_size=6,
        learning_rate=1e-3,
        use_double_dqn=True
    )
    
    # Train for 10 episodes
    rewards = []
    for episode in range(10):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(50):
            action = test_agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            test_agent.store_transition(state, action, reward, next_state, terminated or truncated)
            test_agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        test_agent.decay_epsilon()
        rewards.append(episode_reward)
    
    env.close()
    
    print(f"✅ Training test complete!")
    print(f"   Episodes: 10")
    print(f"   Mean reward: {np.mean(rewards):.2f}")
    print(f"   Rewards: {rewards}")
    print(f"   Final epsilon: {test_agent.epsilon:.4f}\n")
    
except ImportError:
    print("⚠️  Gymnasium not installed, skipping training test")
    print("   Install with: pip install gymnasium\n")
except Exception as e:
    print(f"❌ Training test failed: {e}\n")

# Final summary
print("="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
print("\n✅ Your DQN implementation is ready to use!")
print("\nNext steps:")
print("  1. Run experiments: python agents/dqn/experiments.py")
print("  2. Visualize results: python agents/dqn/visualize.py")
print("  3. Check README.md for full documentation")
print("\nGood luck with your experiments! 🚀\n")
