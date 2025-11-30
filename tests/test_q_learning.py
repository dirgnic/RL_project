"""
Test Q-Learning agent.
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from env import load_environment
from agents.q_learning.q_table import QTable
from agents.q_learning.utils import epsilon_greedy, decay_epsilon, evaluate_policy


def test_q_table_discrete():
    """Test Q-Table with discrete states."""
    print("\n=== Testing Q-Table (Discrete States) ===")
    
    q_table = QTable(num_actions=4)
    
    # Test get and update
    state = (0, 1, 2)
    action = 1
    
    # Initial value should be 0
    q_value = q_table.get_q_value(state, action)
    assert q_value == 0.0, "Initial Q-value should be 0"
    print(f"✓ Initial Q-value: {q_value}")
    
    # Update value
    q_table.update(state, action, 5.0)
    q_value = q_table.get_q_value(state, action)
    assert q_value == 5.0, "Updated Q-value should be 5.0"
    print(f"✓ Updated Q-value: {q_value}")
    
    # Test max Q-value
    q_table.update(state, 2, 10.0)
    max_q = q_table.get_max_q_value(state)
    assert max_q == 10.0, "Max Q-value should be 10.0"
    print(f"✓ Max Q-value: {max_q}")
    
    # Test best action
    best_action = q_table.get_best_action(state)
    assert best_action == 2, "Best action should be 2"
    print(f"✓ Best action: {best_action}")
    
    # Test stats
    stats = q_table.get_stats()
    print(f"✓ Q-Table stats: {stats}")


def test_q_table_continuous():
    """Test Q-Table with continuous states."""
    print("\n=== Testing Q-Table (Continuous States) ===")
    
    # CartPole state bounds
    state_bounds = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
    q_table = QTable(num_actions=2, num_bins=[5, 5, 5, 5], state_bounds=state_bounds)
    
    # Test discretization
    continuous_state = np.array([0.5, 0.2, 0.1, 0.3])
    discrete_state = q_table.discretize_state(continuous_state)
    print(f"✓ Continuous state: {continuous_state}")
    print(f"✓ Discrete state: {discrete_state}")
    
    # Test Q-value operations with continuous states
    q_table.update(continuous_state, 0, 3.5)
    q_value = q_table.get_q_value(continuous_state, 0)
    assert abs(q_value - 3.5) < 0.001, "Q-value should be 3.5"
    print(f"✓ Q-value for continuous state: {q_value}")
    
    # Test save and load
    save_path = project_root / "tests" / "test_q_table.pkl"
    q_table.save(save_path)
    print(f"✓ Saved Q-Table to {save_path}")
    
    loaded_q_table = QTable(num_actions=2, num_bins=[5, 5, 5, 5], state_bounds=state_bounds)
    loaded_q_table.load(save_path)
    loaded_q_value = loaded_q_table.get_q_value(continuous_state, 0)
    assert abs(loaded_q_value - 3.5) < 0.001, "Loaded Q-value should match"
    print(f"✓ Loaded Q-Table, Q-value: {loaded_q_value}")
    
    # Clean up
    save_path.unlink()


def test_epsilon_greedy():
    """Test epsilon-greedy action selection."""
    print("\n=== Testing Epsilon-Greedy ===")
    
    q_table = QTable(num_actions=4)
    state = (0, 0)
    
    # Set Q-values
    q_table.update(state, 0, 1.0)
    q_table.update(state, 1, 5.0)  # Best action
    q_table.update(state, 2, 2.0)
    q_table.update(state, 3, 0.5)
    
    # Test greedy (epsilon=0)
    greedy_actions = [epsilon_greedy(q_table, state, 0.0, 4) for _ in range(10)]
    assert all(a == 1 for a in greedy_actions), "Greedy should always choose best action"
    print(f"✓ Greedy actions (epsilon=0): {greedy_actions[:5]}")
    
    # Test random (epsilon=1)
    random_actions = [epsilon_greedy(q_table, state, 1.0, 4) for _ in range(100)]
    unique_actions = len(set(random_actions))
    assert unique_actions > 1, "Random should explore different actions"
    print(f"✓ Random actions (epsilon=1): {unique_actions} unique actions from 100 samples")
    
    # Test epsilon decay
    epsilon = 1.0
    for _ in range(10):
        epsilon = decay_epsilon(epsilon, 0.01, 0.9)
    print(f"✓ Epsilon after 10 decays (rate=0.9): {epsilon:.4f}")


def test_mini_training():
    """Test mini training loop."""
    print("\n=== Testing Mini Training Loop ===")
    
    # Create environment
    env = load_environment("CartPole-v1", seed=42)
    
    # Initialize Q-Table
    state_bounds = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
    q_table = QTable(num_actions=2, num_bins=[3, 3, 3, 3], state_bounds=state_bounds)
    
    # Mini training
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    
    print("Running 5 episodes...")
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 50:
            action = epsilon_greedy(q_table, state, epsilon, 2)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Q-Learning update
            current_q = q_table.get_q_value(state, action)
            if done:
                target_q = reward
            else:
                max_next_q = q_table.get_max_q_value(next_state)
                target_q = reward + gamma * max_next_q
            
            new_q = current_q + alpha * (target_q - current_q)
            q_table.update(state, action, new_q)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        epsilon = decay_epsilon(epsilon, 0.01, 0.9)
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}, Epsilon = {epsilon:.4f}")
    
    stats = q_table.get_stats()
    print(f"✓ Training completed. Q-Table has {stats['num_states']} states")
    
    env.close()


if __name__ == "__main__":
    test_q_table_discrete()
    test_q_table_continuous()
    test_epsilon_greedy()
    test_mini_training()
    
    print("\n" + "=" * 60)
    print("All Q-Learning tests passed!")
    print("=" * 60)
