"""
Tabular Q-Learning for MyCustomEnv
==================================
A simple, didactic implementation of Q-Learning with a discrete state table.

How it works:
1. The environment gives us continuous observations
2. We discretize them into bins (grid position, fuel level, etc.)
3. We maintain a Q-table: Q[state][action] = expected future reward
4. We update Q-values using the Bellman equation:
   Q(s,a) ← Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
"""

import numpy as np
import pandas as pd
import os
import json
from env import load_environment, DiscreteActionWrapper, FlatObsWrapper, TabularObsWrapper


class TabularQAgent:
    """Simple Q-Learning agent with a table of Q-values."""
    
    def __init__(self, obs_bins, action_size, alpha=0.05, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99):
        """
        Args:
            obs_bins: List of bin counts for each observation dimension
            action_size: Number of possible actions
            alpha: Learning rate (how much to update Q-values)
            gamma: Discount factor (importance of future rewards)
            epsilon: Exploration rate (probability of random action)
        """
        self.obs_bins = obs_bins
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: shape is [bin1, bin2, ..., binN, action_size]
        self.q_table = np.zeros(obs_bins + [action_size])

    def discretize(self, obs):
        """Convert continuous observation to discrete state tuple."""
        return tuple(obs.astype(int))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, done):
        """Update Q-value using Bellman equation."""
        best_next = np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next * (1 - done)
        error = target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.alpha * error

    def decay_epsilon(self):
        """Reduce exploration over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save Q-table to file."""
        np.save(path, self.q_table)

    def load(self, path):
        """Load Q-table from file."""
        self.q_table = np.load(path)


def train():
    """Main training loop."""
    # Setup environment with wrappers
    raw_env = load_environment("MyCustomEnv")
    env = DiscreteActionWrapper(raw_env)    # Discrete actions (0-4)
    env = FlatObsWrapper(env)               # Flat observation vector
    env = TabularObsWrapper(env, grid_bins=6, fuel_bins=3)  # Discretize for table

    # State space: [x_bin, y_bin, fuel_bin, passenger, dx_bin, dy_bin]
    obs_bins = [6, 6, 3, 2, 3, 3]
    action_size = env.action_space.n
    
    agent = TabularQAgent(obs_bins, action_size)
    episodes = 500
    rewards_history = []

    # Load previous best (across runs) if it exists
    os.makedirs('results', exist_ok=True)
    best_avg = -float('inf')
    best_meta_path = 'results/tabular_best.json'
    if os.path.exists(best_meta_path):
        try:
            with open(best_meta_path, 'r') as f:
                meta = json.load(f)
                best_avg = float(meta.get('best_avg', best_avg))
        except Exception:
            pass
    
    print(f"Training Tabular Q-Learning for {episodes} episodes...")
    print(f"State space size: {np.prod(obs_bins)} states, {action_size} actions")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        state = agent.discretize(obs)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.discretize(next_obs)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.1f} | Avg100: {avg:.1f} | ε: {agent.epsilon:.2f}")

            # Save best model so far based on Avg100
            if avg > best_avg:
                best_avg = avg
                agent.save("results/tabular_model.npy")
                # Persist new best metric
                with open(best_meta_path, 'w') as f:
                    json.dump({'best_avg': best_avg}, f)
                print(f"✓ New best Tabular model saved (Avg100={best_avg:.2f})")

    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame({'episode': range(1, episodes + 1), 'reward': rewards_history})
    df.to_csv('results/tabular_training.csv', index=False)
    print("\n✓ Results saved to results/tabular_training.csv")


if __name__ == "__main__":
    train()
