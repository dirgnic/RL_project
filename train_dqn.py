"""
Deep Q-Network (DQN) for MyCustomEnv
====================================
A neural network approximates Q-values instead of using a table.

Key concepts:
1. Q-Network: Neural net that outputs Q(s,a) for all actions given state s
2. Experience Replay: Store transitions, sample randomly to break correlations
3. Target Network: Separate network for stable Q-value targets
4. Double DQN: Use policy net to select action, target net to evaluate

Architecture: State → [64] → [32] → Q-values
"""

import numpy as np
import pandas as pd
import os
from env import load_environment, DiscreteActionWrapper, FlatObsWrapper
from agents.dqn.agent import DQNAgent


def train():
    """Main training loop."""
    # Setup environment
    raw_env = load_environment("MyCustomEnv")
    env = DiscreteActionWrapper(raw_env)  # Discrete actions (0-4)
    env = FlatObsWrapper(env)             # Flat observation vector

    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent with robust configuration
    agent = DQNAgent(
        state_size=obs_dim,
        action_size=action_dim,
        learning_rate=5e-4,         # Slightly lower LR for stability
        epsilon_decay=0.99,         # Slower decay
        buffer_capacity=50000,
        batch_size=64,
        use_double_dqn=True,
        hidden_dims=[256, 128]      # Larger network
    )

    episodes = 500  # Increased for convergence
    rewards_history = []
    
    print(f"Training DQN for {episodes} episodes...")
    print(f"Observation dim: {obs_dim}, Actions: {action_dim}")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.train_step()
            
            obs = next_obs
            total_reward += reward
            
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        if (ep + 1) % 20 == 0:
            avg = np.mean(rewards_history[-20:])
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.1f} | Avg20: {avg:.1f} | ε: {agent.epsilon:.2f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame({'episode': range(1, episodes + 1), 'reward': rewards_history})
    df.to_csv('results/dqn_training.csv', index=False)
    agent.save("results/dqn_model.pth")
    print("\n✓ Results saved to results/dqn_training.csv")


if __name__ == "__main__":
    train()
