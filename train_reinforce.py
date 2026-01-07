"""
REINFORCE (Policy Gradient) for MyCustomEnv
============================================
Instead of learning Q-values, we directly learn a policy π(a|s).

Key concepts:
1. Policy Network: Outputs action probabilities given state
2. Monte Carlo: Wait until episode ends, then update using full returns
3. Policy Gradient: ∇J ≈ Σ log π(a|s) * G  (increase prob of good actions)

This is the simplest policy gradient algorithm - no critic, no baseline.
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from env import load_environment, DiscreteActionWrapper, FlatObsWrapper


class PolicyNetwork(nn.Module):
    """Simple neural network that outputs action probabilities."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)  # Returns logits (unnormalized log-probs)


class REINFORCEAgent:
    """REINFORCE agent with simple policy gradient updates."""
    
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs):
        """Sample action from policy distribution."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.policy(obs_tensor).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, trajectory):
        """Update policy using collected trajectory."""
        # Calculate returns (discounted sum of future rewards)
        returns = []
        G = 0
        for _, reward in reversed(trajectory):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack([lp for lp, _ in trajectory])
        
        # Policy gradient: maximize log_prob * return
        loss = -(log_probs * returns).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def train():
    """Main training loop."""
    # Setup environment
    raw_env = load_environment("MyCustomEnv")
    env = DiscreteActionWrapper(raw_env)
    env = FlatObsWrapper(env)
    
    obs, _ = env.reset()
    obs_dim = len(obs)
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(obs_dim, action_dim, lr=1e-3)
    episodes = 1000
    rewards_history = []
    
    print(f"Training REINFORCE for {episodes} episodes...")
    print(f"Observation dim: {obs_dim}, Actions: {action_dim}")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        trajectory = []
        total_reward = 0
        
        # Collect full episode
        while not done:
            action, log_prob = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((log_prob, reward))
            obs = next_obs
            total_reward += reward
            done = terminated or truncated
        
        # Update after episode ends
        agent.update(trajectory)
        rewards_history.append(total_reward)
        
        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.1f} | Avg100: {avg:.1f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame({'episode': range(1, episodes + 1), 'reward': rewards_history})
    df.to_csv('results/reinforce_training.csv', index=False)
    print("\n✓ Results saved to results/reinforce_training.csv")


if __name__ == "__main__":
    train()
