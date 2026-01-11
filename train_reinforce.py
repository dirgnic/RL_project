"""
REINFORCE (Policy Gradient) with Baseline for MyCustomEnv
=========================================================
Instead of learning Q-values, we directly learn a policy π(a|s),
and use a learned value function V(s) as a baseline to reduce variance.

Key concepts:
1. Policy Network (actor): outputs action probabilities given state
2. Value Network (critic): estimates V(s) as a baseline
3. Monte Carlo: wait until episode ends, then compute discounted returns
4. Policy Gradient with Advantage: ∇J ≈ Σ log π(a|s) * (G - V(s))
"""

import numpy as np
import pandas as pd
import os
import json
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


class ValueNetwork(nn.Module):
    """Value function approximator V(s) used as a learned baseline."""

    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [batch] scalar value per state


class REINFORCEAgent:
    """REINFORCE-style agent with a learned value baseline (actor-critic)."""
    
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(obs_dim, action_dim)
        # Critic / baseline network to reduce variance: V(s)
        self.value = ValueNetwork(obs_dim)
        # Joint optimizer over policy and value parameters
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

    def select_action(self, obs):
        """Sample action from policy distribution."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.policy(obs_tensor).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, trajectory):
        """Update policy using collected trajectory."""
        # Unpack trajectory: each item is (log_prob, reward, state)
        log_probs, rewards, states = zip(*trajectory)

        # Calculate returns (discounted sum of future rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        states_tensor = torch.tensor(np.stack(states), dtype=torch.float32)

        # Critic: fit V(s) ≈ raw return
        values = self.value(states_tensor)

        # Advantage: how much better the return was than what critic expected
        advantages = returns - values.detach()
        # Normalize advantages only (helps policy stability)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Policy gradient with advantage baseline
        log_probs = torch.stack(log_probs)
        policy_loss = -(log_probs * advantages).mean()

        # Value loss (MSE between V and returns)
        value_loss = nn.functional.mse_loss(values, returns)

        # Combined loss (actor + critic) with smaller critic weight
        loss = policy_loss + 0.25 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, path):
        """Save the policy and value networks."""
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
            },
            path,
        )
        
    def load(self, path):
        """Load the policy (and value, if present) networks."""
        state = torch.load(path)
        if isinstance(state, dict) and "policy" in state:
            self.policy.load_state_dict(state["policy"])
            if "value" in state:
                self.value.load_state_dict(state["value"])
        else:
            # Backward-compat: old checkpoints that only stored policy
            self.policy.load_state_dict(state)
        self.policy.eval()


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

    # Load previous best (across runs) if it exists
    os.makedirs('results', exist_ok=True)
    best_avg = -float('inf')
    best_meta_path = 'results/reinforce_best.json'
    if os.path.exists(best_meta_path):
        try:
            with open(best_meta_path, 'r') as f:
                meta = json.load(f)
                best_avg = float(meta.get('best_avg', best_avg))
        except Exception:
            pass
    
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
            # Store (log_prob, reward, state) for actor-critic update
            trajectory.append((log_prob, reward, obs))
            obs = next_obs
            total_reward += reward
            done = terminated or truncated
        
        # Update after episode ends
        loss = agent.update(trajectory)
        rewards_history.append(total_reward)

        # Per-episode log for quick feedback
        print(f"Ep {ep+1}/{episodes} | Reward: {total_reward:.1f} | Loss: {loss:.4f}")

        # Keep best checkpoint logic (will only trigger if episodes >= 100)
        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"  Avg100: {avg:.1f}")

            if avg > best_avg:
                best_avg = avg
                agent.save("results/reinforce_model.pth")
                with open(best_meta_path, 'w') as f:
                    json.dump({'best_avg': best_avg}, f)
                print(f"  ✓ New best REINFORCE model saved (Avg100={best_avg:.2f})")

    # Always save the final model as well (compatible with current obs_dim)
    agent.save("results/reinforce_model.pth")

    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame({'episode': range(1, episodes + 1), 'reward': rewards_history})
    df.to_csv('results/reinforce_training.csv', index=False)
    print("\n✓ Results saved to results/reinforce_training.csv")


if __name__ == "__main__":
    train()
