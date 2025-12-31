# Policy Gradient (REINFORCE) agent for Taxi-v3 (extended)
# (Stub for Person D)

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from custom_env import MyCustomEnv


import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.mean_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_dim = action_dim
    def forward(self, x):
        mean = self.mean_layer(x)
        # Ensure mean is always shape (batch_size, 1)
        if mean.dim() == 2 and mean.shape[1] > 1:
            mean = mean[:, 0].unsqueeze(1)
        std = torch.exp(self.log_std)
        if std.dim() == 1 and std.shape[0] > 1:
            std = std[0].unsqueeze(0)
        return mean, std

class REINFORCEAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-2, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean, std = self.policy(obs_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # Use first value if output is a vector
        return action.view(-1)[0].item(), log_prob.view(-1)[0]
    def update(self, trajectory):
        # trajectory: list of (log_prob, reward)
        returns = []
        G = 0
        for _, r in reversed(trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack([lp for lp, _ in trajectory])
        loss = -(log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    env = MyCustomEnv(render_mode=None)
    obs, _ = env.reset()
    obs_dim = len(obs)
    # Force action_dim = 1 for continuous action environments
    action_dim = 1
    agent = REINFORCEAgent(obs_dim, action_dim)
    episodes = 200
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        trajectory = []
        total_reward = 0
        while not done:
            action, log_prob = agent.select_action(obs)
            action_vec = [action, 0.0]
            next_obs, reward, terminated, truncated, _ = env.step(action_vec)
            trajectory.append((log_prob, reward))
            obs = next_obs
            total_reward += reward
            done = terminated or truncated
        loss = agent.update(trajectory)
        print(f"Episode {ep+1}: Reward {total_reward:.2f}, Loss {loss:.4f}")
