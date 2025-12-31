# Policy Gradient (REINFORCE) agent for Taxi-v3 (extended)
# (Stub for Person D)

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from custom_env import MyCustomEnv


import torch
import torch.nn as nn
import torch.optim as optim


# Discrete policy for MyCustomEnv (Taxi-like)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        logits = self.net(x)
        return logits


class REINFORCEAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-2, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        if obs_tensor.dim() > 1:
            obs_tensor = obs_tensor.flatten().unsqueeze(0)
        elif obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        logits = self.policy(obs_tensor)
        logits = logits.view(-1)  # ensure shape (action_dim,)
        assert logits.dim() == 1, f"logits shape should be 1D, got {logits.shape}"
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()  # shape: ()
        log_prob = dist.log_prob(action)  # shape: ()
        return int(action.item()), float(log_prob.item())

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
    action_dim = 5  # 5 discrete actions for mapping
    agent = REINFORCEAgent(obs_dim, action_dim)
    episodes = 200
    action_map = [
        [0.1, 0.0],    # accelerate
        [-0.05, 0.0],  # brake
        [0.0, -0.5],   # left
        [0.0, 0.5],    # right
        [0.0, 0.0]     # no-op
    ]
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        trajectory = []
        total_reward = 0
        while not done:
            action, log_prob = agent.select_action(obs)
            action_vec = action_map[int(action) % len(action_map)]
            next_obs, reward, terminated, truncated, _ = env.step(action_vec)
            trajectory.append((log_prob, reward))
            obs = next_obs
            total_reward += reward
            done = terminated or truncated
        loss = agent.update(trajectory)
        print(f"Episode {ep+1}: Reward {total_reward:.2f}, Loss {loss:.4f}")
