# Q-learning and SARSA for Taxi-v3 (extended)
# (Stub for Person B)

import numpy as np
from env import load_environment


class TabularQAgent:
    def __init__(self, obs_bins, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.obs_bins = obs_bins  # List of bins for each obs dim
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros(obs_bins + [action_size])

    def discretize(self, obs, obs_low, obs_high):
        ratios = (obs - obs_low) / (obs_high - obs_low)
        ratios = np.clip(ratios, 0, 0.999)
        return tuple((ratios * np.array(self.obs_bins)).astype(int))

    def select_action(self, state_disc):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state_disc])

    def update(self, state_disc, action, reward, next_state_disc, done):
        best_next = np.max(self.q_table[next_state_disc])
        td_target = reward + self.gamma * best_next * (1 - done)
        td_error = td_target - self.q_table[state_disc + (action,)]
        self.q_table[state_disc + (action,)] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class SarsaAgent:
    # TODO: Implement SARSA
    pass


if __name__ == "__main__":
    env = load_environment("MyCustomEnv")
    obs, _ = env.reset()
    obs = np.asarray(obs)
    obs_low = np.min(obs) * np.ones_like(obs)
    obs_high = np.max(obs) * np.ones_like(obs)
    obs_bins = [10] * len(obs)  # 10 bins per obs dim
    action_size = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    agent = TabularQAgent(obs_bins, action_size)
    episodes = 200
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs)
        state_disc = agent.discretize(obs, obs_low, obs_high)
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state_disc)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = np.asarray(next_obs)
            next_state_disc = agent.discretize(next_obs, obs_low, obs_high)
            done = terminated or truncated
            agent.update(state_disc, action, reward, next_state_disc, done)
            state_disc = next_state_disc
            total_reward += reward
        agent.decay_epsilon()
        print(f"Episode {ep+1}: Reward {total_reward}")
