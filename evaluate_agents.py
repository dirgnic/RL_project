import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from custom_env import MyCustomEnv
from agents.tabular.tabular_agents import TabularQAgent
from agents.dqn.agent import DQNAgent
from agents.policy.reinforce import REINFORCEAgent
import numpy as np

def evaluate_tabular_agent(agent, env, episodes=10):
    obs_low = np.zeros(agent.q_table.ndim - 1)
    obs_high = np.ones(agent.q_table.ndim - 1) * 10
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        state_disc = agent.discretize(obs, obs_low, obs_high)
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state_disc)
            # Ensure action is a vector for continuous action space
            if not isinstance(action, (list, np.ndarray)):
                action = [action] * (env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state_disc = agent.discretize(next_obs, obs_low, obs_high)
            done = terminated or truncated
            state_disc = next_state_disc
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def evaluate_dqn_agent(agent, env, episodes=10):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(obs, training=False)
            # Ensure action is a vector for continuous action space
            if not isinstance(action, (list, np.ndarray)):
                action = [action] * (env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return rewards

env = MyCustomEnv(render_mode=None)

# Load agents (assume trained and saved, or use fresh for demo)

# For MyCustomEnv, use 5 discrete actions for REINFORCE (see mapping in evaluate_policy_agent)
action_size = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
tabular_agent = TabularQAgent([10]*len(env.reset()[0]), action_size)
dqn_agent = DQNAgent(state_size=len(env.reset()[0]), action_size=action_size)
obs_dim = np.prod(env.reset()[0].shape)

action_dim = 5  # 5 discrete actions for mapping
policy_agent = REINFORCEAgent(obs_dim, action_dim)


def evaluate_policy_agent(agent, env, episodes=10):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()

        done = False
        total_reward = 0
        while not done:
            flat_obs = np.array(obs).flatten()
            action, _ = agent.select_action(flat_obs)
            # For MyCustomEnv, map discrete action to [acceleration, steering]
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'shape') and env.action_space.shape == (2,):
                # Example: 5 discrete actions mapped to 2D continuous
                # 0: [0.1, 0.0] (accelerate)
                # 1: [-0.05, 0.0] (brake)
                # 2: [0.0, -0.5] (left)
                # 3: [0.0, 0.5] (right)
                # 4: [0.0, 0.0] (no-op)
                action_map = [
                    [0.1, 0.0],    # accelerate
                    [-0.05, 0.0],  # brake
                    [0.0, -0.5],   # left
                    [0.0, 0.5],    # right
                    [0.0, 0.0]     # no-op
                ]
                action = action_map[int(action) % len(action_map)]
            elif not isinstance(action, (list, np.ndarray)):
                action = [action] * (env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return rewards

EVAL_EPISODES = 20
results = {}
results['TabularQ'] = evaluate_tabular_agent(tabular_agent, env, episodes=EVAL_EPISODES)
results['DQN'] = evaluate_dqn_agent(dqn_agent, env, episodes=EVAL_EPISODES)
results['REINFORCE'] = evaluate_policy_agent(policy_agent, env, episodes=EVAL_EPISODES)



# Ensure all results are the same length for DataFrame
min_len = min(len(results['TabularQ']), len(results['DQN']), len(results['REINFORCE']))
results = {k: v[:min_len] for k, v in results.items()}

results_df = pd.DataFrame(results)
results_df.to_csv('evaluation_results.csv', index=False)

