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

def evaluate_policy_agent(agent, env, episodes=10):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = agent.select_action(obs)
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

action_size = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
tabular_agent = TabularQAgent([10]*len(env.reset()[0]), action_size)
dqn_agent = DQNAgent(state_size=len(env.reset()[0]), action_size=action_size)
policy_agent = REINFORCEAgent(len(env.reset()[0]), action_size)


EVAL_EPISODES = 20
results = {}
results['TabularQ'] = evaluate_tabular_agent(tabular_agent, env, episodes=EVAL_EPISODES)
results['DQN'] = evaluate_dqn_agent(dqn_agent, env, episodes=EVAL_EPISODES)
results['REINFORCE'] = evaluate_policy_agent(policy_agent, env, episodes=EVAL_EPISODES)

print(f"TabularQ type: {type(results['TabularQ'])}, len: {len(results['TabularQ']) if hasattr(results['TabularQ'], '__len__') else 'N/A'}")
print(f"DQN type: {type(results['DQN'])}, len: {len(results['DQN']) if hasattr(results['DQN'], '__len__') else 'N/A'}")
print(f"REINFORCE type: {type(results['REINFORCE'])}, len: {len(results['REINFORCE']) if hasattr(results['REINFORCE'], '__len__') else 'N/A'}")

# Ensure all results are the same length for DataFrame
min_len = min(len(results['TabularQ']), len(results['DQN']), len(results['REINFORCE']))
results = {k: v[:min_len] for k, v in results.items()}

results_df = pd.DataFrame(results)
results_df.to_csv('evaluation_results.csv', index=False)
print("Evaluation complete. Results saved to evaluation_results.csv.")
