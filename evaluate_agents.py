
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
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return rewards

 env = MyCustomEnv(render_mode=None)

# Load agents (assume trained and saved, or use fresh for demo)
tabular_agent = TabularQAgent([10]*len(env.reset()[0]), env.action_space.n)
dqn_agent = DQNAgent(state_size=len(env.reset()[0]), action_size=env.action_space.n)
policy_agent = REINFORCEAgent(len(env.reset()[0]), env.action_space.n)

results = {}
results['TabularQ'] = evaluate_tabular_agent(tabular_agent, env)
results['DQN'] = evaluate_dqn_agent(dqn_agent, env)
results['REINFORCE'] = evaluate_policy_agent(policy_agent, env)

env.close()

results_df = pd.DataFrame(results)
results_df.to_csv('evaluation_results.csv', index=False)
print("Evaluation complete. Results saved to evaluation_results.csv.")
