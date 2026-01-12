"""
Evaluate trained agents on the same environment and compare performance.

Runs multiple evaluation episodes for each agent type (DQN, REINFORCE, Tabular)
with deterministic (epsilon=0) policies and saves a CSV + bar plot with average
returns under results/comparison/.

Usage:
    python evaluate_agents.py --episodes 20
"""

import argparse
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env import load_environment, DiscreteActionWrapper, FlatObsWrapper, TabularObsWrapper
from agents.dqn.agent import DQNAgent
from train_reinforce import REINFORCEAgent
from train_tabular import TabularQAgent


def build_env_for_agent(agent_type: str, seed: Optional[int] = None):
    """Create environment (with wrappers) for a given agent type."""
    raw_env = load_environment("MyCustomEnv")
    if seed is not None:
        try:
            raw_env.reset(seed=seed)
        except TypeError:
            # Older gym/highway versions: ignore explicit seeding here
            pass
    env = DiscreteActionWrapper(raw_env)
    env = FlatObsWrapper(env)
    if agent_type == "tabular":
        env = TabularObsWrapper(env, grid_bins=6, fuel_bins=3)
    return raw_env, env


def load_agent(agent_type: str, env) -> Optional[object]:
    """Instantiate and load the trained agent for evaluation.

    Returns None if the model file is missing, in which case a random policy is used.
    """
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if agent_type == "dqn":
        agent = DQNAgent(state_size=obs_dim, action_size=action_dim, hidden_dims=[256, 128])
        try:
            agent.load("results/dqn_model.pth")
            agent.epsilon = 0.0
            return agent
        except FileNotFoundError:
            print("[WARN] DQN model not found, using random policy for DQN.")
            return None

    if agent_type == "reinforce":
        agent = REINFORCEAgent(obs_dim, action_dim)
        try:
            agent.load("results/reinforce_model.pth")
            return agent
        except FileNotFoundError:
            print("[WARN] REINFORCE model not found, using random policy for REINFORCE.")
            return None

    if agent_type == "tabular":
        obs_bins = [6, 6, 3, 2, 3, 3]
        agent = TabularQAgent(obs_bins, action_dim)
        try:
            agent.load("results/tabular_model.npy")
            agent.epsilon = 0.0
            return agent
        except FileNotFoundError:
            print("[WARN] Tabular model not found, using random policy for Tabular.")
            return None

    return None


def run_episode(agent_type: str, agent, raw_env, env, max_steps: int = 500) -> float:
    """Run a single evaluation episode and return total reward."""
    obs, _ = env.reset()
    if agent_type == "tabular" and agent is not None:
        state = agent.discretize(obs)
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        if agent is None:
            action = env.action_space.sample()
        elif agent_type == "dqn":
            action = agent.select_action(obs, training=False)
        elif agent_type == "reinforce":
            action, _ = agent.select_action(obs)
        elif agent_type == "tabular":
            action = agent.select_action(state)
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs = next_obs
        if agent_type == "tabular" and agent is not None:
            state = agent.discretize(next_obs)

        total_reward += float(reward)
        done = terminated or truncated
        steps += 1

    return total_reward


def evaluate_agent(agent_type: str, episodes: int, max_steps: int = 500, base_seed: int = 0) -> Tuple[float, float, List[float]]:
    """Evaluate one agent over multiple episodes, returning mean, std, and all returns."""
    returns: List[float] = []

    for i in range(episodes):
        seed = base_seed + i
        raw_env, env = build_env_for_agent(agent_type, seed=seed)
        agent = load_agent(agent_type, env)
        episode_return = run_episode(agent_type, agent, raw_env, env, max_steps=max_steps)
        returns.append(episode_return)

    returns_arr = np.array(returns, dtype=float)
    return float(returns_arr.mean()), float(returns_arr.std()), returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per agent")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = parser.parse_args()

    agents = ["dqn", "reinforce", "tabular"]
    os.makedirs("results/comparison", exist_ok=True)

    rows = []
    for agent_type in agents:
        print(f"Evaluating {agent_type.upper()} over {args.episodes} episodes...")
        mean_ret, std_ret, ret_list = evaluate_agent(
            agent_type, episodes=args.episodes, max_steps=args.max_steps, base_seed=args.seed
        )
        print(f"  -> mean: {mean_ret:.2f}, std: {std_ret:.2f}")
        rows.append({
            "agent": agent_type,
            "mean_return": mean_ret,
            "std_return": std_ret,
        })

        # Save per-episode returns for this agent
        df_agent = pd.DataFrame({"episode": range(1, args.episodes + 1), "return": ret_list})
        df_agent.to_csv(f"results/comparison/eval_{agent_type}.csv", index=False)

    # Save summary table
    df_summary = pd.DataFrame(rows)
    summary_path = "results/comparison/eval_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n✓ Evaluation summary saved to {summary_path}")

    # Bar plot of mean returns
    plt.figure(figsize=(6, 4))
    plt.bar(df_summary["agent"], df_summary["mean_return"], yerr=df_summary["std_return"], capsize=5)
    plt.ylabel("Average return")
    plt.title("Agent Evaluation: Mean Return ± Std")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_plot = "results/comparison/eval_barplot.png"
    plt.savefig(out_plot, dpi=150)
    print(f"✓ Evaluation bar plot saved to {out_plot}")


if __name__ == "__main__":
    main()
