"""
Visualize Agent Actions on the Map
==================================
Runs one episode with a chosen policy (DQN / REINFORCE / Tabular / random),
logs the ego vehicle coordinates and chosen actions, and produces a scatter
plot showing which action was taken at which point on the road.

Usage examples:
    python visualize_actions.py --agent dqn
    python visualize_actions.py --agent reinforce
    python visualize_actions.py --agent tabular
    python visualize_actions.py --agent random

Output:
    Saves a PNG under results/actions_<agent>.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from env import load_environment, DiscreteActionWrapper, FlatObsWrapper, TabularObsWrapper
from agents.dqn.agent import DQNAgent
from train_reinforce import REINFORCEAgent
from train_tabular import TabularQAgent


ACTION_LABELS = {
    0: "Idle",
    1: "Accelerate",
    2: "Brake",
    3: "Left",
    4: "Right",
}


def build_env_for_agent(agent_type: str):
    """Create env and (optionally) extra wrapper for tabular."""
    raw_env = load_environment("MyCustomEnv", render_mode=None)
    env = DiscreteActionWrapper(raw_env)
    env = FlatObsWrapper(env)

    # For tabular, add tabular wrapper on top
    if agent_type == "tabular":
        env = TabularObsWrapper(env, grid_bins=6, fuel_bins=3)

    return raw_env, env


def load_agent(agent_type: str, env):
    """Instantiate and load the agent for the given type.

    If weights are missing, returns None and the caller can fall back to random.
    """
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if agent_type == "dqn":
        agent = DQNAgent(state_size=obs_dim, action_size=action_dim, hidden_dims=[256, 128])
        try:
            agent.load("results/dqn_model.pth")
            agent.epsilon = 0.0  # deterministic for visualization
            return agent
        except FileNotFoundError:
            print("[WARN] DQN model not found at results/dqn_model.pth, using random policy.")
            return None

    if agent_type == "reinforce":
        agent = REINFORCEAgent(obs_dim, action_dim)
        try:
            agent.load("results/reinforce_model.pth")
            return agent
        except FileNotFoundError:
            print("[WARN] REINFORCE model not found at results/reinforce_model.pth, using random policy.")
            return None

    if agent_type == "tabular":
        obs_bins = [6, 6, 3, 2, 3, 3]
        agent = TabularQAgent(obs_bins, action_dim)
        try:
            agent.load("results/tabular_model.npy")
            agent.epsilon = 0.0
            return agent
        except FileNotFoundError:
            print("[WARN] Tabular model not found at results/tabular_model.npy, using random policy.")
            return None

    # random agent
    return None


def collect_trajectory(agent_type: str, raw_env, env, agent, max_steps: int = 500):
    """Run one episode and collect (x, y, action) for ego vehicle."""
    positions = []
    actions = []

    obs, _ = env.reset()
    if agent_type == "tabular" and agent is not None:
        state = agent.discretize(obs)
    done = False

    steps = 0
    while not done and steps < max_steps:
        # Choose action
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

        # Record ego position from base env
        ego_pos = np.array(raw_env.vehicle.position, dtype=float)
        positions.append(ego_pos)
        actions.append(action)

        # Step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs = next_obs
        if agent_type == "tabular" and agent is not None:
            state = agent.discretize(next_obs)

        done = terminated or truncated
        steps += 1

    if len(positions) == 0:
        return np.zeros((0, 2)), np.array([])

    return np.vstack(positions), np.array(actions)


def plot_actions(agent_type: str, positions: np.ndarray, actions: np.ndarray):
    """Create a scatter plot of positions colored by discrete action."""
    if positions.shape[0] == 0:
        print("[WARN] No positions collected, nothing to plot.")
        return

    x = positions[:, 0]
    y = positions[:, 1]

    plt.figure(figsize=(8, 6))

    # Use a discrete colormap for up to 5 actions
    cmap = plt.get_cmap("tab10")

    for a in sorted(np.unique(actions)):
        mask = actions == a
        label = ACTION_LABELS.get(int(a), f"Action {int(a)}")
        plt.scatter(x[mask], y[mask], s=15, color=cmap(int(a) % 10), label=label, alpha=0.8)

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(f"Ego Actions over Map ({agent_type.upper()} policy)")
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"actions_{agent_type}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved action map to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "reinforce", "tabular", "random"], default="dqn")
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    raw_env, env = build_env_for_agent(args.agent)
    agent = load_agent(args.agent, env)

    positions, actions = collect_trajectory(args.agent, raw_env, env, agent, max_steps=args.max_steps)
    plot_actions(args.agent, positions, actions)


if __name__ == "__main__":
    main()
