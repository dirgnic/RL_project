"""\
Analyze training results for all agents.

Loads the training CSVs for DQN, REINFORCE, and Tabular Q-Learning and prints
summary statistics (mean, last-100 mean, best-100 mean). Also writes a summary
CSV under results/comparison/.

Usage:
    python analyze_results.py
"""

import os
import numpy as np
import pandas as pd


AGENT_FILES = {
    "dqn": "results/dqn_training.csv",
    "reinforce": "results/reinforce_training.csv",
    "tabular": "results/tabular_training.csv",
}


def summarize_agent(name: str, path: str):
    if not os.path.exists(path):
        print(f"✗ {name}: {path} not found")
        return None

    df = pd.read_csv(path)
    rewards = df["reward"].to_numpy(dtype=float)
    mean_all = float(rewards.mean())
    mean_last_100 = float(rewards[-100:].mean()) if rewards.size >= 100 else float(rewards.mean())
    # Best moving window of 100 episodes
    if rewards.size >= 100:
        windows = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        best_100 = float(windows.max())
    else:
        best_100 = float(rewards.mean())

    print(
        f"{name.upper():9s} | episodes={len(rewards):4d} | "
        f"mean={mean_all:6.2f} | last100={mean_last_100:6.2f} | best100={best_100:6.2f}"
    )

    return {
        "agent": name,
        "episodes": len(rewards),
        "mean_reward": mean_all,
        "last100_mean": mean_last_100,
        "best100_mean": best_100,
    }


def main():
    os.makedirs("results/comparison", exist_ok=True)

    rows = []
    for name, path in AGENT_FILES.items():
        stats = summarize_agent(name, path)
        if stats is not None:
            rows.append(stats)

    if not rows:
        print("No training result files found; nothing to summarize.")
        return

    df = pd.DataFrame(rows)
    out_path = "results/comparison/training_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Training summary saved to {out_path}")


if __name__ == "__main__":
    main()
