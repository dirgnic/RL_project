"""
Compare Q-Learning results across different environments and reward types.
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.plotting import plot_comparison, create_comparison_table


def main():
    """Compare Q-Learning results."""
    
    # Define log files to compare
    log_dir = project_root / "results" / "q_learning" / "logs"
    
    # Find all Q-Learning log files
    log_files = list(log_dir.glob("q_learning_*.csv"))
    
    if not log_files:
        print("No Q-Learning log files found!")
        print(f"Looking in: {log_dir}")
        return
    
    # Sort by modification time (most recent first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"Found {len(log_files)} Q-Learning experiments:")
    for i, log_file in enumerate(log_files):
        print(f"  {i+1}. {log_file.name}")
    
    # Group by environment
    cartpole_logs = [f for f in log_files if 'cartpole' in f.name.lower()]
    mountaincar_logs = [f for f in log_files if 'mountaincar' in f.name.lower()]
    lunarlander_logs = [f for f in log_files if 'lunarlander' in f.name.lower()]
    
    plot_dir = project_root / "results" / "q_learning" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare CartPole
    if len(cartpole_logs) >= 2:
        print("\n--- Comparing CartPole experiments ---")
        labels = [f.stem.replace('q_learning_cartpole_', '').split('_20')[0] for f in cartpole_logs[:2]]
        plot_comparison(
            cartpole_logs[:2],
            labels,
            plot_dir / "comparison_cartpole.png",
            title="Q-Learning on CartPole: Default vs Modified Reward"
        )
    
    # Compare MountainCar
    if len(mountaincar_logs) >= 2:
        print("\n--- Comparing MountainCar experiments ---")
        labels = [f.stem.replace('q_learning_mountaincar_', '').split('_20')[0] for f in mountaincar_logs[:2]]
        plot_comparison(
            mountaincar_logs[:2],
            labels,
            plot_dir / "comparison_mountaincar.png",
            title="Q-Learning on MountainCar: Default vs Modified Reward"
        )
    
    # Compare LunarLander
    if len(lunarlander_logs) >= 2:
        print("\n--- Comparing LunarLander experiments ---")
        labels = [f.stem.replace('q_learning_lunarlander_', '').split('_20')[0] for f in lunarlander_logs[:2]]
        plot_comparison(
            lunarlander_logs[:2],
            labels,
            plot_dir / "comparison_lunarlander.png",
            title="Q-Learning on LunarLander: Default vs Modified Reward"
        )
    
    # Create overall comparison table
    print("\n--- Creating comparison table ---")
    table_dir = project_root / "results" / "comparison" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    
    if len(log_files) > 0:
        labels = [f.stem.replace('q_learning_', '').split('_20')[0] for f in log_files]
        create_comparison_table(
            log_files,
            labels,
            table_dir / "q_learning_comparison.csv"
        )
    
    print("\n" + "=" * 60)
    print("Comparison completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
