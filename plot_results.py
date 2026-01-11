"""
Plot Training Results
=====================
Simple script to visualize training curves from all agents.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_results():
    """Load and plot all training results."""
    
    plt.figure(figsize=(12, 5))
    
    # Load available results
    agents = {
        'DQN': 'results/dqn_training.csv',
        'Tabular Q-Learning': 'results/tabular_training.csv',
        'REINFORCE': 'results/reinforce_training.csv'
    }
    
    colors = {'DQN': 'blue', 'Tabular Q-Learning': 'green', 'REINFORCE': 'red'}
    
    for name, path in agents.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Plot raw rewards (faded)
            plt.plot(df['episode'], df['reward'], alpha=0.2, color=colors[name])
            # Plot smoothed rewards
            smoothed = df['reward'].rolling(window=50, min_periods=1).mean()
            plt.plot(df['episode'], smoothed, label=name, linewidth=2, color=colors[name])
            print(f"✓ Loaded {name}: {len(df)} episodes, final avg: {df['reward'].tail(50).mean():.1f}")
        else:
            print(f"✗ {name} results not found")
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to results/training_comparison.png")
    # plt.show()


if __name__ == "__main__":
    plot_results()
