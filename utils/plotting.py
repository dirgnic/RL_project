"""
Plotting utilities for RL experiments.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_learning_curve(log_file, output_file, title="Learning Curve", window=100):
    """
    Plot learning curve from log file.
    
    Args:
        log_file: Path to CSV log file
        output_file: Path to save plot
        title: Plot title
        window: Moving average window size
    """
    # Read log file
    df = pd.read_csv(log_file)
    
    episodes = df['episode'].values
    rewards = df['reward'].values
    
    # Calculate moving average
    moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot raw rewards
    ax.plot(episodes, rewards, alpha=0.3, label='Episode Reward', color='blue')
    
    # Plot moving average
    ax.plot(episodes, moving_avg, label=f'Moving Average (window={window})', 
            color='red', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Learning curve saved to: {output_file}")


def plot_comparison(log_files, labels, output_file, title="Algorithm Comparison", window=100):
    """
    Plot comparison of multiple algorithms.
    
    Args:
        log_files: List of paths to CSV log files
        labels: List of algorithm labels
        output_file: Path to save plot
        title: Plot title
        window: Moving average window size
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        df = pd.read_csv(log_file)
        episodes = df['episode'].values
        rewards = df['reward'].values
        
        # Calculate moving average
        moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
        
        color = colors[i % len(colors)]
        ax.plot(episodes, moving_avg, label=label, color=color, linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Comparison plot saved to: {output_file}")


def plot_multiple_metrics(log_file, output_file, title="Training Metrics"):
    """
    Plot multiple metrics from log file.
    
    Args:
        log_file: Path to CSV log file
        output_file: Path to save plot
        title: Plot title
    """
    # Read log file
    df = pd.read_csv(log_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Rewards
    axes[0, 0].plot(df['episode'], df['reward'], alpha=0.5)
    moving_avg = pd.Series(df['reward']).rolling(window=100, min_periods=1).mean()
    axes[0, 0].plot(df['episode'], moving_avg, color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Steps
    axes[0, 1].plot(df['episode'], df['steps'], alpha=0.5, color='green')
    moving_avg_steps = pd.Series(df['steps']).rolling(window=100, min_periods=1).mean()
    axes[0, 1].plot(df['episode'], moving_avg_steps, color='darkgreen', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon
    axes[1, 0].plot(df['episode'], df['epsilon'], color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate (Epsilon)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution
    axes[1, 1].hist(df['reward'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(df['reward'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["reward"].mean():.2f}')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Multi-metric plot saved to: {output_file}")


def create_comparison_table(log_files, labels, output_file):
    """
    Create comparison table of algorithm performance.
    
    Args:
        log_files: List of paths to CSV log files
        labels: List of algorithm labels
        output_file: Path to save table (CSV)
    """
    results = []
    
    for log_file, label in zip(log_files, labels):
        df = pd.read_csv(log_file)
        
        # Calculate statistics
        last_100 = df['reward'].tail(100)
        
        result = {
            'Algorithm': label,
            'Mean Reward (Last 100)': last_100.mean(),
            'Std Reward (Last 100)': last_100.std(),
            'Max Reward': df['reward'].max(),
            'Min Reward': df['reward'].min(),
            'Mean Steps': df['steps'].mean(),
            'Total Episodes': len(df)
        }
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    
    print(f"Comparison table saved to: {output_file}")
    print("\n" + "=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    
    return results_df
