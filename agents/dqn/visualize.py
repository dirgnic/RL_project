"""
Visualization and Plotting Utilities for DQN
Person C (Ingrid)

Create publication-quality plots for experiments and comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'


class DQNVisualizer:
    """Visualization utilities for DQN experiments"""
    
    def __init__(self, output_dir: str = './plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Visualizer initialized. Plots will be saved to: {output_dir}")
    
    def plot_training_curves(
        self,
        results: Dict,
        title: str = "DQN Training Progress",
        filename: str = "training_curves.png"
    ):
        """Plot training curves for single experiment"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Rewards
        ax = axes[0]
        for result in results['individual_results']:
            seed = result['seed']
            rewards = result['rewards']
            
            # Raw
            ax.plot(rewards, alpha=0.2, color='blue')
            
            # Smoothed
            window = 100
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            ax.plot(smoothed, alpha=0.8, label=f'Seed {seed}', linewidth=2)
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final evaluation comparison
        ax = axes[1]
        seeds = [r['seed'] for r in results['individual_results']]
        final_rewards = [r['final_reward'] for r in results['individual_results']]
        final_stds = [r['final_std'] for r in results['individual_results']]
        
        x_pos = np.arange(len(seeds))
        ax.bar(x_pos, final_rewards, yerr=final_stds, capsize=5, alpha=0.7, color='green')
        ax.axhline(y=np.mean(final_rewards), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(final_rewards):.2f}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Seed {s}' for s in seeds])
        ax.set_ylabel('Evaluation Reward (100 eps)', fontsize=12, fontweight='bold')
        ax.set_title('Final Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath}")
        plt.close()
    
    def plot_comparison(
        self,
        experiments: List[Dict],
        title: str = "Configuration Comparison",
        filename: str = "comparison.png"
    ):
        """Compare multiple experiments"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar plot of mean rewards
        ax = axes[0]
        exp_names = [exp['experiment_name'] for exp in experiments]
        mean_rewards = [exp['mean_final_reward'] for exp in experiments]
        std_rewards = [exp['std_final_reward'] for exp in experiments]
        
        x_pos = np.arange(len(exp_names))
        colors = sns.color_palette("husl", len(exp_names))
        
        bars = ax.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, 
                      alpha=0.7, color=colors)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, mean_rewards, std_rewards)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.1f}±{std:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Evaluation Reward', fontsize=12, fontweight='bold')
        ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Training curves comparison
        ax = axes[1]
        for exp, color in zip(experiments, colors):
            for result in exp['individual_results']:
                rewards = result['rewards']
                
                # Smooth
                window = 100
                smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                ax.plot(smoothed, alpha=0.3, color=color)
            
            # Plot mean of all seeds
            all_rewards = np.array([r['rewards'] for r in exp['individual_results']])
            mean_curve = np.mean(all_rewards, axis=0)
            smoothed_mean = pd.Series(mean_curve).rolling(window=100, min_periods=1).mean()
            ax.plot(smoothed_mean, linewidth=3, color=color, label=exp['experiment_name'])
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward (smoothed)', fontsize=12, fontweight='bold')
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath}")
        plt.close()
    
    def plot_dqn_vs_qlearning(
        self,
        dqn_results: Dict,
        qlearning_results: Dict,
        title: str = "DQN vs Q-Learning Comparison",
        filename: str = "dqn_vs_qlearning.png"
    ):
        """
        Compare DQN with Q-Learning (tabular)
        
        Args:
            dqn_results: DQN experiment results dict
            qlearning_results: Q-Learning results dict (from Irina)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training curves
        ax = axes[0]
        
        # DQN
        dqn_rewards_all = np.array([r['rewards'] for r in dqn_results['individual_results']])
        dqn_mean = np.mean(dqn_rewards_all, axis=0)
        dqn_std = np.std(dqn_rewards_all, axis=0)
        
        episodes_dqn = np.arange(len(dqn_mean))
        ax.plot(episodes_dqn, dqn_mean, linewidth=3, label='DQN (Ingrid)', color='blue')
        ax.fill_between(episodes_dqn, dqn_mean - dqn_std, dqn_mean + dqn_std, 
                        alpha=0.2, color='blue')
        
        # Q-Learning (if available)
        if 'rewards' in qlearning_results:
            ql_rewards = qlearning_results['rewards']
            episodes_ql = np.arange(len(ql_rewards))
            ax.plot(episodes_ql, ql_rewards, linewidth=3, label='Q-Learning (Irina)', color='orange')
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Final performance comparison
        ax = axes[1]
        
        algorithms = ['Q-Learning\n(Tabular)', 'DQN\n(Deep)']
        final_rewards = [
            qlearning_results.get('final_reward', 0),
            dqn_results['mean_final_reward']
        ]
        final_stds = [
            qlearning_results.get('final_std', 0),
            dqn_results['std_final_reward']
        ]
        
        colors_alg = ['orange', 'blue']
        bars = ax.bar(algorithms, final_rewards, yerr=final_stds, capsize=10,
                      alpha=0.7, color=colors_alg, width=0.5)
        
        # Add value labels
        for bar, reward, std in zip(bars, final_rewards, final_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{reward:.1f}±{std:.1f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Evaluation Reward (100 eps)', fontsize=12, fontweight='bold')
        ax.set_title('Final Performance', fontsize=14, fontweight='bold')
        ax.set_ylim(bottom=min(final_rewards) - max(final_stds) - 2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filepath}")
        plt.close()
    
    def create_summary_table(
        self,
        experiments: List[Dict],
        filename: str = "results_table.csv"
    ):
        """Create CSV table with all results"""
        data = []
        
        for exp in experiments:
            data.append({
                'Experiment': exp['experiment_name'],
                'Mean Reward': exp['mean_final_reward'],
                'Std Reward': exp['std_final_reward'],
                'Best Seed Reward': max(r['final_reward'] for r in exp['individual_results']),
                'Worst Seed Reward': min(r['final_reward'] for r in exp['individual_results']),
                'Learning Rate': exp['config'].get('learning_rate', 'N/A'),
                'Gamma': exp['config'].get('gamma', 'N/A'),
                'Buffer Size': exp['config'].get('buffer_capacity', 'N/A'),
                'Double DQN': exp['config'].get('use_double_dqn', False)
            })
        
        df = pd.DataFrame(data)
        filepath = f"{self.output_dir}/{filename}"
        df.to_csv(filepath, index=False, float_format='%.2f')
        print(f"[OK] Saved table: {filepath}")
        
        # Also print to console
        print("\n" + "="*80)
        print("[INFO] RESULTS TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        return df


def load_experiment_results(results_dir: str) -> Dict:
    """Load experiment results from JSON file"""
    results_path = Path(results_dir) / "results.json"
    
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"[OK] Loaded results from: {results_path}")
    return results


def visualize_all_experiments(experiments_dir: str = './experiments'):
    """Load and visualize all experiments in directory"""
    viz = DQNVisualizer(f'{experiments_dir}/plots')
    
    # Find all experiment directories
    exp_dirs = [d for d in Path(experiments_dir).iterdir() if d.is_dir() and d.name != 'plots']
    
    if not exp_dirs:
        print("[ERROR] No experiment directories found!")
        return
    
    print(f"\n[INFO] Found {len(exp_dirs)} experiment directories\n")
    
    all_experiments = []
    for exp_dir in exp_dirs:
        results = load_experiment_results(str(exp_dir))
        if results:
            all_experiments.append(results)
            viz.plot_training_curves(
                results,
                title=f"Training: {results['experiment_name']}",
                filename=f"training_{results['experiment_name']}.png"
            )
    
    # Comparison plot
    if len(all_experiments) > 1:
        viz.plot_comparison(
            all_experiments,
            title="All Experiments Comparison",
            filename="all_experiments_comparison.png"
        )
        
        viz.create_summary_table(
            all_experiments,
            filename="all_results_summary.csv"
        )
    
    print("\n[OK] All visualizations complete!")
    print(f"[INFO] Check plots in: {viz.output_dir}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("[INFO] DQN VISUALIZATION UTILITIES")
    print("Person C (Ingrid)")
    print("="*60 + "\n")
    
    # Example usage
    print("Usage:")
    print("  from agents.dqn.visualize import DQNVisualizer, visualize_all_experiments")
    print("  visualize_all_experiments('./experiments')")
    print("\nOr run: python agents/dqn/visualize.py")
    print("to visualize all experiments in ./experiments directory\n")
    
    # Try to visualize if experiments directory exists
    if Path('./experiments').exists():
        visualize_all_experiments('./experiments')
    else:
        print("[TIP] Run experiments first using experiments.py!")
