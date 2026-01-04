"""
Hyperparameter Experiments and Comparison Scripts
Person C (Ingrid)

Run different DQN configurations and compare results
"""

import gymnasium as gym
import numpy as np
import torch
import json
import os
from typing import List, Dict
from datetime import datetime
from agents.dqn.agent import DQNAgent


class DQNExperimentRunner:
    """Run multiple DQN experiments with different configurations"""
    
    def __init__(self, base_output_dir: str = './experiments'):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"Experiment Runner Initialized")
        print(f"Output directory: {base_output_dir}\n")
    
    def run_experiment(
        self,
        experiment_name: str,
        config: Dict,
        num_seeds: int = 3,
        num_episodes: int = 2000
    ) -> Dict:
        """
        Run single experiment with multiple seeds
        
        Args:
            experiment_name: Name for this experiment
            config: DQN configuration dict
            num_seeds: Number of random seeds to try
            num_episodes: Episodes per seed
            
        Returns:
            results: Dict with all results
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*60}")
        print(f"Config: {json.dumps(config, indent=2)}")
        print(f"Seeds: {num_seeds}, Episodes: {num_episodes}\n")
        
        exp_dir = f"{self.base_output_dir}/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)
        
        all_results = []
        seeds = [42 + i * 111 for i in range(num_seeds)]
        
        for seed_idx, seed in enumerate(seeds):
            print(f"Seed {seed} ({seed_idx+1}/{num_seeds})")
            result = self._train_single(seed, config, num_episodes)
            all_results.append(result)
            print(f"   Final Reward: {result['final_reward']:.2f}\n")
        
        # Aggregate results
        aggregated = {
            'experiment_name': experiment_name,
            'config': config,
            'seeds': seeds,
            'individual_results': all_results,
            'mean_final_reward': np.mean([r['final_reward'] for r in all_results]),
            'std_final_reward': np.std([r['final_reward'] for r in all_results]),
            'mean_best_reward': np.mean([r['best_reward'] for r in all_results]),
            'std_best_reward': np.std([r['best_reward'] for r in all_results])
        }
        
        # Save results
        with open(f"{exp_dir}/results.json", 'w') as f:
            json.dump(aggregated, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"Experiment Complete!")
        print(f"   Mean Final Reward: {aggregated['mean_final_reward']:.2f} +/- {aggregated['std_final_reward']:.2f}")
        print(f"   Mean Best Reward: {aggregated['mean_best_reward']:.2f} +/- {aggregated['std_best_reward']:.2f}")
        print(f"   Results saved to: {exp_dir}\n")
        
        return aggregated
    
    def _train_single(self, seed: int, config: Dict, num_episodes: int) -> Dict:
        """Train with single seed"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = gym.make('Taxi-v3')
        env.reset(seed=seed)
        
        # Remove keys not accepted by DQNAgent
        agent_config = dict(config)
        agent_config.pop('max_steps', None)
        
        # Use TaxiDQNNetwork - designed specifically for Taxi-v3
        # It decodes state integers into meaningful features (taxi pos, passenger, dest)
        from agents.dqn.network import TaxiDQNNetwork
        agent = DQNAgent(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            hidden_dims=[64, 64],
            **agent_config
        )
        # Replace networks with TaxiDQNNetwork for better performance
        agent.policy_net = TaxiDQNNetwork(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            hidden_dims=[64, 64]
        ).to(agent.device)
        agent.target_net = TaxiDQNNetwork(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            hidden_dims=[64, 64]
        ).to(agent.device)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        # Re-initialize optimizer with new network parameters
        agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=config.get('learning_rate', 1e-3))
        
        rewards = []
        best_reward = -float('inf')
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for _ in range(config.get('max_steps', 200)):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, terminated or truncated)
                agent.train_step()
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            agent.decay_epsilon()
            rewards.append(episode_reward)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
        
        env.close()
        
        # Final evaluation
        env = gym.make('Taxi-v3')
        eval_rewards = []
        for _ in range(100):
            state, _ = env.reset()
            ep_reward = 0
            for _ in range(200):
                action = agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            eval_rewards.append(ep_reward)
        env.close()
        
        return {
            'seed': seed,
            'rewards': rewards,
            'best_reward': best_reward,
            'final_reward': np.mean(eval_rewards),
            'final_std': np.std(eval_rewards)
        }
    
    def compare_configurations(self, configs: List[Dict]) -> Dict:
        """
        Compare multiple configurations
        
        Args:
            configs: List of (name, config) tuples
            
        Returns:
            comparison_results: Dict with comparison metrics
        """
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER COMPARISON")
        print(f"{'='*60}\n")
        
        all_experiments = []
        
        for name, config in configs:
            result = self.run_experiment(name, config, num_seeds=3, num_episodes=1500)
            all_experiments.append(result)
        
        # Print comparison table
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}\n")
        print(f"{'Experiment':<30} {'Mean Reward':<15} {'Best Reward':<15}")
        print("-" * 60)
        
        for exp in all_experiments:
            print(f"{exp['experiment_name']:<30} "
                  f"{exp['mean_final_reward']:>6.2f} ± {exp['std_final_reward']:<5.2f} "
                  f"{exp['mean_best_reward']:>6.2f} ± {exp['std_best_reward']:<5.2f}")
        
        print("=" * 60 + "\n")
        
        return {
            'experiments': all_experiments,
            'best_config': max(all_experiments, key=lambda x: x['mean_final_reward'])
        }


def run_baseline_experiment():
    """Run baseline DQN experiment"""
    runner = DQNExperimentRunner('./experiments_baseline')
    
    config = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_freq': 100,
        'use_double_dqn': False,
        # 'max_steps': 200  # Not for DQNAgent
    }
    
    return runner.run_experiment('baseline_dqn', config, num_seeds=3, num_episodes=2000)


def run_hyperparameter_sweep():
    """Run hyperparameter sweep experiments"""
    runner = DQNExperimentRunner('./experiments_sweep')
    
    configs = [
        ('baseline', {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_decay': 0.995,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'target_update_freq': 100,
            'use_double_dqn': False,
            'max_steps': 200
        }),
        ('double_dqn', {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_decay': 0.995,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'target_update_freq': 100,
            'use_double_dqn': True,  # Enable Double DQN
            'max_steps': 200
        }),
        ('larger_buffer', {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_decay': 0.995,
            'buffer_capacity': 50000,  # 5x larger
            'batch_size': 64,
            'target_update_freq': 100,
            'use_double_dqn': False,
            'max_steps': 200
        }),
        ('higher_lr', {
            'learning_rate': 5e-3,  # 5x higher
            'gamma': 0.99,
            'epsilon_decay': 0.995,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'target_update_freq': 100,
            'use_double_dqn': False,
            'max_steps': 200
        }),
        ('slower_epsilon', {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_decay': 0.998,  # Slower decay
            'buffer_capacity': 10000,
            'batch_size': 64,
            'target_update_freq': 100,
            'use_double_dqn': False,
            'max_steps': 200
        })
    ]
    
    return runner.compare_configurations(configs)


def run_double_dqn_comparison():
    """Compare standard DQN vs Double DQN"""
    runner = DQNExperimentRunner('./experiments_double_dqn')
    
    configs = [
        ('standard_dqn', {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_decay': 0.995,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'target_update_freq': 100,
            'use_double_dqn': False,
            'max_steps': 200
        }),
        ('double_dqn', {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon_decay': 0.995,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'target_update_freq': 100,
            'use_double_dqn': True,
            'max_steps': 200
        })
    ]
    
    return runner.compare_configurations(configs)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DQN EXPERIMENTS FOR TAXI-V3")
    print("Person C (Ingrid)")
    print("="*60 + "\n")
    
    print("Choose experiment:")
    print("1. Baseline DQN (single config, 3 seeds)")
    print("2. Hyperparameter Sweep (5 configs)")
    print("3. Double DQN Comparison (2 configs)")
    print("4. Run all experiments")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        run_baseline_experiment()
    elif choice == "2":
        run_hyperparameter_sweep()
    elif choice == "3":
        run_double_dqn_comparison()
    elif choice == "4":
        print("\nRunning ALL experiments...\n")
        run_baseline_experiment()
        run_hyperparameter_sweep()
        run_double_dqn_comparison()
        print("\nAll experiments complete!")
    else:
        print("Invalid choice!")
