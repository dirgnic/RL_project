"""
Q-Learning training script.
"""
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from env import load_environment, CustomRewardWrapper
from agents.q_learning.q_table import QTable
from agents.q_learning.utils import epsilon_greedy, decay_epsilon
from utils.logging import Logger
from utils.plotting import plot_learning_curve


def train_q_learning(config):
    """
    Train Q-Learning agent.
    
    Args:
        config: Dictionary with configuration parameters
    """
    # Extract config parameters
    env_name = config['environment']['name']
    reward_type = config['environment']['reward_type']
    
    alpha = config['hyperparameters']['alpha']
    gamma = config['hyperparameters']['gamma']
    epsilon_start = config['hyperparameters']['epsilon_start']
    epsilon_end = config['hyperparameters']['epsilon_end']
    epsilon_decay = config['hyperparameters']['epsilon_decay']
    num_episodes = config['hyperparameters']['num_episodes']
    
    seed = config.get('seed', 42)
    experiment_name = config.get('experiment_name', 'q_learning_experiment')
    
    # Create environment
    print(f"Creating environment: {env_name}")
    env = load_environment(env_name, seed=seed)
    
    if reward_type != "default":
        env = CustomRewardWrapper(env, reward_type=reward_type, env_name=env_name)
    
    # Get environment info
    num_actions = env.action_space.n
    
    # Initialize Q-Table
    if config['discretization']['enabled']:
        bins = [config['discretization']['bins'][key] for key in sorted(config['discretization']['bins'].keys())]
        
        # Define state bounds based on environment
        if env_name == "CartPole-v1":
            state_bounds = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
        elif env_name == "MountainCar-v0":
            state_bounds = [(-1.2, 0.6), (-0.07, 0.07)]
        else:
            raise ValueError(f"State bounds not defined for {env_name}")
        
        q_table = QTable(num_actions=num_actions, num_bins=bins, state_bounds=state_bounds)
    else:
        q_table = QTable(num_actions=num_actions)
    
    # Initialize logger
    log_dir = project_root / "results" / "q_learning" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(log_dir, experiment_name)
    
    # Training loop
    epsilon = epsilon_start
    best_reward = -float('inf')
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon_start} -> {epsilon_end}")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Choose action (epsilon-greedy)
            action = epsilon_greedy(q_table, state, epsilon, num_actions)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Q-Learning update
            current_q = q_table.get_q_value(state, action)
            
            if done:
                # Terminal state
                target_q = reward
            else:
                # Non-terminal state
                max_next_q = q_table.get_max_q_value(next_state)
                target_q = reward + gamma * max_next_q
            
            # Update Q-value
            new_q = current_q + alpha * (target_q - current_q)
            q_table.update(state, action, new_q)
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Decay epsilon
        epsilon = decay_epsilon(epsilon, epsilon_end, epsilon_decay)
        
        # Log episode
        logger.log(episode, episode_reward, steps, epsilon)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Print progress
        if (episode + 1) % config['logging']['log_interval'] == 0:
            avg_reward = np.mean([logger.log_file.split(',')[-1] for _ in range(min(100, episode + 1))])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Best: {best_reward:.2f} | "
                  f"Epsilon: {epsilon:.4f} | "
                  f"States: {q_table.get_stats()['num_states']}")
        
        # Save model periodically
        if config['logging']['save_model'] and (episode + 1) % config['logging']['save_interval'] == 0:
            model_dir = project_root / "results" / "q_learning" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            q_table.save(model_dir / f"{experiment_name}_ep{episode + 1}.pkl")
    
    # Save final model
    model_dir = project_root / "results" / "q_learning" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    q_table.save(model_dir / f"{experiment_name}_final.pkl")
    
    # Plot learning curve
    plot_dir = project_root / "results" / "q_learning" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(
        logger.log_file,
        plot_dir / f"{experiment_name}_learning_curve.png",
        title=f"Q-Learning: {experiment_name}"
    )
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final epsilon: {epsilon:.4f}")
    print(f"Q-Table stats:")
    for key, value in q_table.get_stats().items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    env.close()
    return q_table


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Q-Learning agent')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Train
    train_q_learning(config)


if __name__ == "__main__":
    main()
