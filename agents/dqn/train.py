"""
COMPLETE DQN Training Script for Taxi-v3
Person C (Ingrid) - DQN Engineer

Features:
- Multi-seed training
- Evaluation tracking
- Visualization
- Hyperparameter experiments
- Model checkpointing

Run with: python agents/dqn/train.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from custom_env import MyCustomEnv
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import deque
from agents.dqn.agent import DQNAgent

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def train_dqn(
    env=None,
    num_episodes=2000,
    max_steps=200,
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_capacity=50000,  # Tuned value
    batch_size=128,         # Tuned value
    target_update_freq=500, # Tuned value
    seed=42
):
    """
    Train DQN agent on Taxi-v3
    
    Args:
        env_name: Gym environment name
        num_episodes: Number of episodes to train
        max_steps: Max steps per episode
        ... (other hyperparameters)
        seed: Random seed for reproducibility
    
    Returns:
        agent: Trained DQNAgent
        rewards: List of episode rewards
        losses: List of training losses
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    if env is None:
        env = MyCustomEnv(render_mode=None)
    obs, _ = env.reset(seed=seed)
    # Assume obs is np.ndarray with extra features at the end
    state_size = obs.shape[0]
    action_size = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    print(f"Environment: {env.__class__.__name__}")
    print(f"State space: {state_size}")
    print(f"Action space: {action_size}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("-" * 50)
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    recent_rewards = deque(maxlen=100)
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            if not isinstance(action, (list, np.ndarray)):
                action = [action, 0.0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            episode_reward += reward
            state = next_state
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        if episode_loss:
            episode_losses.append(np.mean(episode_loss))
        
        # Log progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Buffer Size: {len(agent.memory)}")
            print("-" * 50)
    
    env.close()
    
    return agent, episode_rewards, episode_losses


def plot_results(rewards, losses, save_path='dqn_results.png'):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')
    # Smooth with moving average
    window = 100
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, label=f'Moving Avg ({window})', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('DQN Training: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    ax2.plot(losses, alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('DQN Training: Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    plt.show()


def evaluate_agent(agent, env_name='Taxi-v3', num_episodes=100, seed=42):
    """
    Evaluate trained agent
    
    Returns:
        avg_reward: Average reward over episodes
        success_rate: Percentage of successful deliveries
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    episode_rewards = []
    successes = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(200):
            # Ensure state is float32 numpy array for DQN
            if isinstance(state, np.ndarray):
                state_eval = state.astype(np.float32)
            else:
                state_eval = np.array(state, dtype=np.float32)
            action = agent.select_action(state_eval, training=False)  # Greedy
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                if reward == 20:  # Successful delivery
                    successes += 1
                break
        episode_rewards.append(episode_reward)
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    success_rate = successes / num_episodes * 100
    
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    return avg_reward, success_rate


if __name__ == "__main__":
    print("=" * 50)
    print("DQN Training on Taxi-v3")
    print("Person C (Ingrid)")
    print("=" * 50)
    print()
    
    # Train agent
    agent, rewards, losses = train_dqn(
        num_episodes=200,  # Increased for better learning
        learning_rate=1e-3,
        seed=42
    )
    
    # Plot results
    plot_results(rewards, losses)
    
    # Evaluate
    evaluate_agent(agent, num_episodes=100)
    
    # Save model
    agent.save('dqn_taxi_model.pth')
    print("\nModel saved to dqn_taxi_model.pth")
    
    print("\nTraining complete!")
    
    # TODO for Ingrid:
    # 1. Run this script and observe training
    # 2. If reward doesn't improve, tune hyperparameters
    # 3. Compare results with Irina's Q-learning
    # 4. Try different seeds (42, 123, 456)
    # 5. Document what hyperparameters work best
