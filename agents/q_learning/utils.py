"""
Utility functions for Q-Learning.
"""
import numpy as np


def epsilon_greedy(q_table, state, epsilon, num_actions):
    """
    Epsilon-greedy action selection.
    
    Args:
        q_table: Q-Table object
        state: Current state
        epsilon: Exploration probability
        num_actions: Number of actions
    
    Returns:
        Selected action
    """
    if np.random.random() < epsilon:
        # Explore: random action
        return np.random.randint(0, num_actions)
    else:
        # Exploit: best action
        return q_table.get_best_action(state)


def decay_epsilon(epsilon, epsilon_end, decay_rate):
    """
    Decay epsilon value.
    
    Args:
        epsilon: Current epsilon
        epsilon_end: Minimum epsilon value
        decay_rate: Decay rate (multiplicative)
    
    Returns:
        New epsilon value
    """
    return max(epsilon_end, epsilon * decay_rate)


def linear_decay_epsilon(episode, total_episodes, epsilon_start, epsilon_end):
    """
    Linear decay of epsilon.
    
    Args:
        episode: Current episode
        total_episodes: Total number of episodes
        epsilon_start: Starting epsilon
        epsilon_end: Ending epsilon
    
    Returns:
        Epsilon value for current episode
    """
    return epsilon_start - (epsilon_start - epsilon_end) * (episode / total_episodes)


def evaluate_policy(env, q_table, num_episodes=10, render=False, seed=None):
    """
    Evaluate trained policy.
    
    Args:
        env: Environment
        q_table: Trained Q-Table
        num_episodes: Number of evaluation episodes
        render: Whether to render
        seed: Random seed
    
    Returns:
        Average reward and list of episode rewards
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        if seed is not None:
            state, info = env.reset(seed=seed + episode)
        else:
            state, info = env.reset()
        
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Greedy action (no exploration)
            action = q_table.get_best_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation over {num_episodes} episodes:")
    print(f"  Average reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")
    
    return avg_reward, episode_rewards
