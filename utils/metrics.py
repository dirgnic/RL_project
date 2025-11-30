"""
Performance metrics for RL experiments.
"""
import numpy as np


def calculate_success_rate(rewards, threshold):
    """
    Calculate success rate based on reward threshold.
    
    Args:
        rewards: List of episode rewards
        threshold: Reward threshold for success
    
    Returns:
        Success rate (0-1)
    """
    return np.mean(np.array(rewards) >= threshold)


def calculate_average_return(rewards, window=100):
    """
    Calculate average return over last N episodes.
    
    Args:
        rewards: List of episode rewards
        window: Window size for averaging
    
    Returns:
        Average return
    """
    if len(rewards) < window:
        return np.mean(rewards)
    return np.mean(rewards[-window:])


def calculate_stability(rewards, window=100):
    """
    Calculate training stability (inverse of variance).
    
    Args:
        rewards: List of episode rewards
        window: Window size
    
    Returns:
        Stability score
    """
    if len(rewards) < window:
        return np.std(rewards)
    return np.std(rewards[-window:])


def calculate_sample_efficiency(rewards, threshold, max_episodes):
    """
    Calculate sample efficiency (episodes to reach threshold).
    
    Args:
        rewards: List of episode rewards
        threshold: Target reward threshold
        max_episodes: Maximum episodes
    
    Returns:
        Episodes to reach threshold (or max_episodes if not reached)
    """
    for i, reward in enumerate(rewards):
        if reward >= threshold:
            return i + 1
    return max_episodes


def calculate_convergence_speed(rewards, window=100):
    """
    Calculate convergence speed (slope of last N episodes).
    
    Args:
        rewards: List of episode rewards
        window: Window size
    
    Returns:
        Convergence speed
    """
    if len(rewards) < window:
        x = np.arange(len(rewards))
        y = np.array(rewards)
    else:
        x = np.arange(window)
        y = np.array(rewards[-window:])
    
    # Linear regression
    slope = np.polyfit(x, y, 1)[0]
    return slope


def print_metrics_summary(rewards, steps=None, label="Algorithm"):
    """
    Print summary of performance metrics.
    
    Args:
        rewards: List of episode rewards
        steps: List of episode steps (optional)
        label: Algorithm label
    """
    print("\n" + "=" * 60)
    print(f"{label} - Performance Metrics")
    print("=" * 60)
    
    print(f"Total Episodes: {len(rewards)}")
    print(f"Mean Reward: {np.mean(rewards):.2f}")
    print(f"Std Reward: {np.std(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"Median Reward: {np.median(rewards):.2f}")
    
    # Last 100 episodes
    if len(rewards) >= 100:
        last_100 = rewards[-100:]
        print(f"\nLast 100 Episodes:")
        print(f"  Mean: {np.mean(last_100):.2f}")
        print(f"  Std: {np.std(last_100):.2f}")
        print(f"  Min: {np.min(last_100):.2f}")
        print(f"  Max: {np.max(last_100):.2f}")
    
    if steps is not None:
        print(f"\nMean Steps: {np.mean(steps):.2f}")
        print(f"Std Steps: {np.std(steps):.2f}")
    
    print("=" * 60)
