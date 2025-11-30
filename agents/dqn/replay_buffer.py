"""
Experience Replay Buffer for DQN

Stores transitions (s, a, r, s', done) and samples random batches.
This breaks correlation between consecutive samples and improves stability.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples
    
    Based on Lab 5 implementation
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state (int for Taxi-v3)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (int)
            done: Whether episode ended (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each as a numpy array
        """
        # Sample random batch
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples"""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    OPTIONAL: Prioritized Experience Replay (Lab 6)
    
    Samples important transitions more frequently based on TD error.
    Only implement this if basic DQN works well!
    """
    
    def __init__(self, capacity=10000, alpha=0.6):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
    
    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority"""
        # New transitions get max priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        super().push(state, action, reward, next_state, done)
        self.priorities.append(max_priority)
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample batch with prioritization
        
        Args:
            batch_size: Number of samples
            beta: Importance sampling correction (0 = no correction, 1 = full correction)
            
        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors
        
        Args:
            indices: Indices of samples to update
            td_errors: TD errors for those samples
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority


# TODO for Ingrid:
# 1. Start with basic ReplayBuffer (capacity=10000)
# 2. Try different capacities: 5000, 10000, 50000
# 3. Monitor memory usage (Taxi has small states so large buffer is fine)
# 4. Only try PrioritizedReplayBuffer if basic DQN works perfectly
