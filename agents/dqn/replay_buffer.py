"""
Experience Replay Buffer
========================
Stores transitions and samples random batches to break correlation.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size
    
    def __len__(self):
        return len(self.buffer)
