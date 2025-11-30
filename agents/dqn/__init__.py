"""
DQN Agent for Taxi-v3
Person C (Ingrid) - Deep Value-Based RL

Based on Labs 5 & 6
"""

from .network import DQNNetwork
from .replay_buffer import ReplayBuffer
from .agent import DQNAgent

__all__ = ['DQNNetwork', 'ReplayBuffer', 'DQNAgent']
