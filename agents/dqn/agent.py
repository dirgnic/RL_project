"""
DQN Agent for Taxi-v3 - COMPLETE IMPLEMENTATION
Person C (Ingrid)

This is a production-ready, fully tested DQN implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from .network import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent - COMPLETE & TESTED
    
    Implements:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Gradient clipping
    - Double DQN (optional)
    """
    
    def __init__(
        self,
        state_size: int = 500,
        action_size: int = 6,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[torch.device] = None,
        use_double_dqn: bool = False,
        hidden_dims: list = None,
        embedding_dim: int = 64,
        use_one_hot: bool = False,
        extra_feature_dim: int = 0
    ):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Number of possible states (500 for Taxi-v3)
            action_size: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay per episode
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Steps between target updates
            device: torch device (cuda/cpu)
            use_double_dqn: Whether to use Double DQN
            hidden_dims: Hidden layer sizes
            embedding_dim: State embedding dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Networks
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        self.policy_net = DQNNetwork(
            state_size, action_size, embedding_dim, hidden_dims,
            use_one_hot=use_one_hot, extra_feature_dim=extra_feature_dim
        ).to(self.device)
        self.target_net = DQNNetwork(
            state_size, action_size, embedding_dim, hidden_dims,
            use_one_hot=use_one_hot, extra_feature_dim=extra_feature_dim
        ).to(self.device)
        self.use_one_hot = use_one_hot
        self.extra_feature_dim = extra_feature_dim
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss
        self.criterion = nn.SmoothL1Loss()  # Huber loss (more stable than MSE)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        # Tracking
        self.steps = 0
        self.episodes = 0
    
    def select_action(self, state: int, extra_features: np.ndarray = None, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            action: Selected action
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.long, device=self.device)
            if self.extra_feature_dim > 0 and extra_features is not None:
                extra_tensor = torch.tensor([extra_features], dtype=torch.float32, device=self.device)
                q_values = self.policy_net(state_tensor, extra_tensor)
            else:
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int, 
        done: bool
    ):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self, extra_features_batch=None, next_extra_features_batch=None) -> Optional[float]:
        """
        Perform one training step
        
        Returns:
            loss: Training loss or None if buffer not ready
        """
        if not self.memory.is_ready(self.batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.long, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        # Handle extra features if provided
        if self.extra_feature_dim > 0 and extra_features_batch is not None:
            extra_features = torch.tensor(extra_features_batch, dtype=torch.float32, device=self.device)
            current_q = self.policy_net(states, extra_features).gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.extra_feature_dim > 0 and next_extra_features_batch is not None:
                next_extra_features = torch.tensor(next_extra_features_batch, dtype=torch.float32, device=self.device)
                if self.use_double_dqn:
                    next_actions = self.policy_net(next_states, next_extra_features).argmax(1)
                    next_q = self.target_net(next_states, next_extra_features).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    next_q = self.target_net(next_states, next_extra_features).max(1)[0]
            else:
                if self.use_double_dqn:
                    next_actions = self.policy_net(next_states).argmax(1)
                    next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes += 1
    
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'gamma': self.gamma,
                'use_double_dqn': self.use_double_dqn
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"Model loaded from {filepath}")
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'buffer_size': len(self.memory)
        }
