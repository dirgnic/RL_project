"""
DQN Network Architecture for Taxi-v3

This network takes a discrete state index and outputs Q-values for all actions.
For Taxi-v3: 500 possible states → 6 actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete state space (Taxi-v3)
    
    Architecture:
        State Index → Embedding → Hidden Layers → Q-values
    
    Args:
        state_size: Number of possible states (500 for Taxi-v3)
        action_size: Number of actions (6 for Taxi-v3)
        embedding_dim: Size of state embedding (default: 64)
        hidden_dims: List of hidden layer sizes (default: [128, 64])
    """
    
    def __init__(self, state_size=500, action_size=6, embedding_dim=64, hidden_dims=[128, 64]):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # State embedding layer (learns to represent discrete states)
        self.embedding = nn.Embedding(state_size, embedding_dim)
        
        # Build hidden layers
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        
        # Output layer: Q-values for each action
        self.output = nn.Linear(hidden_dims[-1], action_size)
        
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Tensor of shape (batch_size,) containing state indices
            
        Returns:
            q_values: Tensor of shape (batch_size, action_size)
        """
        # Embed the discrete state
        x = self.embedding(state)
        
        # Pass through hidden layers
        x = self.hidden(x)
        
        # Get Q-values for all actions
        q_values = self.output(x)
        
        return q_values


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN: Separates value and advantage streams
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
    
    This is an OPTIONAL extension from Lab 6
    """
    
    def __init__(self, state_size=500, action_size=6, embedding_dim=64, hidden_dims=[128, 64]):
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared embedding
        self.embedding = nn.Embedding(state_size, embedding_dim)
        
        # Shared hidden layers
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, state):
        """
        Forward pass with dueling architecture
        
        Args:
            state: Tensor of shape (batch_size,) containing state indices
            
        Returns:
            q_values: Tensor of shape (batch_size, action_size)
        """
        # Shared embedding and hidden layers
        x = self.embedding(state)
        x = self.shared(x)
        
        # Compute value and advantage
        value = self.value_stream(x)  # (batch_size, 1)
        advantage = self.advantage_stream(x)  # (batch_size, action_size)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# TODO for Ingrid:
# 1. Decide which network to use (start with DQNNetwork)
# 2. Experiment with embedding_dim (32, 64, 128)
# 3. Experiment with hidden_dims ([128, 64], [256, 128], [512, 256, 128])
# 4. Try DuelingDQNNetwork if basic DQN works well
