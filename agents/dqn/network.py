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
    
    def __init__(self, state_size=500, action_size=6, embedding_dim=64, hidden_dims=[128, 64],
                 use_one_hot=False, extra_feature_dim=0):
        super(DQNNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.use_one_hot = use_one_hot
        self.extra_feature_dim = extra_feature_dim
        if use_one_hot:
            input_dim = state_size + extra_feature_dim
        else:
            self.embedding = nn.Embedding(state_size, embedding_dim)
            input_dim = embedding_dim + extra_feature_dim
        # Build hidden layers
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], action_size)

    def forward(self, state, extra_features=None):
        """
        Forward pass
        Args:
            state: Tensor of shape (batch_size,) containing state indices (int) or one-hot (if use_one_hot)
            extra_features: Optional tensor of shape (batch_size, extra_feature_dim)
        Returns:
            q_values: Tensor of shape (batch_size, action_size)
        """
        if self.use_one_hot:
            # Convert state indices to one-hot
            x = F.one_hot(state, num_classes=self.state_size).float()
        else:
            x = self.embedding(state)
        if self.extra_feature_dim > 0 and extra_features is not None:
            x = torch.cat([x, extra_features], dim=-1)
        x = self.hidden(x)
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
