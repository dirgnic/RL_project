import numpy as np
"""
DQN Network Architecture for Taxi-v3

This network takes a discrete state index and outputs Q-values for all actions.
For Taxi-v3: 500 possible states → 6 actions
"""

import torch
import torch.nn as nn


def decode_taxi_state(state):
    """
    Decode Taxi-v3 state integer into meaningful features.
    
    State encoding: ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest
    
    Returns: [taxi_row, taxi_col, pass_loc, dest] normalized to [0, 1]
    """
    if isinstance(state, torch.Tensor):
        state = state.long()
        dest = state % 4
        state = state // 4
        pass_loc = state % 5
        state = state // 5
        taxi_col = state % 5
        taxi_row = state // 5
        # Normalize to [0, 1]
        features = torch.stack([
            taxi_row.float() / 4.0,
            taxi_col.float() / 4.0,
            pass_loc.float() / 4.0,
            dest.float() / 3.0
        ], dim=-1)
    else:
        # Numpy/int version
        state = int(state)
        dest = state % 4
        state = state // 4
        pass_loc = state % 5
        state = state // 5
        taxi_col = state % 5
        taxi_row = state // 5
        features = np.array([
            taxi_row / 4.0,
            taxi_col / 4.0,
            pass_loc / 4.0,
            dest / 3.0
        ], dtype=np.float32)
    return features


class TaxiDQNNetwork(nn.Module):
    """
    DQN Network specifically designed for Taxi-v3.
    
    Instead of using embeddings, this decodes the state integer into
    meaningful features (taxi position, passenger location, destination)
    and uses a simple MLP to predict Q-values.
    
    This is more sample-efficient for Taxi-v3's small state space.
    """
    
    def __init__(self, state_size=500, action_size=6, hidden_dims=[64, 64]):
        super(TaxiDQNNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.obs_dim = state_size  # For compatibility with agent
        
        # Input: 4 decoded features (taxi_row, taxi_col, pass_loc, dest)
        input_dim = 4
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], action_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: Integer state indices, shape (batch_size,) or scalar
        
        Returns:
            Q-values for all actions, shape (batch_size, action_size)
        """
        # Handle different input types
        if isinstance(state, (int, np.integer)):
            state = torch.tensor([state], dtype=torch.long)
        elif isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.long)
        elif isinstance(state, torch.Tensor):
            state = state.long()
        
        # Ensure at least 1D
        if state.dim() == 0:
            state = state.unsqueeze(0)
        
        # Decode state into features
        features = decode_taxi_state(state)
        
        # Forward through network
        x = self.hidden(features)
        q_values = self.output(x)
        
        return q_values


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for continuous state spaces.
    
    Architecture:
        State Vector → Hidden Layers → Q-values
    
    Args:
        obs_dim: Dimension of observation/state vector
        action_size: Number of actions
        hidden_dims: List of hidden layer sizes (default: [128, 64])
    """
    
    def __init__(self, obs_dim=5, action_size=2, hidden_dims=[128, 64]):
        super(DQNNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_size = action_size
        input_dim = obs_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], action_size)

    def forward(self, state):
        # Accepts continuous state (batch_size, obs_dim) or (obs_dim,)
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        elif isinstance(state, torch.Tensor):
            state = state.to(torch.float32)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        x = state
        x = self.hidden(x)
        return self.output(x)


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
        self.obs_dim = state_size  # Add obs_dim for agent compatibility
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
        # Ensure state is tensor of type long for embedding
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.long)
        elif isinstance(state, torch.Tensor):
            state = state.to(torch.long)
        # Only unsqueeze if scalar (single state), not for batch
        if state.dim() == 0:
            state = state.unsqueeze(0)
        x = self.embedding(state)
        x = self.shared(x)
        value = self.value_stream(x)  # (batch_size, 1)
        advantage = self.advantage_stream(x)  # (batch_size, action_size)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# TODO for Ingrid:
# 1. Decide which network to use (start with DQNNetwork)
# 2. Experiment with embedding_dim (32, 64, 128)
# 3. Experiment with hidden_dims ([128, 64], [256, 128], [512, 256, 128])
# 4. Try DuelingDQNNetwork if basic DQN works well
