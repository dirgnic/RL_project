"""
DQN Network Architecture
========================
Simple feedforward neural network for Q-value approximation.
"""

import torch
import torch.nn as nn
import numpy as np


class DQNNetwork(nn.Module):
    """
    Simple DQN: State → Hidden Layers → Q-values for each action
    
    Args:
        obs_dim: Size of observation/state vector
        action_size: Number of actions
        hidden_dims: List of hidden layer sizes [64, 32]
    """
    
    def __init__(self, obs_dim, action_size, hidden_dims=[256, 128]):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], action_size)

    def forward(self, state):
        """Forward pass: state → Q-values."""
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        x = self.hidden(state)
        return self.output(x)
