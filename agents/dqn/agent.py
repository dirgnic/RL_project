"""
DQN Agent for Taxi-v3

Main agent class that combines network, replay buffer, and training logic.
Person C (Ingrid) - implement and tune this!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .network import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent
    
    Based on Labs 5 & 6 implementations
    """
    
    def __init__(
        self,
        state_size=500,
        action_size=6,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        device=None
    ):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Number of possible states (500 for Taxi-v3)
            action_size: Number of actions (6 for Taxi-v3)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_capacity: Replay buffer size
            batch_size: Batch size for training
            target_update_freq: Steps between target network updates
            device: torch device (cuda/cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Create main and target networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        # Training step counter
        self.steps = 0
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state (int)
            training: Whether in training mode (use epsilon-greedy) or evaluation (greedy)
            
        Returns:
            action: Selected action (int)
        """
        # Exploration: random action
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation: best action according to Q-network
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.long, device=self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step (if buffer is ready)
        
        Returns:
            loss: Training loss (float) or None if buffer not ready
        """
        # Check if buffer has enough samples
        if not self.memory.is_ready(self.batch_size):
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.long, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Compute current Q-values: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


# TODO for Ingrid:
# 1. Start with these hyperparameters, then tune
# 2. Monitor loss - should decrease over time
# 3. If loss explodes, reduce learning_rate
# 4. If training is slow, increase batch_size
# 5. If unstable, update target network more frequently
# 6. Compare with Irina's Q-learning results!
