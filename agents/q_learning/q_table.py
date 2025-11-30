"""
Q-Table implementation for Q-Learning.
"""
import numpy as np
import pickle


class QTable:
    """
    Q-Table pentru Q-Learning.
    
    Suportă atât discrete states (dict) cât și continuous states (cu discretizare).
    """
    
    def __init__(self, num_actions, num_bins=None, state_bounds=None, init_value=0.0):
        """
        Initialize Q-Table.
        
        Args:
            num_actions: Numărul de acțiuni
            num_bins: Lista cu numărul de bins pentru fiecare dimensiune a state-ului (pentru discretizare)
            state_bounds: Lista cu bounds pentru fiecare dimensiune [(min, max), ...]
            init_value: Valoarea inițială pentru Q-values
        """
        self.num_actions = num_actions
        self.num_bins = num_bins
        self.state_bounds = state_bounds
        self.init_value = init_value
        
        # Use dictionary for sparse representation
        self.q_table = {}
        
        # For discretization
        if num_bins is not None and state_bounds is not None:
            self.bins = self._create_bins()
        else:
            self.bins = None
    
    def _create_bins(self):
        """Create bins for discretization."""
        bins = []
        for i, num_bin in enumerate(self.num_bins):
            low, high = self.state_bounds[i]
            bins.append(np.linspace(low, high, num_bin))
        return bins
    
    def discretize_state(self, state):
        """
        Discretize continuous state.
        
        Args:
            state: Continuous state (numpy array or list)
        
        Returns:
            Tuple representing discretized state
        """
        if self.bins is None:
            # If no discretization needed, convert to tuple
            return tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        discrete_state = []
        for i, val in enumerate(state):
            # Clip value to bounds
            val = np.clip(val, self.state_bounds[i][0], self.state_bounds[i][1])
            # Find bin index
            bin_idx = np.digitize(val, self.bins[i]) - 1
            bin_idx = np.clip(bin_idx, 0, self.num_bins[i] - 1)
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def get_q_value(self, state, action):
        """
        Get Q-value for (state, action) pair.
        
        Args:
            state: State (will be discretized if needed)
            action: Action index
        
        Returns:
            Q-value
        """
        state = self.discretize_state(state)
        
        if state not in self.q_table:
            self.q_table[state] = np.full(self.num_actions, self.init_value)
        
        return self.q_table[state][action]
    
    def get_max_q_value(self, state):
        """
        Get maximum Q-value for a state.
        
        Args:
            state: State
        
        Returns:
            Maximum Q-value
        """
        state = self.discretize_state(state)
        
        if state not in self.q_table:
            self.q_table[state] = np.full(self.num_actions, self.init_value)
        
        return np.max(self.q_table[state])
    
    def get_best_action(self, state):
        """
        Get best action for a state (greedy policy).
        
        Args:
            state: State
        
        Returns:
            Best action index
        """
        state = self.discretize_state(state)
        
        if state not in self.q_table:
            self.q_table[state] = np.full(self.num_actions, self.init_value)
        
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, value):
        """
        Update Q-value for (state, action) pair.
        
        Args:
            state: State
            action: Action index
            value: New Q-value
        """
        state = self.discretize_state(state)
        
        if state not in self.q_table:
            self.q_table[state] = np.full(self.num_actions, self.init_value)
        
        self.q_table[state][action] = value
    
    def save(self, filepath):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'num_actions': self.num_actions,
                'num_bins': self.num_bins,
                'state_bounds': self.state_bounds,
                'init_value': self.init_value
            }, f)
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.num_actions = data['num_actions']
        self.num_bins = data['num_bins']
        self.state_bounds = data['state_bounds']
        self.init_value = data['init_value']
        
        if self.num_bins is not None and self.state_bounds is not None:
            self.bins = self._create_bins()
        
        print(f"Q-table loaded from {filepath}")
    
    def get_stats(self):
        """Get statistics about Q-table."""
        if not self.q_table:
            return {
                'num_states': 0,
                'num_entries': 0,
                'mean_q_value': 0,
                'max_q_value': 0,
                'min_q_value': 0
            }
        
        all_q_values = np.concatenate([v for v in self.q_table.values()])
        
        return {
            'num_states': len(self.q_table),
            'num_entries': len(all_q_values),
            'mean_q_value': np.mean(all_q_values),
            'max_q_value': np.max(all_q_values),
            'min_q_value': np.min(all_q_values)
        }


if __name__ == "__main__":
    # Test Q-Table
    print("Testing Q-Table...")
    
    # Test with discrete states
    q_table = QTable(num_actions=2)
    
    state = (0, 1, 0, 1)
    action = 1
    
    print(f"Initial Q-value: {q_table.get_q_value(state, action)}")
    q_table.update(state, action, 5.0)
    print(f"Updated Q-value: {q_table.get_q_value(state, action)}")
    print(f"Best action: {q_table.get_best_action(state)}")
    
    # Test with continuous states (discretization)
    print("\nTesting with discretization...")
    q_table_disc = QTable(
        num_actions=2,
        num_bins=[10, 10, 10, 10],
        state_bounds=[(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
    )
    
    continuous_state = [0.5, 1.0, 0.1, -0.5]
    print(f"Continuous state: {continuous_state}")
    print(f"Discretized state: {q_table_disc.discretize_state(continuous_state)}")
    
    print("\nQ-Table stats:")
    print(q_table.get_stats())
