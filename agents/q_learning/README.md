# Q-Learning Agent

Implementation of the Q-Learning algorithm for reinforcement learning.

## Files

- `q_table.py` - Q-Table implementation with discretization support
- `train.py` - Training script
- `utils.py` - Utility functions (epsilon-greedy, evaluation)

## Features

- **Q-Table with Discretization**: Handles continuous state spaces by discretizing them into bins
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Configurable Hyperparameters**: Alpha (learning rate), gamma (discount factor), epsilon decay
- **Multiple Reward Types**: Supports custom reward shaping through environment wrappers
- **Logging and Visualization**: CSV logging and learning curve plots
- **Model Persistence**: Save and load trained Q-Tables

## Usage

### Training

Train using a configuration file:

```bash
python agents/q_learning/train.py --config experiments/configs/q_learning_cartpole_default.json
```

### Evaluation

Evaluate a trained model:

```python
from env import load_environment
from agents.q_learning.q_table import QTable
from agents.q_learning.utils import evaluate_policy

# Load environment
env = load_environment("CartPole-v1", seed=42)

# Load trained Q-Table
q_table = QTable(num_actions=env.action_space.n)
q_table.load("results/q_learning/models/q_learning_cartpole_default_final.pkl")

# Evaluate
avg_reward, episode_rewards = evaluate_policy(env, q_table, num_episodes=10, render=True)
```

### Custom Training Loop

```python
from env import load_environment, CustomRewardWrapper
from agents.q_learning.q_table import QTable
from agents.q_learning.utils import epsilon_greedy, decay_epsilon

# Create environment
env = load_environment("CartPole-v1", seed=42)
env = CustomRewardWrapper(env, reward_type="bonus_center", env_name="CartPole-v1")

# Initialize Q-Table
state_bounds = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
q_table = QTable(num_actions=2, num_bins=[10, 10, 10, 10], state_bounds=state_bounds)

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
num_episodes = 1000

# Training loop
for episode in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Select action
        action = epsilon_greedy(q_table, state, epsilon, env.action_space.n)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Q-Learning update
        current_q = q_table.get_q_value(state, action)
        if done:
            target_q = reward
        else:
            max_next_q = q_table.get_max_q_value(next_state)
            target_q = reward + gamma * max_next_q
        
        new_q = current_q + alpha * (target_q - current_q)
        q_table.update(state, action, new_q)
        
        state = next_state
        episode_reward += reward
    
    # Decay epsilon
    epsilon = decay_epsilon(epsilon, epsilon_end, epsilon_decay)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {epsilon:.4f}")

# Save model
q_table.save("my_q_learning_model.pkl")
```

## Configuration File Format

```json
{
    "experiment_name": "q_learning_cartpole_default",
    "seed": 42,
    "environment": {
        "name": "CartPole-v1",
        "reward_type": "default"
    },
    "hyperparameters": {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "num_episodes": 1000
    },
    "discretization": {
        "enabled": true,
        "bins": {
            "0": 10,
            "1": 10,
            "2": 10,
            "3": 10
        }
    },
    "logging": {
        "log_interval": 50,
        "save_model": true,
        "save_interval": 200
    }
}
```

## Hyperparameter Tuning

### Learning Rate (alpha)
- Controls how much new information overrides old information
- Range: 0.01 - 0.5
- Higher values: faster learning but less stable
- Lower values: slower learning but more stable

### Discount Factor (gamma)
- Determines importance of future rewards
- Range: 0.9 - 0.99
- Higher values: agent considers long-term rewards
- Lower values: agent focuses on immediate rewards

### Epsilon Decay
- Controls exploration vs exploitation trade-off
- Start high (1.0) for exploration
- Decay to low value (0.01-0.05) for exploitation
- Typical decay rate: 0.995

### Discretization Bins
- More bins: better precision but larger Q-Table
- Fewer bins: smaller Q-Table but coarser approximation
- Typical range: 10-20 bins per dimension

## Results

Results are saved in:
- `results/q_learning/logs/` - CSV log files
- `results/q_learning/plots/` - Learning curve plots
- `results/q_learning/models/` - Trained Q-Tables

## Tips

1. **Start with default hyperparameters** and tune based on performance
2. **Monitor Q-Table size** - too many states can slow training
3. **Use appropriate discretization** for your environment
4. **Compare with and without reward shaping** to measure impact
5. **Run multiple seeds** to ensure reproducibility

## References

- Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
