# Reinforcement Learning Project

University project for implementing and comparing reinforcement learning algorithms.

## Overview


This project implements and compares three RL algorithms using [HighwayEnv](https://github.com/eleurent/highway-env):
- **DQN** (Deep Q-Network)
- **PPO** (Proximal Policy Optimization)
- **A2C** (Advantage Actor-Critic)

All agents are trained and evaluated on the HighwayEnv environment with custom reward shaping and logging.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/dirgnic/RL_project.git
cd RL_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

See [SETUP.md](SETUP.md) for detailed installation instructions.


### Train Agents

```bash
# Train DQN agent
python train_dqn.py

# Train PPO agent
python train_ppo.py

# Train A2C agent
python train_a2c.py
```

### Evaluate and Plot Results

```bash
# Evaluate all trained agents
python evaluate_agents.py

# Plot comparison of agent performance
python plot_results.py
```

## Project Structure

```
RL_proj/
├── env/                # Environment loader and reward wrapper
├── agents/dqn/         # (Legacy) DQN implementation (custom)
├── train_dqn.py        # SB3 DQN training script
├── train_ppo.py        # SB3 PPO training script
├── train_a2c.py        # SB3 A2C training script
├── evaluate_agents.py  # Evaluate all trained agents
├── plot_results.py     # Plot agent performance
├── results/            # Training logs, models, plots
├── utils/              # Logging, plotting, metrics
├── tests/              # Test scripts
└── notebooks/          # Jupyter notebooks
```

## Features

- **HighwayEnv Only**: Focused on the HighwayEnv driving environment
- **Reward Shaping**: Custom reward wrapper for HighwayEnv
- **SB3 Agents**: DQN, PPO, and A2C via Stable Baselines3
- **Logging**: CSV logging for training and evaluation
- **Visualization**: Boxplots and comparison of agent performance
- **Model Persistence**: Save and load trained models
- **Reproducibility**: Seed management for consistent results


## Documentation

- **Setup Guide**: [SETUP.md](SETUP.md)
- **Implementation Summary**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Environment Options**: [ENVIRONMENT_OPTIONS.md](ENVIRONMENT_OPTIONS.md)



## Requirements

- Python 3.10+
- Gymnasium
- HighwayEnv
- Stable Baselines3
- PyTorch
- Matplotlib, Pandas, Seaborn

See [requirements.txt](requirements.txt) for the full list.

## Testing

```bash
# Test environment loading and wrappers
python tests/test_environment.py
```

## Results

Results are organized by agent:
- `results/dqn/` - DQN results
- `results/ppo/` - PPO results
- `results/a2c/` - A2C results
- `results/comparison/` - Comparison tables and plots

## Contributing

1. Create a new branch for your feature
2. Make changes and test
3. Commit with descriptive message
4. Push and create pull request

## License

This is a university project for educational purposes.


## Resources

- [HighwayEnv Documentation](https://highway-env.farama.org/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
