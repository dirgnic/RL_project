# Reinforcement Learning Project

University project for implementing and comparing reinforcement learning algorithms.

## Overview

This project implements three reinforcement learning algorithms:
- **Q-Learning** (Tabular method)
- **DQN** (Deep Q-Network)
- **PPO** (Proximal Policy Optimization)

Each algorithm is tested on Gymnasium environments with both default and modified (reward-shaped) reward functions.

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

### Run Q-Learning

```bash
# Train Q-Learning on CartPole
python agents/q_learning/train.py --config experiments/configs/q_learning_cartpole_default.json

# Run all Q-Learning experiments
./experiments/scripts/run_q_learning.sh

# Compare results
python experiments/scripts/compare_q_learning.py
```

## Project Structure

```
RL_proj/
├── env/                      # Environment module (loading, reward shaping)
├── agents/
│   ├── q_learning/          # Q-Learning implementation
│   ├── dqn/                 # DQN implementation
│   └── ppo/                 # PPO implementation
├── experiments/
│   ├── configs/             # Experiment configurations (JSON)
│   └── scripts/             # Experiment runners
├── results/                  # Training results (logs, plots, models)
├── utils/                    # Utilities (logging, plotting, metrics)
├── tests/                    # Test scripts
└── notebooks/                # Jupyter notebooks
```

## Features

- **Multiple Environments**: CartPole, MountainCar, LunarLander
- **Reward Shaping**: Custom reward wrappers for each environment
- **Configurable Experiments**: JSON config files for easy experimentation
- **Logging**: CSV logging with TensorBoard support
- **Visualization**: Learning curves, comparison plots
- **Model Persistence**: Save and load trained models
- **Reproducibility**: Seed management for consistent results

## Documentation

- **Setup Guide**: [SETUP.md](SETUP.md)
- **Project Plan**: [PROIECT_ECHIPA/01_PLAN_PROIECT.md](PROIECT_ECHIPA/01_PLAN_PROIECT.md)
- **Environment Research**: [PROIECT_ECHIPA/02_ENVIRONMENT_RESEARCH.md](PROIECT_ECHIPA/02_ENVIRONMENT_RESEARCH.md)
- **RL Theory**: [PROIECT_ECHIPA/00_CHEAT_SHEET_RL.md](PROIECT_ECHIPA/00_CHEAT_SHEET_RL.md)
- **Q-Learning README**: [agents/q_learning/README.md](agents/q_learning/README.md)

## Team

- **Ingrid** - Q-Learning implementation, environment research
- **Matei** - DQN implementation
- **Iustin** - PPO implementation
- **Irina** - Comparative analysis, visualization

## Requirements

- Python 3.10+
- Gymnasium 0.29.1
- NumPy 1.24.3
- PyTorch 2.0.1
- Stable Baselines3 2.1.0
- Matplotlib, Pandas, TensorBoard

See [requirements.txt](requirements.txt) for full list.

## Testing

```bash
# Test environment loading and wrappers
python tests/test_environment.py

# Test Q-Learning implementation
python tests/test_q_learning.py
```

## Results

Results are organized by algorithm:
- `results/q_learning/` - Q-Learning results
- `results/dqn/` - DQN results
- `results/ppo/` - PPO results
- `results/comparison/` - Comparison tables and plots

## Contributing

1. Create a new branch for your feature
2. Make changes and test
3. Commit with descriptive message
4. Push and create pull request

## License

This is a university project for educational purposes.

## Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- Course materials in `Cursuri/` and `Laboratoare/` folders
