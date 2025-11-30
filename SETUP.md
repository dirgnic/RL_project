# Setup Guide

This guide will help you set up your development environment for the RL project.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/dirgnic/RL_project.git
cd RL_project
```

### 2. Create Virtual Environment

**Option A: Using venv (recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**Option B: Using conda**

```bash
# Create conda environment
conda create -n rl_proj python=3.10

# Activate conda environment
conda activate rl_proj
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- gymnasium (RL environments)
- numpy (numerical computations)
- matplotlib (plotting)
- pandas (data handling)
- torch (deep learning for DQN)
- stable-baselines3 (PPO implementation)

### 4. Verify Installation

Run the test scripts to verify everything is working:

```bash
# Test environment loading and wrappers
python tests/test_environment.py

# Test Q-Learning implementation
python tests/test_q_learning.py
```

If all tests pass, you're ready to start!

## Quick Start

### Run a Q-Learning Experiment

```bash
# Train Q-Learning on CartPole with default reward
python agents/q_learning/train.py --config experiments/configs/q_learning_cartpole_default.json
```

### Run All Q-Learning Experiments

```bash
# Make script executable
chmod +x experiments/scripts/run_q_learning.sh

# Run all experiments
./experiments/scripts/run_q_learning.sh
```

### Compare Results

```bash
python experiments/scripts/compare_q_learning.py
```

## Directory Structure

```
RL_proj/
├── env/                      # Environment module
│   ├── base_env.py          # Environment loading
│   ├── reward_wrapper.py    # Reward shaping
│   └── README.md
├── agents/                   # RL algorithms
│   ├── q_learning/          # Q-Learning
│   ├── dqn/                 # Deep Q-Network
│   └── ppo/                 # Proximal Policy Optimization
├── experiments/
│   ├── configs/             # Experiment configurations
│   └── scripts/             # Experiment runners
├── results/
│   ├── q_learning/          # Q-Learning results
│   ├── dqn/                 # DQN results
│   ├── ppo/                 # PPO results
│   └── comparison/          # Comparison results
├── utils/                    # Utility functions
│   ├── logging.py           # Logging utilities
│   ├── plotting.py          # Plotting utilities
│   ├── seed.py              # Seed management
│   └── metrics.py           # Performance metrics
├── tests/                    # Test scripts
├── notebooks/                # Jupyter notebooks
├── requirements.txt          # Python dependencies
└── README.md                 # Main documentation
```

## Common Issues

### Issue: Import errors

**Solution**: Make sure you activated your virtual environment:

```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Issue: Gymnasium environment not found

**Solution**: Some environments require additional dependencies:

```bash
# For Box2D environments (LunarLander)
pip install gymnasium[box2d]

# For all environments
pip install gymnasium[all]
```

### Issue: PyTorch not installing

**Solution**: Install PyTorch separately based on your system:

Visit https://pytorch.org/get-started/locally/ and follow instructions for your system.

### Issue: Permission denied for shell scripts

**Solution**: Make scripts executable:

```bash
chmod +x experiments/scripts/*.sh
```

## Development Workflow

### 1. Create a New Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Edit code, run experiments, etc.

### 3. Test Your Changes

```bash
# Run relevant tests
python tests/test_environment.py
python tests/test_q_learning.py
```

### 4. Commit and Push

```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

Go to GitHub and create a pull request to merge your changes.

## Tips

1. **Always activate your virtual environment** before running code
2. **Run tests** after making changes to ensure nothing broke
3. **Use configuration files** for experiments to keep track of hyperparameters
4. **Save results** with descriptive names including date and experiment details
5. **Document your experiments** in the lab notebooks

## Next Steps

- Read the documentation in `PROIECT_ECHIPA/` folder
- Check the Q-Learning README: `agents/q_learning/README.md`
- Explore example configurations: `experiments/configs/`
- Start with simple environments (CartPole) before moving to complex ones

## Getting Help

- Check existing documentation in the repo
- Ask team members in the WhatsApp group
- Review lab materials in `Laboratoare/` folder
- Consult course slides in `Cursuri/` folder

## Resources

- Gymnasium Documentation: https://gymnasium.farama.org/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Stable Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
- Course Materials: Check `Cursuri/` and `Laboratoare/` folders
