# Quick Command Reference

## Setup (One Time)

```bash
# Clone and setup
git clone https://github.com/dirgnic/RL_project.git
cd RL_project
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
pip install gymnasium[box2d]  # For LunarLander
```

## Before Each Session

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows
```

## Testing

```bash
# Test environment module
python tests/test_environment.py

# Test Q-Learning module
python tests/test_q_learning.py
```

## Running Experiments

```bash
# Single experiment (CartPole default)
python agents/q_learning/train.py --config experiments/configs/q_learning_cartpole_default.json

# Single experiment (CartPole modified)
python agents/q_learning/train.py --config experiments/configs/q_learning_cartpole_modified.json

# Single experiment (MountainCar default)
python agents/q_learning/train.py --config experiments/configs/q_learning_mountaincar_default.json

# Single experiment (MountainCar modified)
python agents/q_learning/train.py --config experiments/configs/q_learning_mountaincar_modified.json

# Single experiment (LunarLander default)
python agents/q_learning/train.py --config experiments/configs/q_learning_lunarlander_default.json

# Single experiment (LunarLander modified)
python agents/q_learning/train.py --config experiments/configs/q_learning_lunarlander_modified.json

# Run ALL Q-Learning experiments
chmod +x experiments/scripts/run_q_learning.sh  # First time only
./experiments/scripts/run_q_learning.sh
```

## Analyzing Results

```bash
# Compare Q-Learning results
python experiments/scripts/compare_q_learning.py

# Open Jupyter notebook
jupyter notebook notebooks/q_learning_quickstart.ipynb
```

## Git Commands

```bash
# Check status
git status

# Add all changes
git add .

# Commit changes
git commit -m "Your message here"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main

# Create new branch
git checkout -b feature/your-feature-name

# Switch branches
git checkout main
```

## Results Location

```bash
# View logs
ls -la results/q_learning/logs/

# View plots
ls -la results/q_learning/plots/

# View models
ls -la results/q_learning/models/

# View comparison tables
ls -la results/comparison/tables/
```

## Troubleshooting

```bash
# Check Python version (need 3.10+)
python --version

# Check if in virtual environment (should show venv path)
which python

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Install specific environment support
pip install gymnasium[box2d]  # LunarLander
pip install gymnasium[atari]  # Atari games
pip install gymnasium[all]    # Everything

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Quick File Locations

```bash
# Q-Learning implementation
agents/q_learning/q_table.py
agents/q_learning/train.py
agents/q_learning/utils.py

# Environment module
env/base_env.py
env/reward_wrapper.py

# Utilities
utils/logging.py
utils/plotting.py
utils/metrics.py

# Configurations
experiments/configs/*.json

# Tests
tests/test_environment.py
tests/test_q_learning.py

# Documentation
README.md
SETUP.md
IMPLEMENTATION_SUMMARY.md
agents/q_learning/README.md
```

## Common Tasks

### Run experiment with custom hyperparameters

1. Copy config file: `cp experiments/configs/q_learning_cartpole_default.json experiments/configs/my_custom.json`
2. Edit: `nano experiments/configs/my_custom.json`
3. Run: `python agents/q_learning/train.py --config experiments/configs/my_custom.json`

### View training progress

```bash
# Watch log file in real-time
tail -f results/q_learning/logs/q_learning_cartpole_default_*.csv
```

### Quick test environment

```bash
python -c "from env import load_environment; env = load_environment('CartPole-v1'); print(env.observation_space, env.action_space)"
```

### Load and test trained model

```python
from env import load_environment
from agents.q_learning.q_table import QTable
from agents.q_learning.utils import evaluate_policy

env = load_environment("CartPole-v1")
q_table = QTable(num_actions=2, num_bins=[10,10,10,10], state_bounds=[(-2.4,2.4),(-3,3),(-0.5,0.5),(-2,2)])
q_table.load("results/q_learning/models/q_learning_cartpole_default_final.pkl")
evaluate_policy(env, q_table, num_episodes=5, render=False)
```

## Python Quick Tests

```bash
# Test import
python -c "import gymnasium; import numpy; import torch; print('All imports OK')"

# Test environment
python -c "from env import load_environment; e = load_environment('CartPole-v1'); print('Environment OK')"

# Test Q-Table
python -c "from agents.q_learning.q_table import QTable; q = QTable(2); print('Q-Table OK')"
```

## Useful Aliases (add to ~/.zshrc)

```bash
alias rlenv='source ~/Desktop/An_III/final_projs/RL_proj/venv/bin/activate'
alias rlcd='cd ~/Desktop/An_III/final_projs/RL_proj'
alias rltest='python tests/test_environment.py && python tests/test_q_learning.py'
alias rlrun='./experiments/scripts/run_q_learning.sh'
```

After adding aliases, run: `source ~/.zshrc`

Then you can use:
```bash
rlcd    # Go to project
rlenv   # Activate environment
rltest  # Run tests
rlrun   # Run all experiments
```
