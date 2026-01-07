# Reinforcement Learning Project

A clean, didactic implementation of RL algorithms for a custom Highway-env based environment.

## Quick Start

```bash
# Train agents
python train_tabular.py      # Tabular Q-Learning (fastest)
python train_reinforce.py    # Policy Gradient (best results)
python train_dqn.py          # Deep Q-Network

# Plot results
python plot_results.py
```

## Project Structure

```
├── custom_env.py          # Custom Highway-env environment
├── train_tabular.py       # Tabular Q-Learning training
├── train_dqn.py           # DQN training  
├── train_reinforce.py     # REINFORCE training
├── plot_results.py        # Visualize training curves
├── env/                   # Environment wrappers
│   ├── __init__.py
│   └── wrappers.py        # DiscreteAction, FlatObs, TabularObs
├── agents/
│   └── dqn/               # DQN implementation
│       ├── agent.py       # DQNAgent class
│       ├── network.py     # Neural network
│       └── replay_buffer.py
└── results/               # Training outputs (CSV, plots)
```

## Algorithms

### 1. Tabular Q-Learning
- Uses a table to store Q(state, action) values
- State space is discretized into bins
- Fast training (~500 episodes)

### 2. DQN (Deep Q-Network)
- Neural network approximates Q-values
- Experience replay for stable learning
- Target network for stable targets
- ~200 episodes

### 3. REINFORCE
- Directly learns a policy π(a|s)
- Monte Carlo returns (full episode)
- Often the best performer (~1000 episodes)

## Environment

`MyCustomEnv` is a Highway-env based environment with:
- **Observations**: Vehicle kinematics + fuel + passenger info + target direction
- **Actions**: Discrete (Idle, Accelerate, Brake, Left, Right)
- **Goal**: Pick up passengers and deliver them while managing fuel
