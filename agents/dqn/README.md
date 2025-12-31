# DQN Implementation for Taxi-v3
**Person C (Ingrid) - DQN Engineer**

Complete Deep Q-Network implementation with experiments and visualizations.

---

## 📁 Structure

```
agents/dqn/
├── __init__.py           # Module exports
├── network.py            # Neural network architectures (DQN, Dueling DQN)
├── replay_buffer.py      # Experience replay (basic, prioritized)
├── agent.py              # Complete DQN agent with Double DQN
├── train.py              # Training script
├── experiments.py        # Hyperparameter experiments
├── visualize.py          # Plotting and visualization utilities
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd agents/dqn
pip install -r requirements.txt
```

Or install from workspace root:
```bash
pip install gymnasium numpy torch matplotlib pandas seaborn tensorboard
```

### 2. Run Basic Training

```python
from agents.dqn import DQNAgent
import gymnasium as gym

# Create environment and agent
env = gym.make('Taxi-v3')
agent = DQNAgent(
    state_size=500,  # Taxi-v3 has 500 discrete states
    action_size=6,   # 6 actions
    use_double_dqn=True  # Enable Double DQN
)

# Training loop
for episode in range(2000):
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(200):
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, terminated or truncated)
        agent.train_step()
        
        episode_reward += reward
        state = next_state
        
        if terminated or truncated:
            break
    
    agent.decay_epsilon()
    print(f"Episode {episode+1}, Reward: {episode_reward}")
```

### 3. Run Experiments

```bash
# Run all experiments (baseline, hyperparameter sweep, Double DQN comparison)
python agents/dqn/experiments.py

# Or interactively choose experiments
python agents/dqn/experiments.py
# Then select: 1 (baseline), 2 (sweep), 3 (comparison), or 4 (all)
```

### 4. Visualize Results

```bash
# Automatically visualize all experiments
python agents/dqn/visualize.py
```

---

## 🧪 Experiments

### Baseline Experiment
```python
from agents.dqn.experiments import run_baseline_experiment
results = run_baseline_experiment()
```

Runs standard DQN with default hyperparameters:
- Learning rate: 1e-3
- Gamma: 0.99
- Epsilon decay: 0.995
- Buffer: 10,000
- Batch size: 64
- 3 random seeds
- 2000 episodes per seed

### Hyperparameter Sweep
```python
from agents.dqn.experiments import run_hyperparameter_sweep
comparison = run_hyperparameter_sweep()
```

Compares 5 configurations:
1. **Baseline**: Standard DQN
2. **Double DQN**: Uses Double DQN to reduce overestimation
3. **Larger Buffer**: 50,000 capacity (5x larger)
4. **Higher LR**: 5e-3 learning rate (5x faster)
5. **Slower Epsilon**: 0.998 decay (more exploration)

### Double DQN Comparison
```python
from agents.dqn.experiments import run_double_dqn_comparison
results = run_double_dqn_comparison()
```

Direct comparison: Standard DQN vs Double DQN

---

## 📊 Visualization

### Training Curves
```python
from agents.dqn.visualize import DQNVisualizer

viz = DQNVisualizer(output_dir='./my_plots')
viz.plot_training_curves(results, title="My Experiment")
```

Creates:
- Raw + smoothed reward curves (all seeds)
- Final evaluation comparison (bar plot with error bars)

### Multi-Experiment Comparison
```python
viz.plot_comparison([exp1, exp2, exp3], title="Config Comparison")
```

Creates:
- Side-by-side bar plot with error bars
- Training curves overlay (all experiments)

### DQN vs Q-Learning
```python
viz.plot_dqn_vs_qlearning(dqn_results, qlearning_results)
```

Compares deep RL (DQN) with tabular method (Q-Learning from Irina).

### Results Table
```python
df = viz.create_summary_table([exp1, exp2, exp3])
```

Exports CSV with all metrics (mean, std, best/worst seed, hyperparameters).

---

## ⚙️ Hyperparameters

### DQN Agent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_size` | 500 | Number of discrete states (Taxi-v3) |
| `action_size` | 6 | Number of actions (Taxi-v3) |
| `learning_rate` | 1e-3 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Epsilon decay per episode |
| `buffer_capacity` | 10000 | Replay buffer size |
| `batch_size` | 64 | Training batch size |
| `target_update_freq` | 100 | Steps between target network updates |
| `use_double_dqn` | False | Enable Double DQN |
| `hidden_dims` | [128, 64] | Hidden layer sizes |
| `embedding_dim` | 64 | State embedding dimension |

### Tuning Guide

**Learning Rate:**
- Too high (>5e-3): Unstable, oscillating loss
- Too low (<1e-4): Very slow learning
- Sweet spot: 1e-3 to 5e-3

**Epsilon Decay:**
- Fast decay (0.99): Exploits early, may get stuck
- Slow decay (0.998): More exploration, slower convergence
- Sweet spot: 0.995 to 0.997

**Buffer Size:**
- Small (<5000): Correlated samples, overfitting
- Large (>50000): Slower sampling, more memory
- Sweet spot: 10,000 to 20,000

**Batch Size:**
- Small (<32): Noisy gradients, unstable
- Large (>128): Slower updates, less diversity
- Sweet spot: 64 to 128

**Double DQN:**
- Use when: Q-values seem overestimated
- Skip when: Standard DQN already works well

---

## 🧠 Neural Network Architecture

### Standard DQN (`network.py`)
```
Input (state: discrete 0-499)
  ↓
Embedding Layer (64 dims)
  ↓
Dense (128 units, ReLU)
  ↓
Dense (64 units, ReLU)
  ↓
Output (6 Q-values)
```

### Dueling DQN (optional)
```
Input (state: discrete 0-499)
  ↓
Embedding Layer (64 dims)
  ↓
Dense (128 units, ReLU)
  ↓
     ┌─────────────────────┐
     │                     │
Value Stream         Advantage Stream
Dense (64, ReLU)     Dense (64, ReLU)
Dense (1)            Dense (6)
     │                     │
     └─────────────────────┘
              ↓
    Q(s,a) = V(s) + A(s,a) - mean(A)
```

---

## 💾 Model Checkpointing

### Save Model
```python
agent.save('my_model.pth')
```

Saves:
- Policy network weights
- Target network weights
- Optimizer state
- Epsilon value
- Training steps
- Full config

### Load Model
```python
agent = DQNAgent(state_size=500, action_size=6)
agent.load('my_model.pth')
```

Restores complete training state for resuming or evaluation.

---

## 📈 Monitoring Training

### Get Statistics
```python
stats = agent.get_stats()
print(f"Epsilon: {stats['epsilon']:.4f}")
print(f"Steps: {stats['steps']}")
print(f"Episodes: {stats['episodes']}")
print(f"Buffer: {stats['buffer_size']}")
```

### TensorBoard (optional)
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/dqn_experiment')

for episode in range(num_episodes):
    # ... training code ...
    writer.add_scalar('Reward/train', episode_reward, episode)
    writer.add_scalar('Loss/train', loss, agent.steps)
    writer.add_scalar('Epsilon', agent.epsilon, episode)

writer.close()
```

Run: `tensorboard --logdir=runs`

---

## 🔬 Comparison with Q-Learning

### Expected Performance

**Q-Learning (Tabular - Irina):**
- ✅ Fast convergence (1000 episodes)
- ✅ Guaranteed convergence (small state space)
- ❌ No generalization to new states
- ❌ Doesn't scale to large state spaces

**DQN (Deep RL - Ingrid):**
- ✅ Can handle large/continuous state spaces
- ✅ Learns representations (embeddings)
- ✅ Experience replay breaks correlation
- ❌ Slower convergence (2000+ episodes)
- ❌ Less stable (requires tuning)

### When to Use Each

| Scenario | Q-Learning | DQN |
|----------|------------|-----|
| Small state space (<1000) | ✅ Preferred | ⚠️ Overkill |
| Large state space (>10,000) | ❌ Memory | ✅ Scales |
| Continuous states | ❌ Need discretization | ✅ Native |
| Fast prototyping | ✅ Simple | ❌ Complex |
| Sample efficiency | ⚠️ On-policy | ✅ Off-policy |

---

## 🐛 Troubleshooting

### Problem: Loss Exploding
**Solution:** Lower learning rate (try 5e-4), increase gradient clipping

### Problem: No Learning After 500 Episodes
**Solution:** Check epsilon decay (too fast?), increase buffer size

### Problem: Unstable Q-values
**Solution:** Enable Double DQN, tune target update frequency

### Problem: Import Errors
**Solution:**
```bash
pip install -r requirements.txt
# Or individually:
pip install gymnasium torch numpy matplotlib
```

### Problem: CUDA Out of Memory
**Solution:** Agent automatically uses CPU. To force CPU:
```python
agent = DQNAgent(state_size=500, action_size=6, device='cpu')
```

---

## 📚 References

**DQN Paper:**
Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
https://www.nature.com/articles/nature14236

**Double DQN Paper:**
van Hasselt et al. (2015) - "Deep Reinforcement Learning with Double Q-learning"
https://arxiv.org/abs/1509.06461

**Dueling DQN Paper:**
Wang et al. (2016) - "Dueling Network Architectures for Deep Reinforcement Learning"
https://arxiv.org/abs/1511.06581

**Labs:**
- Lab 5: Q-Learning and DQN basics
- Lab 6: Advanced DQN (Dueling, PER, Noisy, Rainbow)

---

## ✅ Checklist for Report

- [ ] Run baseline DQN (3 seeds)
- [ ] Run hyperparameter sweep (5 configs)
- [ ] Compare with Irina's Q-Learning results
- [ ] Generate all plots (training, comparison, DQN vs Q-Learning)
- [ ] Create results table (CSV)
- [ ] Document best hyperparameters
- [ ] Explain Double DQN performance difference
- [ ] Discuss when to use DQN vs tabular methods
- [ ] NO LLM-generated text in report! (write analysis yourself)

---

## 👥 Team Collaboration

**Person A (Iustin):** Extended Taxi-v3 environment
**Person B (Irina):** Q-Learning implementation
**Person C (Ingrid):** DQN implementation (this module)
**Person D (Matei):** REINFORCE implementation

Share your results:
```python
# Export results for team
import json
with open('dqn_results_for_team.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 🎯 Next Steps

1. ✅ Complete DQN implementation
2. ✅ Run baseline experiments
3. ✅ Tune hyperparameters
4. ⏳ Compare with Irina's Q-Learning
5. ⏳ Try Dueling DQN (optional)
6. ⏳ Try Prioritized Experience Replay (optional)
7. ⏳ Write analysis for report (NO LLM TEXT!)

---

**Questions?** Check Labs 5 & 6 or ask team members!

**Good luck! 🚀**
