#  RL PROJECT - Quick Start Guide

**Echipă:** Irina, Ingrid, Matei, Iustin  
**Environment:** TBD (CartPole / LunarLander / MountainCar)  
**Deadline:** Săptămâna 14 (ianuarie 2026)

---

##  STRUCTURĂ PROIECT

```
RL_PROJECT/

 README.md                          # Documentație principală
 requirements.txt                   # Dependențe Python
 .gitignore                        # Fișiere ignorate de git

 env/                              # Environment + wrappers
    __init__.py
    base_env.py                  # Load environment default
    reward_wrapper.py            # Wrapper pentru reward modificat
    README.md                    # Documentație environment

 agents/                           # Algoritmi RL
    __init__.py
   
    q_learning/                  # Agent #1 - Q-Learning (Ingrid)
       __init__.py
       train.py                 # Script training
       q_table.py               # Implementare Q-table
       utils.py                 # Helper functions
       README.md
   
    dqn/                         # Agent #2 - Deep Q-Network (Matei)
       __init__.py
       train.py
       model.py                 # Neural network
       replay_buffer.py         # Experience replay
       utils.py
       README.md
   
    ppo/                         # Agent #3 - PPO (Iustin)
        __init__.py
        train.py
        model.py                 # Actor-Critic network
        utils.py
        README.md

 experiments/                      # Configurații experimente
    configs/
       q_learning_default.json
       q_learning_modified.json
       dqn_default.json
       dqn_modified.json
       ppo_default.json
       ppo_modified.json
   
    scripts/
        run_all_experiments.sh   # Script pentru rulare automată
        compare_results.py       # Script comparație rezultate

 results/                          # Output experimente
    q_learning/
       logs/                    # CSV files cu reward-uri
       plots/                   # Grafice learning curves
       models/                  # Q-tables salvate
   
    dqn/
       logs/
       plots/
       models/                  # Neural network weights
   
    ppo/
       logs/
       plots/
       models/
   
    comparison/                   # Comparații între algoritmi
        tables/                  # CSV cu metrici comparative
        plots/                   # Grafice comparative

 notebooks/                        # Jupyter notebooks (explorare)
    01_environment_exploration.ipynb
    02_q_learning_debug.ipynb
    03_dqn_visualization.ipynb
    04_final_analysis.ipynb

 docs/                             # Documentație proiect
    presentation.pptx            # PowerPoint prezentare
    environment_analysis.md      # Analiză environment
    algorithms_overview.md       # Explicații algoritmi
    experiments_report.md        # Raport experimente
    images/                      # Imagini pentru docs

 tests/                            # Unit tests (optional, dar nice)
    test_environment.py
    test_q_learning.py
    test_dqn.py

 utils/                            # Utilities comune
     __init__.py
     logging.py                   # Setup logging
     plotting.py                  # Funcții plotting
     seed.py                      # Seed management
     metrics.py                   # Calculare metrici
```

---

##  SETUP INIȚIAL

### 1. Creați repo GitHub
```bash
# Pe GitHub.com
# New Repository → RL_Project
# Public sau Private (la alegere)

# Local
git clone https://github.com/your-username/RL_Project.git
cd RL_Project
```

### 2. Setup Python environment
```bash
# Cu conda (recomandat)
conda create -n rl_project python=3.10
conda activate rl_project

# Sau cu venv
python3 -m venv rl_project_env
source rl_project_env/bin/activate  # Mac/Linux
# rl_project_env\Scripts\activate  # Windows
```

### 3. Instalați dependențe
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```txt
# Core
gymnasium==0.29.1
numpy==1.24.3
matplotlib==3.7.1
pandas==2.0.3

# Deep Learning (pentru DQN & PPO)
torch==2.0.1
tensorboard==2.13.0

# Optional (dar recomandat)
stable-baselines3==2.1.0  # Pentru PPO (opțional)
wandb==0.15.5             # Logging advanced (opțional)
pillow==10.0.0            # Pentru imagini
seaborn==0.12.2           # Plotting frumos
tqdm==4.65.0              # Progress bars
```

### 4. Verificați instalarea
```python
# test_installation.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

print(" Gymnasium:", gym.__version__)
print(" NumPy:", np.__version__)
print(" Matplotlib:", matplotlib.__version__)
print(" PyTorch:", torch.__version__)

# Test environment
env = gym.make("CartPole-v1")
print(" Environment loading works!")
env.close()
```

```bash
python test_installation.py
```

---

##  `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
rl_project_env/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs & Results (optional, dacă sunt mari)
# results/*/logs/*.csv
# results/*/models/*.pth

# Temporary
*.log
*.tmp
temp/
```

---

##  PRIMII PAȘI (după setup)

### 1. Test environment (toată echipa)
```python
# test_env.py
import gymnasium as gym

env_name = "CartPole-v1"  # sau LunarLander-v2, MountainCar-v0
env = gym.make(env_name, render_mode="human")

print(f" Testing {env_name}")
print(f"State space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Run random agent
state, info = env.reset(seed=42)
total_reward = 0

for step in range(200):
    action = env.action_space.sample()  # random action
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"Episode finished after {step+1} timesteps")
        print(f"Total reward: {total_reward}")
        break

env.close()
```

### 2. Implementați wrapper reward (Irina + Ingrid)
```python
# env/reward_wrapper.py
import gymnasium as gym
from gymnasium import Wrapper

class CustomRewardWrapper(Wrapper):
    """
    Wrapper pentru modificarea reward-ului.
    Exemplu pentru CartPole: penalizăm oscilații.
    """
    def __init__(self, env, reward_type="default"):
        super().__init__(env)
        self.reward_type = reward_type
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        if self.reward_type == "penalize_oscillations":
            # Penalizăm viteza mare (oscilații)
            cart_velocity = state[1]
            angular_velocity = state[3]
            reward -= 0.1 * abs(angular_velocity)
            reward -= 0.05 * abs(cart_velocity)
        
        elif self.reward_type == "bonus_center":
            # Bonus pentru poziție centrală
            cart_position = state[0]
            pole_angle = state[2]
            reward += 0.5 * (1 - abs(cart_position) / 2.4)
            reward += 0.5 * (1 - abs(pole_angle) / 0.418)
        
        # Adaugă alte tipuri de reward aici
        
        return state, reward, terminated, truncated, info

# Usage:
# env = gym.make("CartPole-v1")
# env = CustomRewardWrapper(env, reward_type="penalize_oscillations")
```

### 3. Implementați logging utils
```python
# utils/logging.py
import csv
import os

class Logger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"{experiment_name}.csv")
        
        # Creează fișier CSV
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Reward', 'Steps', 'Epsilon'])
    
    def log(self, episode, reward, steps, epsilon=None):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, steps, epsilon])
    
    def log_dict(self, data_dict):
        """Pentru logging mai complex"""
        pass  # implementare
```

---

##  PLOTTING UTILS

```python
# utils/plotting.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_learning_curve(log_file, save_path, title="Learning Curve", window=100):
    """
    Plotează learning curve din CSV log.
    
    Args:
        log_file: path la CSV cu coloane [Episode, Reward, Steps, ...]
        save_path: unde să salveze imaginea
        title: titlul graficului
        window: dimensiunea ferestrei pentru moving average
    """
    df = pd.read_csv(log_file)
    
    episodes = df['Episode'].values
    rewards = df['Reward'].values
    
    # Moving average
    rewards_smooth = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label='Raw reward')
    plt.plot(episodes, rewards_smooth, linewidth=2, label=f'Moving avg (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved plot: {save_path}")

def plot_comparison(log_files, labels, save_path, title="Algorithm Comparison"):
    """
    Compară learning curves pentru mai mulți algoritmi.
    
    Args:
        log_files: list de paths la CSV files
        labels: list de labels pentru fiecare algoritm
        save_path: unde să salveze
        title: titlu
    """
    plt.figure(figsize=(12, 7))
    
    for log_file, label in zip(log_files, labels):
        df = pd.read_csv(log_file)
        episodes = df['Episode'].values
        rewards = df['Reward'].values
        rewards_smooth = pd.Series(rewards).rolling(window=100, min_periods=1).mean()
        plt.plot(episodes, rewards_smooth, linewidth=2, label=label)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved comparison plot: {save_path}")
```

---

##  EXPERIMENT CONFIG TEMPLATE

```json
# experiments/configs/q_learning_default.json
{
  "experiment_name": "q_learning_cartpole_default",
  "algorithm": "Q-Learning",
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
    "num_episodes": 5000
  },
  "discretization": {
    "enabled": true,
    "bins": {
      "cart_position": 10,
      "cart_velocity": 10,
      "pole_angle": 10,
      "pole_velocity": 10
    }
  },
  "logging": {
    "log_interval": 100,
    "save_model": true,
    "save_interval": 1000
  },
  "seed": 42
}
```

---

##  WORKFLOW TIPIC

### Pentru Ingrid (Q-Learning):

1. **Setup:**
   ```bash
   cd RL_PROJECT/agents/q_learning
   ```

2. **Develop:**
   - Scrie cod în `train.py`, `q_table.py`
   - Testează cu script mic

3. **Train:**
   ```bash
   python train.py --config ../../experiments/configs/q_learning_default.json
   ```

4. **Analiză:**
   ```bash
   python ../../utils/plotting.py --log results/q_learning/logs/default.csv
   ```

5. **Commit:**
   ```bash
   git add agents/q_learning/
   git commit -m "Implement Q-Learning baseline"
   git push
   ```

---

##  DOCUMENTAȚIE TEMPLATE

### `README.md` principal:

```markdown
#  Reinforcement Learning Project

**Echipă:** Irina, Ingrid, Matei, Iustin  
**Universitate:** București, Facultatea Matematică-Informatică  
**Curs:** Introducere în Reinforcement Learning (2025-2026)

##  Despre Proiect

Acest proiect compară trei algoritmi de Reinforcement Learning pe environment-ul [CartPole-v1]:
- **Q-Learning** (tabular, value-based)
- **Deep Q-Network (DQN)** (deep, value-based)
- **Proximal Policy Optimization (PPO)** (policy-based)

Scopul este să analizăm efectul **reward shaping** asupra performanței fiecărui algoritm.

##  Obiective

1. Implementarea a 3 algoritmi RL
2. Modificarea reward-ului (reward shaping)
3. Compararea rezultatelor
4. Analiză experimentală

##  Setup

[Vezi QUICK_START.md]

##  Rezultate

[Adaugă la final]

##  Contribuții

- **Irina:** Environment design, reward shaping, coordonare
- **Ingrid:** Q-Learning implementation, analiză comparativă
- **Matei:** DQN implementation, neural networks
- **Iustin:** PPO implementation, experimente

##  Referințe

- Sutton & Barto - Reinforcement Learning: An Introduction
- Gymnasium Documentation
- Stable Baselines3
```

---

##  CHECKLIST ÎNAINTE DE PREZENTARE

- [ ] Toate cele 3 algoritmi rulează
- [ ] Rezultate salvate pentru reward default + modified
- [ ] Grafice generate și frumoase
- [ ] Tabel comparativ complet
- [ ] README.md actualizat
- [ ] Cod comentat
- [ ] PowerPoint finalizat
- [ ] Fiecare membru știe ce prezintă
- [ ] Demo video/gif pregătit (optional)
- [ ] Rehearsal făcut (minim 2x)

---

**GOOD LUCK! **

**Questions?** Check `01_PLAN_PROIECT.md` or ask Irina!
