# 🚀 QUICK START GUIDE - DQN for Taxi-v3

**Person C (Ingrid)**  
**5-Minute Setup & Run**

---

## Step 1: Install (2 minutes)

```bash
cd ~/Desktop/An_III/final_projs/RL_proj
source .venv/bin/activate  # Or create if not exists

# Install all dependencies
pip install gymnasium numpy torch matplotlib pandas seaborn tensorboard
```

---

## Step 2: Test (1 minute)

```bash
# Verify everything works
python agents/dqn/test_dqn.py
```

✅ Should show: "ALL TESTS PASSED!"

---

## Step 3: Run Quick Experiment (2 minutes)

```python
# Run this in Python or Jupyter
from agents.dqn.experiments import run_baseline_experiment

# This runs DQN with 3 seeds, 2000 episodes each
results = run_baseline_experiment()

# Results automatically saved to: ./experiments_baseline/
```

**Or from command line:**
```bash
python agents/dqn/experiments.py
# Choose option: 1 (baseline)
```

---

## Step 4: Visualize (30 seconds)

```bash
python agents/dqn/visualize.py
```

✅ Plots saved to: `./experiments/plots/`

---

## 🎯 What You Get

After running the quick experiment:

1. **Models**: `./experiments_baseline/*/models/`
   - `best_model_seed_42.pth`
   - `best_model_seed_123.pth`
   - `best_model_seed_456.pth`
   - `final_model_seed_*.pth` (x3)

2. **Results**: `./experiments_baseline/*/logs/`
   - `results_seed_42.json`
   - `results_seed_123.json`
   - `results_seed_456.json`
   - `results_all_seeds.json`

3. **Plots**: `./experiments_baseline/*/plots/`
   - `multi_seed_results.png` (training curves + comparison)

---

## 📊 Expected Performance

**After 2000 episodes:**
- Mean reward: +7 to +10
- Success rate: 70-90%
- Training time: ~30-60 minutes (3 seeds)

**Baseline metrics:**
- Random policy: -250 to -500
- Your DQN: +7 to +10 ✅

---

## 🔬 Run Full Experiments (Optional)

**All experiments (2-3 hours):**
```bash
python agents/dqn/experiments.py
# Choose option: 4 (all experiments)
```

This runs:
1. Baseline DQN (3 seeds)
2. Hyperparameter sweep (5 configs × 3 seeds)
3. Double DQN comparison (2 configs × 3 seeds)

---

## 🤝 Compare with Q-Learning

```python
from agents.dqn.visualize import DQNVisualizer, load_experiment_results

# Load your DQN results
dqn_results = load_experiment_results('./experiments_baseline/baseline_dqn_...')

# Get Q-Learning results from Irina
qlearning_results = {
    'rewards': [...],  # Training rewards
    'final_reward': 8.5,  # Final evaluation
    'final_std': 1.2
}

# Create comparison plot
viz = DQNVisualizer('./comparison_plots')
viz.plot_dqn_vs_qlearning(dqn_results, qlearning_results)
```

---

## 📚 Documentation

- **Full guide**: `agents/dqn/README.md`
- **Complete status**: `PROIECT_ECHIPA/INGRID_COMPLETE.md`
- **Task checklist**: `PROIECT_ECHIPA/INGRID_DQN_TASKS.md`

---

## ✅ Verification Checklist

Before experiments:
- [ ] Dependencies installed
- [ ] Test script passes (`test_dqn.py`)
- [ ] No import errors

After experiments:
- [ ] Results files exist (JSON)
- [ ] Models saved (PTH files)
- [ ] Plots generated (PNG files)
- [ ] Mean reward > +7

---

## 💡 Troubleshooting

**Problem: Import errors**
```bash
pip install -r agents/dqn/requirements.txt
```

**Problem: Test fails**
- Check gymnasium is installed: `pip install gymnasium`
- Check torch is installed: `pip install torch`

**Problem: Slow training**
- Normal! DQN takes 30-60 min for full run
- Start with fewer episodes to test: change `num_episodes=500`

**Problem: Poor performance**
- Run hyperparameter sweep to find better config
- Check epsilon decay (may need slower: 0.997 or 0.998)

---

## 🎯 Next Steps

1. ✅ Run baseline (this guide)
2. ⏳ Run full experiments (hyperparameter sweep)
3. ⏳ Compare with Irina's Q-Learning
4. ⏳ Write analysis (NO LLM TEXT!)
5. ⏳ Prepare presentation

---

**Good luck! 🚀**

Total time to complete: 5 minutes (setup) + 30-60 minutes (training)
