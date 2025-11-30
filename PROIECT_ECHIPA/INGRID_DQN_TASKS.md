# 🎯 INGRID'S TASKS - DQN Implementation for Taxi-v3

**Role:** Person C - Deep Value-Based RL Engineer  
**Last Updated:** November 30, 2025

---

## 📋 YOUR RESPONSIBILITIES

### 1. **DQN Implementation** (Labs 5 & 6)
- [ ] Implement DQN architecture for Taxi-v3
- [ ] Design neural network for discrete state space
  - Input: State embedding (one-hot or learned embedding)
  - Output: Q-values for 6 actions (North, South, East, West, Pickup, Dropoff)
- [ ] Implement experience replay buffer
- [ ] Implement target network with periodic updates
- [ ] Add epsilon-greedy exploration

### 2. **State Representation**
- [ ] Decide on state encoding:
  - Option 1: One-hot encoding of discrete state (500 states)
  - Option 2: Learned embedding layer
  - Option 3: Feature vector (taxi_row, taxi_col, passenger_loc, destination)
- [ ] Handle extended Taxi features (fuel, extra passengers, traffic)

### 3. **Network Architecture Design**
```python
# Example architecture to implement:
State (discrete index) → Embedding Layer → 
Hidden Layers (128, 64) → Output (6 Q-values)
```
- [ ] Tune network size (start small: 128-64-32)
- [ ] Choose activation functions (ReLU recommended)
- [ ] Implement proper initialization

### 4. **DQN Components**
- [ ] **Replay Buffer:**
  - Store (state, action, reward, next_state, done)
  - Implement sampling with batch_size=32 or 64
  
- [ ] **Target Network:**
  - Copy main network every N steps (e.g., every 100 steps)
  - Stabilizes training
  
- [ ] **Loss Function:**
  - MSE between Q(s,a) and target: r + γ * max_a' Q_target(s', a')

### 5. **Hyperparameter Tuning**
- [ ] Learning rate: Try [1e-3, 5e-4, 1e-4]
- [ ] Gamma (discount): 0.99 (standard)
- [ ] Epsilon schedule: Start 1.0 → decay to 0.01
- [ ] Replay buffer size: 10000 or 50000
- [ ] Batch size: 32 or 64
- [ ] Target network update frequency: 100-500 steps

### 6. **Extensions (Optional - Lab 6)**
If you have time, try ONE of these:
- [ ] **Dueling DQN:** Separate value and advantage streams
- [ ] **Double DQN:** Use main network to select action, target to evaluate
- [ ] **Prioritized Experience Replay:** Sample important transitions more

### 7. **Experiments & Comparison**
- [ ] **Baseline:** Run DQN on original Taxi-v3
- [ ] **Extended:** Run DQN on Iustin's extended Taxi
- [ ] **Compare with Irina's Q-learning:**
  - Training speed (episodes to convergence)
  - Sample efficiency
  - Final performance
- [ ] **Ablation studies:**
  - DQN without replay buffer
  - DQN without target network
  - Different network sizes

### 8. **Visualization & Logs**
- [ ] Plot training curves:
  - Episode reward over time
  - Loss over time
  - Epsilon decay
- [ ] Create comparison plots:
  - DQN vs Q-learning vs REINFORCE
- [ ] Log key metrics:
  - Average reward per 100 episodes
  - Steps per episode
  - Convergence point

### 9. **Documentation (Your Part of Report)**
Write these sections (NO LLM TEXT!):
- [ ] **DQN Theory:**
  - Explain replay buffer
  - Explain target network
  - Why these help with stability
- [ ] **Network Architecture:**
  - Diagram of your network
  - Justification for design choices
- [ ] **Results:**
  - Training curves
  - Comparison with tabular methods
  - Analysis: Why DQN works better/worse in certain scenarios
- [ ] **Challenges:**
  - What was difficult to tune
  - What hyperparameters mattered most

---

## 🗓️ TIMELINE

### Week 1 (Dec 2-5) - Setup & Basic DQN
- [ ] Study Labs 5 & 6 thoroughly
- [ ] Understand Taxi-v3 environment (coordinate with Iustin)
- [ ] Implement basic DQN (no extensions yet)
- [ ] Test on original Taxi-v3

### Week 2 (Dec 9-12) - Extended Taxi & Tuning
- [ ] Wait for Iustin's extended Taxi wrapper
- [ ] Adapt DQN to handle extensions
- [ ] Hyperparameter tuning
- [ ] Compare with Irina's Q-learning results

### Week 3 (Dec 16-19) - Experiments & Analysis
- [ ] Run full experiments (multiple seeds)
- [ ] Generate all plots
- [ ] Start writing your report sections
- [ ] Coordinate comparison plots with team

### Week 4 (Dec 23-26) - Documentation
- [ ] Finish report sections
- [ ] Prepare slides for your part
- [ ] Practice explaining DQN
- [ ] Final code cleanup

---

## 📚 RESOURCES TO STUDY

### Must Read:
- **Lab 5:** Q-learning to DQN transition
- **Lab 6:** DQN improvements (Dueling, Double, PER)
- **Course 3:** Deep RL basics
- **DQN Paper:** Mnih et al. (2015) - Nature paper

### Key Concepts to Understand:
1. **Why neural networks for Q-values?**
   - Function approximation for large state spaces
   - Generalization to unseen states

2. **Why experience replay?**
   - Breaks correlation between consecutive samples
   - More efficient use of data

3. **Why target network?**
   - Stabilizes training
   - Prevents chasing a moving target

4. **Taxi-v3 specifics:**
   - State space: 500 discrete states (5x5 grid × 5 passenger locations × 4 destinations)
   - Action space: 6 discrete actions
   - Rewards: +20 for successful delivery, -1 per step, -10 for illegal pickup/dropoff

---

## 🤝 COORDINATION WITH TEAM

### With Iustin (Person A - Environment):
- Get extended Taxi wrapper early (Week 2)
- Understand new state space dimensions
- Test that DQN works on extended version

### With Irina (Person B - Tabular):
- Share results format (reward curves, metrics)
- Discuss state encoding strategies
- Run experiments with same seeds for fair comparison

### With Matei (Person D - Policy):
- Coordinate on comparison plots
- Share logging/plotting utilities
- Discuss which environment extensions challenge each algorithm most

---

## ✅ SUCCESS CRITERIA

You'll know you're done when:
- [ ] DQN trains successfully on Taxi-v3
- [ ] Clear improvement over random policy
- [ ] Can explain every hyperparameter choice
- [ ] Have comparison plots with Q-learning
- [ ] Report sections written in your own words
- [ ] Can present and answer questions about DQN

---

## 🆘 WHEN TO ASK FOR HELP

Ask team/professor if:
- DQN loss explodes or doesn't decrease
- Can't match Q-learning performance
- Unsure how to encode extended Taxi states
- Need help with PyTorch/TensorFlow
- Comparison seems unfair (different training setups)

---

## 🎯 PRIORITY ORDER

1. **HIGHEST:** Basic DQN working on original Taxi-v3
2. **HIGH:** Proper logging and comparison with Q-learning
3. **MEDIUM:** DQN on extended Taxi
4. **LOW:** Extensions (Dueling/Double DQN)
5. **LOWEST:** Advanced visualizations

**Remember:** Better to have solid basic DQN than buggy advanced DQN!
