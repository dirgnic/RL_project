# Gymnasium Environments - Complete Guide & Recommendations

## 📊 Environment Categories Overview

### 1. **Classic Control** (Simple, Good for Learning)
### 2. **Box2D** (Physics-based, Medium Complexity)
### 3. **Toy Text** (Discrete, Tabular RL)
### 4. **MuJoCo** (Advanced Robotics - Requires License)
### 5. **Atari** (Deep RL, Image-based)

---

## 🎯 RECOMMENDED FOR YOUR PROJECT

### ⭐⭐⭐ **TOP 3 CHOICES** (Best for Academic Project)

#### 1. **CartPole-v1** ✅ CURRENTLY IMPLEMENTED
- **Type**: Classic Control
- **State Space**: Continuous (4D)
  - Cart position: [-4.8, 4.8]
  - Cart velocity: [-∞, ∞]
  - Pole angle: [-24°, 24°]
  - Pole angular velocity: [-∞, ∞]
- **Action Space**: Discrete (2) - Push left or right
- **Episode Length**: Max 500 steps
- **Success Criteria**: Balance pole for 500 steps (reward ≥ 475)
- **Difficulty**: ⭐ Easy
- **Installation**: `pip install gymnasium` ✅ Already working!

**Why Recommend:**
- ✅ Perfect for Q-Learning with discretization
- ✅ Fast training (converges in 500-1000 episodes)
- ✅ Easy to understand and visualize
- ✅ Good for demonstrating reward shaping
- ✅ No special dependencies

**Reward Shaping Ideas:**
- ✅ Already implemented: `bonus_center`, `penalize_oscillations`, `progressive_penalty`
- Bonus for keeping pole vertical
- Penalty for cart moving to edges
- Bonus for low velocity

---

#### 2. **MountainCar-v0** ✅ CURRENTLY IMPLEMENTED
- **Type**: Classic Control
- **State Space**: Continuous (2D)
  - Position: [-1.2, 0.6]
  - Velocity: [-0.07, 0.07]
- **Action Space**: Discrete (3) - Push left, no push, push right
- **Episode Length**: Max 200 steps
- **Success Criteria**: Reach flag at position ≥ 0.5
- **Difficulty**: ⭐⭐ Medium (sparse rewards!)
- **Installation**: `pip install gymnasium` ✅ Already working!

**Why Recommend:**
- ✅ Classic RL benchmark problem
- ✅ Demonstrates importance of reward shaping (default reward is -1 per step)
- ✅ Shows exploration challenges
- ✅ Great for comparing algorithms
- ✅ Small state space (easy discretization)

**Reward Shaping Ideas:**
- ✅ Already implemented: `height_based`, `momentum_based`, `potential_based`
- Reward for gaining height
- Reward for building momentum
- Potential-based shaping (most effective!)

---

#### 3. **LunarLander-v2** ⚠️ REQUIRES BOX2D
- **Type**: Box2D (Physics simulation)
- **State Space**: Continuous (8D)
  - Position (x, y)
  - Velocity (x, y)
  - Angle, angular velocity
  - Left leg contact, right leg contact
- **Action Space**: Discrete (4) - Do nothing, fire left, fire main, fire right
- **Episode Length**: Max 1000 steps
- **Success Criteria**: Land safely (reward ≥ 200)
- **Difficulty**: ⭐⭐⭐ Hard
- **Installation**: 
  ```bash
  brew install swig  # macOS
  pip install 'gymnasium[box2d]'
  ```

**Why Recommend:**
- ✅ Visually impressive (good for presentations!)
- ✅ Real-world physics
- ✅ Good for DQN and PPO
- ✅ Well-documented benchmark
- ⚠️ Harder to train with Q-Learning
- ⚠️ Requires Box2D installation

**Reward Shaping Ideas:**
- ✅ Already implemented: `fuel_efficiency`, `safety_first`
- Penalty for fuel consumption
- Bonus for soft landings
- Penalty for crashing
- Reward for staying centered

---

## 🎮 ALL CLASSIC CONTROL ENVIRONMENTS

### **Acrobot-v1**
- **State**: 6D (sin/cos of 2 joint angles + velocities)
- **Actions**: 3 (apply +1, 0, -1 torque)
- **Goal**: Swing tip above a line
- **Difficulty**: ⭐⭐ Medium
- **Good for**: Continuous state Q-Learning, DQN
- **Reward Shaping**: Height-based rewards, energy penalties

### **CartPole-v1** ✅ IMPLEMENTED
(See above)

### **MountainCar-v0** ✅ IMPLEMENTED
(See above)

### **MountainCarContinuous-v0**
- **State**: 2D (position, velocity)
- **Actions**: Continuous [-1, 1] (force)
- **Goal**: Reach flag
- **Difficulty**: ⭐⭐⭐ Hard
- **Good for**: PPO, DDPG, SAC
- **Recommendation**: Use discrete version (easier)

### **Pendulum-v1**
- **State**: 3D (cos(θ), sin(θ), angular velocity)
- **Actions**: Continuous [-2, 2] (torque)
- **Goal**: Keep pendulum upright
- **Difficulty**: ⭐⭐ Medium
- **Good for**: PPO, DDPG (continuous control)
- **Reward Shaping**: Angle and velocity penalties

---

## 📦 BOX2D ENVIRONMENTS (Requires Box2D)

### **LunarLander-v2** ✅ CONFIGS READY
(See above - Installation required)

### **LunarLanderContinuous-v2**
- Same as LunarLander but continuous actions
- **Difficulty**: ⭐⭐⭐⭐ Very Hard
- **Good for**: PPO, DDPG, TD3
- **Recommendation**: Use discrete version

### **BipedalWalker-v3**
- **State**: 24D (hull angle, velocities, joints, ground contact, etc.)
- **Actions**: 4 continuous (hip/knee motors)
- **Goal**: Walk forward without falling
- **Difficulty**: ⭐⭐⭐⭐⭐ Very Hard
- **Good for**: Advanced PPO, SAC
- **Recommendation**: Too complex for your project

### **BipedalWalkerHardcore-v3**
- Even harder version with obstacles
- **Recommendation**: Skip

### **CarRacing-v2**
- **State**: 96×96 RGB image
- **Actions**: 3 continuous (steering, gas, brake)
- **Difficulty**: ⭐⭐⭐⭐⭐ Very Hard
- **Good for**: Deep RL with CNNs
- **Recommendation**: Too complex

---

## 📝 TOY TEXT ENVIRONMENTS (Perfect for Tabular Q-Learning!)

### **FrozenLake-v1** ⭐ EXCELLENT FOR Q-LEARNING
- **State**: 16 discrete positions (4×4 grid)
- **Actions**: 4 (up, down, left, right)
- **Goal**: Reach goal without falling in holes
- **Difficulty**: ⭐ Easy (but slippery!)
- **Installation**: Built-in
- **Why Good**: Perfect for pure Q-Learning (no discretization needed!)

```python
import gymnasium as gym
env = gym.make('FrozenLake-v1', is_slippery=True)
```

**Variants:**
- `FrozenLake-v1` (4×4)
- `FrozenLake8x8-v1` (8×8, harder)

### **Taxi-v3** ⭐ GREAT FOR Q-LEARNING
- **State**: 500 discrete states (taxi location, passenger location, destination)
- **Actions**: 6 (move 4 directions, pickup, dropoff)
- **Goal**: Pick up passenger and drop at destination
- **Difficulty**: ⭐⭐ Medium
- **Why Good**: Perfect discrete environment, more complex than FrozenLake

### **CliffWalking-v0** ⭐ EXCELLENT FOR SARSA VS Q-LEARNING
- **State**: 48 discrete positions (4×12 grid)
- **Actions**: 4 (up, down, left, right)
- **Goal**: Navigate from start to goal avoiding cliff
- **Difficulty**: ⭐⭐ Medium
- **Why Good**: Classic for comparing SARSA and Q-Learning!

### **Blackjack-v1**
- **State**: 704 discrete states (player sum, dealer card, usable ace)
- **Actions**: 2 (hit, stand)
- **Difficulty**: ⭐⭐ Medium
- **Why Good**: Card game, different domain

---

## 🎮 ATARI ENVIRONMENTS (Requires ROM files)

**Installation**: `pip install 'gymnasium[atari]'` + `pip install 'gymnasium[accept-rom-license]'`

Popular Games:
- **Pong-v5**: Simple, good for DQN
- **Breakout-v5**: Classic benchmark
- **SpaceInvaders-v5**: More complex
- **MsPacman-v5**: Very complex

**Difficulty**: ⭐⭐⭐⭐ Very Hard  
**Good for**: Deep Q-Networks (DQN), CNN-based RL  
**Recommendation**: Too complex for your project (requires image processing)

---

## 🤖 MUJOCO ENVIRONMENTS (Requires License)

**Installation**: Requires MuJoCo license (free for students)

Popular Environments:
- Ant-v4, HalfCheetah-v4, Hopper-v4, Humanoid-v4, Walker2d-v4

**Difficulty**: ⭐⭐⭐⭐⭐ Very Hard  
**Good for**: Advanced continuous control research  
**Recommendation**: Skip (license required, very complex)

---

## 🎯 MY RECOMMENDATIONS FOR YOUR PROJECT

### **Option A: Simple & Fast** ⭐⭐⭐⭐⭐ BEST CHOICE
**Environments**: CartPole-v1 + MountainCar-v0  
**Why**: 
- ✅ Already implemented and working!
- ✅ Fast training (results in minutes)
- ✅ Clear demonstration of reward shaping
- ✅ Works perfectly with Q-Learning, DQN, PPO
- ✅ No installation issues
- ✅ Great for presentations

**Algorithms**:
1. Q-Learning (Ingrid) ✅
2. DQN (Matei)
3. PPO (Iustin)

### **Option B: Add Visual Impact** ⭐⭐⭐⭐
**Environments**: CartPole-v1 + MountainCar-v0 + LunarLander-v2  
**Why**:
- ✅ LunarLander looks impressive in presentations
- ✅ More complex physics
- ⚠️ Requires Box2D installation
- ⚠️ Slower training
- ⚠️ Q-Learning might struggle (need more episodes)

### **Option C: Pure Tabular RL** ⭐⭐⭐⭐
**Environments**: FrozenLake-v1 + Taxi-v3 + CliffWalking-v0  
**Why**:
- ✅ Perfect for pure Q-Learning (no discretization!)
- ✅ Fast training
- ✅ Clear state-action tables
- ✅ CliffWalking great for SARSA comparison
- ⚠️ Less visually impressive
- ⚠️ DQN might be overkill

### **Option D: Mixed Approach** ⭐⭐⭐
**Environments**: CartPole-v1 (continuous) + FrozenLake-v1 (discrete)  
**Why**:
- ✅ Shows both discretization and pure tabular
- ✅ Fast training on both
- ✅ Different types of problems

---

## 📊 COMPARISON TABLE

| Environment | State Space | Action Space | Difficulty | Training Time | Visual Appeal | Box2D? | Recommend |
|------------|-------------|--------------|------------|---------------|---------------|---------|-----------|
| **CartPole-v1** | 4D Continuous | 2 Discrete | ⭐ | ~5 min | ⭐⭐⭐ | No | ⭐⭐⭐⭐⭐ |
| **MountainCar-v0** | 2D Continuous | 3 Discrete | ⭐⭐ | ~10 min | ⭐⭐ | No | ⭐⭐⭐⭐⭐ |
| **LunarLander-v2** | 8D Continuous | 4 Discrete | ⭐⭐⭐ | ~30 min | ⭐⭐⭐⭐⭐ | Yes | ⭐⭐⭐⭐ |
| **FrozenLake-v1** | 16 Discrete | 4 Discrete | ⭐ | ~2 min | ⭐ | No | ⭐⭐⭐⭐ |
| **Taxi-v3** | 500 Discrete | 6 Discrete | ⭐⭐ | ~5 min | ⭐⭐ | No | ⭐⭐⭐ |
| **Acrobot-v1** | 6D Continuous | 3 Discrete | ⭐⭐ | ~15 min | ⭐⭐⭐ | No | ⭐⭐⭐ |
| **Pendulum-v1** | 3D Continuous | 1 Continuous | ⭐⭐ | ~10 min | ⭐⭐⭐ | No | ⭐⭐ |
| **Atari Games** | 210×160 Image | Varies | ⭐⭐⭐⭐ | Hours | ⭐⭐⭐⭐⭐ | No | ⭐ |

---

## 💡 FINAL RECOMMENDATION FOR YOUR PROJECT

### ✅ **STICK WITH CURRENT CHOICE**: CartPole-v1 + MountainCar-v0

**Reasons**:
1. ✅ **Already implemented** - you have working code!
2. ✅ **All tests passing** - proven to work
3. ✅ **Fast training** - quick iteration and experiments
4. ✅ **Great for all 3 algorithms**:
   - Q-Learning: Works perfectly with discretization ✅
   - DQN: Good continuous state approximation
   - PPO: Good for policy gradients
5. ✅ **Clear reward shaping demonstration**
6. ✅ **No dependency issues** (no Box2D needed)
7. ✅ **Professor will approve** - classic benchmarks

### Optional: Add ONE More Environment

**If you want 3 environments**, best choices:

1. **FrozenLake-v1** (Easiest to add)
   - No new dependencies
   - Perfect for pure Q-Learning demo
   - 5 minutes to implement
   - Shows tabular vs discretization comparison

2. **LunarLander-v2** (Most impressive)
   - Requires Box2D installation
   - Looks great in presentations
   - More challenging but doable
   - You already have configs ready!

3. **Acrobot-v1** (Good middle ground)
   - No new dependencies
   - Different mechanics than CartPole
   - Medium complexity

---

## 🚀 CODE EXAMPLES

### Quick Test Any Environment

```python
import gymnasium as gym

# Test any environment
env_name = "FrozenLake-v1"  # Change this
env = gym.make(env_name)

print(f"Environment: {env_name}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Random agent test
state, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### Add FrozenLake to Your Project (5 minutes)

```python
# In env/reward_wrapper.py, add to CustomRewardWrapper:

def _frozenlake_reward(self, state, reward, terminated, truncated, info):
    """Reward shaping for FrozenLake."""
    if self.reward_type == "default":
        return reward
    
    elif self.reward_type == "distance_based":
        # Reward based on distance to goal
        goal_pos = 15  # Bottom-right corner in 4x4
        row, col = state // 4, state % 4
        goal_row, goal_col = goal_pos // 4, goal_pos % 4
        distance = abs(row - goal_row) + abs(col - goal_col)
        distance_reward = -0.01 * distance
        
        if terminated and reward > 0:  # Reached goal
            return reward + 1.0
        elif terminated:  # Fell in hole
            return reward - 1.0
        return distance_reward
    
    return reward
```

---

## 📝 MEETING #1 TALKING POINTS

### What to Present:

1. **Current Environments**: CartPole-v1 + MountainCar-v0
   - Working implementations ✅
   - Multiple reward types ✅
   - Fast training ✅

2. **Proposal**: Keep these two OR add one more
   - **Option A**: Add FrozenLake (easiest)
   - **Option B**: Add LunarLander (most impressive)
   - **Option C**: Stay with current two (safest)

3. **Ask Professor**:
   - Is 2 environments sufficient?
   - Preference: discrete (FrozenLake) vs physics (LunarLander)?
   - Any specific benchmarks required?

---

## 📚 Resources

- **Gymnasium Docs**: https://gymnasium.farama.org/
- **Environment List**: https://gymnasium.farama.org/environments/
- **Your Lab Materials**: `Laboratoare/lab2_envs_mdp/`

**Good luck with your project! 🚀**
