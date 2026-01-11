"""
Deep Q-Network (DQN) for MyCustomEnv
====================================
A neural network approximates Q-values instead of using a table.

Key concepts:
1. Q-Network: Neural net that outputs Q(s,a) for all actions given state s
2. Experience Replay: Store transitions, sample randomly to break correlations
3. Target Network: Separate network for stable Q-value targets
4. Double DQN: Use policy net to select action, target net to evaluate

Architecture: State → [64] → [32] → Q-values
"""

import numpy as np
import pandas as pd
import os
import json
from env import load_environment, DiscreteActionWrapper, FlatObsWrapper
from agents.dqn.agent import DQNAgent


def train():
    """Main training loop."""
    # Setup environment
    raw_env = load_environment("MyCustomEnv")
    env = DiscreteActionWrapper(raw_env)  # Discrete actions (0-4)
    env = FlatObsWrapper(env)             # Flat observation vector

    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent with robust configuration
    agent = DQNAgent(
        state_size=obs_dim,
        action_size=action_dim,
        learning_rate=5e-4,         # Slightly lower LR for stability
        epsilon_decay=0.99,         # Slower decay
        buffer_capacity=50000,
        batch_size=64,
        use_double_dqn=True,
        hidden_dims=[256, 128]      # Larger network
    )

    episodes = 500  # Increased for convergence
    rewards_history = []

    # Load previous best (across runs) if it exists
    os.makedirs('results', exist_ok=True)
    best_avg = -float('inf')
    best_meta_path = 'results/dqn_best.json'
    if os.path.exists(best_meta_path):
        try:
            with open(best_meta_path, 'r') as f:
                meta = json.load(f)
                best_avg = float(meta.get('best_avg', best_avg))
        except Exception:
            pass
    
    print(f"Training DQN for {episodes} episodes...")
    print(f"Observation dim: {obs_dim}, Actions: {action_dim}")

    global_steps = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.train_step()
            
            obs = next_obs
            total_reward += reward

            # Count environment steps and occasionally run a tiny visual demo
            global_steps += 1
            if global_steps % 100 == 0:
                try:
                    print("\n[Visual] Running short demo episode...")
                    demo_env = DiscreteActionWrapper(load_environment("MyCustomEnv", render_mode="human"))
                    demo_env = FlatObsWrapper(demo_env)
                    demo_obs, _ = demo_env.reset()
                    demo_done = False
                    demo_steps = 0
                    while not demo_done and demo_steps < 150:
                        demo_action = agent.select_action(demo_obs, training=False)
                        demo_obs, _, demo_term, demo_trunc, _ = demo_env.step(demo_action)
                        demo_env.render()
                        demo_done = demo_term or demo_trunc
                        demo_steps += 1
                    demo_env.close()
                except Exception:
                    # Demo is purely for visualization; ignore any errors
                    pass
            
        agent.decay_epsilon()
        rewards_history.append(total_reward)

        # Per-episode log so you always see progress
        print(f"Ep {ep+1}/{episodes} | Reward: {total_reward:.1f} | ε: {agent.epsilon:.2f}")
        
        if (ep + 1) % 20 == 0:
            avg = np.mean(rewards_history[-20:])
            print(f"  Avg20: {avg:.1f}")

            # Save best model so far based on moving average
            if avg > best_avg:
                best_avg = avg
                agent.save("results/dqn_model.pth")
                # Persist new best metric
                with open(best_meta_path, 'w') as f:
                    json.dump({'best_avg': best_avg}, f)
                print(f"  ✓ New best DQN model saved (Avg20={best_avg:.2f})")

    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame({'episode': range(1, episodes + 1), 'reward': rewards_history})
    df.to_csv('results/dqn_training.csv', index=False)
    print("\n✓ Results saved to results/dqn_training.csv")


if __name__ == "__main__":
    train()
