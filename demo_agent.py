"""
Demo: Trained Agent in Action
=============================
Runs a trained agent in the environment with visualization.
"""

import sys
import numpy as np
import torch
import time
from env import load_environment, DiscreteActionWrapper, FlatObsWrapper, TabularObsWrapper
from agents.dqn.agent import DQNAgent


def demo_dqn(model_path="results/dqn_model.pth"):
    """Watch the DQN agent drive."""
    print(f"\nLoading DQN agent from {model_path}...")
    
    # 1. Setup Environment (Human Render Mode)
    raw_env = load_environment("MyCustomEnv", render_mode="human")
    env = DiscreteActionWrapper(raw_env)
    env = FlatObsWrapper(env)
    
    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 2. Load Agent
    agent = DQNAgent(state_size=obs_dim, action_size=action_dim)
    try:
        agent.load(model_path)
        agent.epsilon = 0.0  # Force exploitation
    except FileNotFoundError:
        print("Model file not found! Train the agent first.")
        return

    print("Agent loaded. Starting demo...")
    
    # 3. Run Loop
    done = False
    total_reward = 0
    
    while not done:
        # Select best action
        action = agent.select_action(obs, training=False)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Slow down so we can see
        time.sleep(0.05)
        
        # Simple logging
        print(f"Reward: {reward:.2f} | Total: {total_reward:.2f}", end="\r")
        
    print(f"\nEpisode finished! Total Reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    demo_dqn()
