"""
Demo: Trained Agent in Action
=============================
Runs a trained agent in the environment with visualization.

Usage:
    python demo_agent.py [dqn|reinforce|tabular]
"""

import sys
import numpy as np
import torch
import time
import argparse
from env import load_environment, DiscreteActionWrapper, FlatObsWrapper, TabularObsWrapper

# Import Agents
from agents.dqn.agent import DQNAgent
from train_reinforce import REINFORCEAgent
from train_tabular import TabularQAgent

def run_demo(agent_type="dqn"):
    print(f"\n--- Starting Demo for {agent_type.upper()} ---")
    
    # 1. Setup Base Environment
    # We use render_mode="human" to see the window
    raw_env = load_environment("MyCustomEnv", render_mode="human")
    env = DiscreteActionWrapper(raw_env)
    env = FlatObsWrapper(env)
    
    agent = None
    
    # 2. Configure Agent & Specific Wrappers
    if agent_type == "dqn":
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        # Must match training configuration in train_dqn.py
        agent = DQNAgent(
            state_size=obs_dim,
            action_size=action_dim,
            hidden_dims=[256, 128],
        )
        try:
            agent.load("results/dqn_model.pth")
            agent.epsilon = 0.05  # Low epsilon for demo
        except FileNotFoundError:
            print("Error: results/dqn_model.pth not found. Train usage: python train_dqn.py")
            return

    elif agent_type == "reinforce":
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = REINFORCEAgent(obs_dim, action_dim)
        try:
            agent.load("results/reinforce_model.pth")
        except FileNotFoundError:
            print("Error: results/reinforce_model.pth not found. Train using: python train_reinforce.py")
            return

    elif agent_type == "tabular":
        # Add Tabular Wrapper on top
        env = TabularObsWrapper(env, grid_bins=6, fuel_bins=3)
        obs_bins = [6, 6, 3, 2, 3, 3] # Must match training config
        action_dim = env.action_space.n
        agent = TabularQAgent(obs_bins, action_dim)
        try:
            agent.load("results/tabular_model.npy")
            agent.epsilon = 0.0  # Force exploitation
        except FileNotFoundError:
            print("Error: results/tabular_model.npy not found. Train using: python train_tabular.py")
            return
            
    else:
        print(f"Unknown agent type: {agent_type}")
        return

    print("Agent loaded successfully.")
    print("Press Ctrl+C to stop.")
    
    # 3. Run Loop
    try:
        obs, _ = env.reset()
        if agent_type == "tabular":
            state = agent.discretize(obs)
            
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # Select Action
            if agent_type == "dqn":
                action = agent.select_action(obs, training=False)
            elif agent_type == "reinforce":
                # REINFORCE select_action returns (action, log_prob)
                action, _ = agent.select_action(obs) 
            elif agent_type == "tabular":
                action = agent.select_action(state)
            
            # Step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Update state/obs
            obs = next_obs
            if agent_type == "tabular":
                state = agent.discretize(next_obs)
                
            total_reward += reward
            done = terminated or truncated
            step += 1
            
            # Visualization delay
            time.sleep(0.05)
            
            # Render is handled by the env (pygame window)
            env.render()
            
            print(f"Step: {step} | Reward: {reward:.2f} | Total: {total_reward:.2f}", end="\r")
            
        print(f"\nEpisode finished! Total Reward: {total_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", nargs="?", default="dqn", choices=["dqn", "reinforce", "tabular"], help="Agent to demo")
    args = parser.parse_args()
    
    run_demo(args.agent)
