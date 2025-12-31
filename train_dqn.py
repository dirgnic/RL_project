env.close()

from env import load_environment

# Use MyCustomEnv for all training
env = load_environment("MyCustomEnv")

# If using SB3, wrap with Monitor if needed
# from stable_baselines3.common.monitor import Monitor
# env = Monitor(env, filename="dqn_monitor.csv")

# Example: If using your custom DQN agent, instantiate and train here
# from agents.dqn.agent import DQNAgent
# ...

print("Custom environment loaded. Ready for training!")
