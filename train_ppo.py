env.close()

from env import load_environment

# Use MyCustomEnv for all training
env = load_environment("MyCustomEnv")

# If using SB3, wrap with Monitor if needed
# from stable_baselines3.common.monitor import Monitor
# env = Monitor(env, filename="ppo_monitor.csv")

# Example: If using your custom PPO agent, instantiate and train here
# ...

print("Custom environment loaded. Ready for PPO training!")
