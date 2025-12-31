import gymnasium as gym
from env import load_environment

# Create the environment with human rendering
env = load_environment("highway-v0", render_mode="human")

state, info = env.reset()
done = False

total_reward = 0

while not done:
    action = env.action_space.sample()  # Take random actions
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished. Total reward: {total_reward}")
env.close()
