import pandas as pd
from stable_baselines3 import DQN, PPO, A2C

from env import load_environment

# Load trained models
models = {
    'DQN': DQN.load('dqn_highway_test'),
    'PPO': PPO.load('ppo_highway_test'),
    'A2C': A2C.load('a2c_highway_test')
}

# Evaluate each agent
results = {}
for name, model in models.items():
    env = load_environment("MyCustomEnv")
    episode_rewards = []
    for episode in range(10):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
    results[name] = episode_rewards
    env.close()

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('evaluation_results.csv', index=False)
print("Evaluation complete. Results saved to evaluation_results.csv.")
