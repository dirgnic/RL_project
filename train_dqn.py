from env import load_environment
from agents.dqn.agent import DQNAgent
import numpy as np

# Use MyCustomEnv for all training
env = load_environment("MyCustomEnv")

obs_dim = np.prod(env.reset()[0].shape)
action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
agent = DQNAgent(state_size=obs_dim, action_size=action_dim)

num_episodes = 200
for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_flat = np.array(obs).flatten()
        action = agent.select_action(state_flat)
        next_obs, reward, terminated, truncated, _ = env.step([action, 0.0])
        next_state_flat = np.array(next_obs).flatten()
        agent.store_transition(state_flat, action, reward, next_state_flat, terminated or truncated)
        agent.train_step()
        obs = next_obs
        total_reward += reward
        done = terminated or truncated
    agent.decay_epsilon()
    print(f"Episode {ep+1}: Reward {total_reward:.2f}, Epsilon {agent.epsilon:.3f}")

# Save the trained model
agent.save("dqn_taxi_model.pth")

print("Training complete. Model saved as dqn_taxi_model.pth.")
