# DQN for Taxi-v3 (extended)
# (Stub for Person C)

from agents.dqn.agent import DQNAgent
from env import load_environment

if __name__ == "__main__":
    env = load_environment("MyCustomEnv")
    obs, _ = env.reset()
    state_size = len(obs)
    action_size = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    # TODO: Training loop for DQN on extended Taxi
    print("DQN agent stub ready.")
