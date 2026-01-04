"""
Test environment module.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from env import load_environment, CustomRewardWrapper


def test_load_environment():
    """Test loading environments."""
    print("\n=== Testing Environment Loading ===")
    
    environments = ["CartPole-v1", "MountainCar-v0", "LunarLander-v2"]
    
    for env_name in environments:
        try:
            env = load_environment(env_name, seed=42)
            print(f"[OK] {env_name} loaded successfully")
            print(f"  Observation space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")
            env.close()
        except Exception as e:
            if "Box2D" in str(e):
                print(f"[WARN] {env_name} skipped: Box2D not installed (install with: pip install 'gymnasium[box2d]')")
            else:
                print(f"[ERROR] {env_name} failed: {e}")
        environments.append("highway-v0")


def test_reward_wrappers():
    """Test reward wrappers."""
    print("\n=== Testing Reward Wrappers ===")
    
    # CartPole
    env = load_environment("CartPole-v1", seed=42)
    reward_types = ["default", "bonus_center", "penalize_oscillations"]
    
    print("\nCartPole-v1:")
    for reward_type in reward_types:
        try:
            wrapped_env = CustomRewardWrapper(env, reward_type=reward_type, env_name="CartPole-v1")
            state, _ = wrapped_env.reset()
            state, reward, terminated, truncated, _ = wrapped_env.step(0)
            print(f"  [OK] {reward_type}: reward = {reward:.4f}")
        except Exception as e:
            print(f"  [ERROR] {reward_type}: {e}")
    
    env.close()
    
    # MountainCar
    env = load_environment("MountainCar-v0", seed=42)
    reward_types = ["default", "height_based", "potential_based"]
    
    print("\nMountainCar-v0:")
    for reward_type in reward_types:
        try:
            wrapped_env = CustomRewardWrapper(env, reward_type=reward_type, env_name="MountainCar-v0")
            state, _ = wrapped_env.reset()
            state, reward, terminated, truncated, _ = wrapped_env.step(0)
            print(f"  [OK] {reward_type}: reward = {reward:.4f}")
        except Exception as e:
            print(f"  [ERROR] {reward_type}: {e}")
    
    env.close()
    
    # LunarLander (skip if Box2D not available)
    try:
        env = load_environment("LunarLander-v2", seed=42)
        reward_types = ["default", "fuel_efficiency", "safety_first"]
        
        print("\nLunarLander-v2:")
        for reward_type in reward_types:
            try:
                wrapped_env = CustomRewardWrapper(env, reward_type=reward_type, env_name="LunarLander-v2")
                state, _ = wrapped_env.reset()
                state, reward, terminated, truncated, _ = wrapped_env.step(0)
                print(f"  [OK] {reward_type}: reward = {reward:.4f}")
            except Exception as e:
                print(f"  [ERROR] {reward_type}: {e}")
        
        env.close()
    except Exception as e:
        if "Box2D" in str(e):
            print("\nLunarLander-v2: [WARN] Skipped (Box2D not installed)")
        # HighwayEnv
        try:
            env = load_environment("highway-v0", seed=42)
            reward_types = ["default"]
            print("\nHighway-v0:")
            for reward_type in reward_types:
                try:
                    wrapped_env = CustomRewardWrapper(env, reward_type=reward_type, env_name="highway-v0")
                    state, _ = wrapped_env.reset()
                    state, reward, terminated, truncated, _ = wrapped_env.step(0)
                    print(f"  [OK] {reward_type}: reward = {reward:.4f}")
                except Exception as e:
                    print(f"  [ERROR] {reward_type}: {e}")
            env.close()
        except Exception as e:
            print(f"[ERROR] Highway-v0 failed: {e}")


def test_random_agent():
    """Test environment with random agent."""
    print("\n=== Testing Random Agent ===")
    
    env = load_environment("CartPole-v1", seed=42)
    wrapped_env = CustomRewardWrapper(env, reward_type="bonus_center", env_name="CartPole-v1")
    
    state, _ = wrapped_env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 100:
        action = wrapped_env.action_space.sample()
        state, reward, terminated, truncated, _ = wrapped_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    print(f"Random agent on CartPole-v1 (bonus_center):")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    
    wrapped_env.close()


if __name__ == "__main__":
    def test_load_environment():
        print("\nTesting custom environment loader...")
        try:
            env = load_environment("MyCustomEnv", seed=42)
            obs, info = env.reset()
            print(f"  MyCustomEnv loaded successfully. obs type: {type(obs)}")
            env.close()
        except Exception as e:
            print(f"  Failed to load MyCustomEnv: {e}")
