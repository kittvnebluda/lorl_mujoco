import time

import gymnasium as gym
from gymnasium.utils.env_checker import check_env


def random_agent():
    with gym.make("UnitreeGo2Env-v0") as env:
        check_env(env.unwrapped)

    with gym.make("UnitreeGo2Env-v0", render_mode="human") as env:
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0.0

        print("Starting random rollout... Press Ctrl+C to stop.")

        try:
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated

                env.render()
                time.sleep(0.02)

                if done:
                    print(f"Episode finished | Total reward: {total_reward:.2f}")
                    obs, info = env.reset()
                    total_reward = 0
                    done = False

        except KeyboardInterrupt:
            print("Stopped by user")


if __name__ == "__main__":
    random_agent()
