
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dqn_model import DQN


ENV_NAME = "LunarLander-v3"
MODEL_PATH = Path("models/dqn_lunar_lander_best.pth")

EVAL_EPISODES = 20


def evaluate_agent():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run train.py first."
        )

    model = DQN().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    env = gym.make(ENV_NAME)

    rewards = []

    try:
        for episode in range(1, EVAL_EPISODES + 1):

            state, info = env.reset(seed=1000 + episode)

            done = False
            total_reward = 0.0

            while not done:

                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    q_values = model(state_tensor)

                    action = torch.argmax(
                        q_values,
                        dim=1
                    ).item()

                state, reward, terminated, truncated, info = (
                    env.step(action)
                )

                done = terminated or truncated
                total_reward += reward

            rewards.append(total_reward)

            print(
                f"Evaluation episode {episode}: "
                f"{total_reward:.2f}"
            )

    finally:
        env.close()

    print("\nEvaluation Results")
    print("-------------------------")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std reward: {np.std(rewards):.2f}")
    print(f"Minimum reward: {np.min(rewards):.2f}")
    print(f"Maximum reward: {np.max(rewards):.2f}")

    print(
        f"Episodes with reward >= 200: "
        f"{sum(r >= 200 for r in rewards)}/{EVAL_EPISODES}"
    )


if __name__ == "__main__":
    evaluate_agent()
