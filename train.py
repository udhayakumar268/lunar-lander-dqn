
import csv
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_model import DQN, select_action
from replay_buffer import ReplayBuffer


# ==========================================
# 1. Configuration
# ==========================================

ENV_NAME = "LunarLander-v3"

NUM_EPISODES = 500
BUFFER_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99
LEARNING_RATE = 0.0005

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_FREQUENCY = 100
CHECKPOINT_FREQUENCY = 100

SEED = 42

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")


# ==========================================
# 2. Reproducibility
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================================
# 3. DQN Training Step
# ==========================================

def train_step(
    model,
    target_model,
    memory,
    optimizer,
    loss_fn,
    device
):

    if len(memory) < BATCH_SIZE:
        return None

    batch = memory.sample(BATCH_SIZE)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.as_tensor(
        np.array(states),
        dtype=torch.float32,
        device=device
    )

    actions = torch.as_tensor(
        actions,
        dtype=torch.long,
        device=device
    )

    rewards = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    next_states = torch.as_tensor(
        np.array(next_states),
        dtype=torch.float32,
        device=device
    )

    dones = torch.as_tensor(
        dones,
        dtype=torch.float32,
        device=device
    )

    # Policy network predictions
    all_q_values = model(states)

    predicted_q = all_q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # Target network Bellman targets
    with torch.no_grad():
        next_q_values = target_model(next_states)

        max_next_q = next_q_values.max(dim=1).values

        target_q = (
            rewards
            + GAMMA * max_next_q * (1 - dones)
        )

    # Calculate loss
    loss = loss_fn(predicted_q, target_q)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Prevent excessively large gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=10.0
    )

    optimizer.step()

    return loss.item()


# ==========================================
# 4. Save Model
# ==========================================

def save_model(model, path):
    torch.save(
        model.state_dict(),
        path
    )


# ==========================================
# 5. Save Training History
# ==========================================

def save_history(history):

    path = OUTPUT_DIR / "training_history.csv"

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "reward",
                "average_loss",
                "epsilon",
                "steps",
                "moving_average_reward"
            ]
        )

        writer.writeheader()
        writer.writerows(history)


# ==========================================
# 6. Plot Training Results
# ==========================================

def plot_training(history):

    episodes = [row["episode"] for row in history]
    rewards = [row["reward"] for row in history]

    moving_average = [
        row["moving_average_reward"]
        for row in history
    ]

    losses = [
        row["average_loss"]
        for row in history
    ]

    # Reward plot
    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        rewards,
        alpha=0.4,
        label="Episode reward"
    )

    plt.plot(
        episodes,
        moving_average,
        label="Moving average (100 episodes)"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Lunar Lander DQN Training Rewards")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_rewards.png",
        dpi=150
    )

    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 5))

    plt.plot(episodes, losses)

    plt.xlabel("Episode")
    plt.ylabel("Average training loss")
    plt.title("DQN Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_loss.png",
        dpi=150
    )

    plt.close()


# ==========================================
# 7. Main Training Function
# ==========================================

def train_agent():

    set_seed(SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    env = gym.make(ENV_NAME)

    env.action_space.seed(SEED)

    # Policy network
    model = DQN().to(device)

    # Target network
    target_model = DQN().to(device)

    target_model.load_state_dict(
        model.state_dict()
    )

    target_model.eval()

    # Replay memory
    memory = ReplayBuffer(BUFFER_CAPACITY)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_fn = nn.SmoothL1Loss()

    epsilon = EPSILON_START
    global_steps = 0

    episode_rewards = []
    history = []

    best_moving_average = float("-inf")

    try:
        for episode in range(1, NUM_EPISODES + 1):

            state, info = env.reset(
                seed=SEED if episode == 1 else None
            )

            done = False
            total_reward = 0.0
            steps = 0
            episode_losses = []

            while not done:

                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                # Epsilon-greedy action
                action = select_action(
                    model,
                    state_tensor,
                    epsilon
                )

                next_state, reward, terminated, truncated, info = (
                    env.step(action)
                )

                done = terminated or truncated

                # Store terminated, not truncated,
                # for Bellman bootstrapping
                memory.push(
                    state.copy(),
                    action,
                    reward,
                    next_state.copy(),
                    terminated
                )

                # Train policy network
                loss = train_step(
                    model,
                    target_model,
                    memory,
                    optimizer,
                    loss_fn,
                    device
                )

                if loss is not None:
                    episode_losses.append(loss)

                global_steps += 1

                # Synchronise target network
                if global_steps % TARGET_UPDATE_FREQUENCY == 0:
                    target_model.load_state_dict(
                        model.state_dict()
                    )

                total_reward += reward
                steps += 1
                state = next_state

            # Episode statistics
            episode_rewards.append(total_reward)

            average_loss = (
                float(np.mean(episode_losses))
                if episode_losses else 0.0
            )

            # Moving average over last 100 episodes
            moving_average_reward = float(
                np.mean(episode_rewards[-100:])
            )

            # Decay exploration
            epsilon = max(
                EPSILON_MIN,
                epsilon * EPSILON_DECAY
            )

            history.append({
                "episode": episode,
                "reward": total_reward,
                "average_loss": average_loss,
                "epsilon": epsilon,
                "steps": steps,
                "moving_average_reward": moving_average_reward
            })

            # Save best model after 100 episodes
            if (
                episode >= 100
                and moving_average_reward > best_moving_average
            ):
                best_moving_average = moving_average_reward

                save_model(
                    model,
                    MODEL_DIR / "dqn_lunar_lander_best.pth"
                )

            print(
                f"Episode {episode}/{NUM_EPISODES} | "
                f"Reward: {total_reward:.2f} | "
                f"Avg100: {moving_average_reward:.2f} | "
                f"Loss: {average_loss:.4f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Steps: {steps}"
            )

            # Periodic checkpoints
            if episode % CHECKPOINT_FREQUENCY == 0:

                checkpoint = {
                    "episode": episode,
                    "model_state_dict": model.state_dict(),
                    "target_model_state_dict": target_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "global_steps": global_steps
                }

                torch.save(
                    checkpoint,
                    MODEL_DIR / f"checkpoint_episode_{episode}.pth"
                )

                save_history(history)
                plot_training(history)

                print(
                    f"Checkpoint saved at episode {episode}"
                )

        # Save final model
        save_model(
            model,
            MODEL_DIR / "dqn_lunar_lander_final.pth"
        )

        save_history(history)
        plot_training(history)

        print("\nTraining completed!")
        print(
            f"Final 100-episode average reward: "
            f"{np.mean(episode_rewards[-100:]):.2f}"
        )

    finally:
        env.close()

    return model, history


# ==========================================
# 8. Main Execution
# ==========================================

if __name__ == "__main__":
    model, history = train_agent()
