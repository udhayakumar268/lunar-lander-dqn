
import random

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, state):
        return self.network(state)


def select_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)

    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values, dim=1).item()
