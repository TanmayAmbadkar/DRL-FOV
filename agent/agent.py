import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import FOVSelectionNet


class Agent:
    def __init__(self, action_space=9, n_frames=5):

        self.memory = deque(maxlen=300)
        self.inventory = []
        self.action_space = action_space
        self.gamma = 0.3
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.n_frames = n_frames
        self.model = FOVSelectionNet(n_frames=self.n_frames)
        self.model.to(self.device)
        print(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    @torch.no_grad()
    def training_action(self, state):

        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        out = F.softmax(
            self.model(
                torch.Tensor(state)
                .reshape(1, self.n_frames, 240, 480, 1)
                .permute([0, 1, 4, 2, 3])
                .to(self.device)
            ),
            dim=1,
        )

        return torch.argmax(out).item()

    @torch.no_grad()
    def action(self, state):

        out = F.softmax(
            self.model(
                torch.Tensor(state)
                .reshape(1, self.n_frames, 240, 480, 1)
                .permute([0, 1, 4, 2, 3])
                .to(self.device)
            ),
            dim=1,
        )

        return torch.argmax(out).item()

    def batch_train(self, batch_size):

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:

            if not done:
                with torch.no_grad():
                    reward = reward + self.gamma * torch.max(
                        (
                            self.model(
                                torch.Tensor(next_state)
                                .reshape(1, self.n_frames, 240, 480, 1)
                                .permute([0, 1, 4, 2, 3])
                                .to(self.device)
                            )
                        )
                    )

            with torch.no_grad():
                target = self.model(
                    torch.Tensor(state)
                    .reshape(1, self.n_frames, 240, 480, 1)
                    .permute([0, 1, 4, 2, 3])
                    .to(self.device)
                )

            target[0][action] = reward
            outputs = self.model(
                torch.Tensor(state)
                .reshape(1, self.n_frames, 240, 480, 1)
                .permute([0, 1, 4, 2, 3])
                .to(self.device)
            )
            criterion = nn.MSELoss()
            td_loss = criterion(outputs, target)
            self.optimizer.zero_grad()
            td_loss.backward()

            self.optimizer.step()

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
