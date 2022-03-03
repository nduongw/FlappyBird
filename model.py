import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()

        self.action_space = action_space
        self.observation_space = observation_space

        self.fc1 = nn.Linear(self.observation_space, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, self.action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
