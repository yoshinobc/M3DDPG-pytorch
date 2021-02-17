import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        y = torch.tanh(h)
        return y


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = state.squeeze(0)
        h = F.relu(self.fc1(torch.cat([state, action], axis=1)))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return y

