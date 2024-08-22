import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Critic(nn.Module):
    def __init__(self, env, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

        # Apply Xavier initialization to all layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.q_value.weight)

        # Optional: Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.q_value.bias)



    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_value(x)
        return q_value