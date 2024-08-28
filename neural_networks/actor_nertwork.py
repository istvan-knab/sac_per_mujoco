import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from train_setup.seed_all import seed_all

class Actor(nn.Module):
    def __init__(self, env, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, env.action_space.shape[0])
        self.std = nn.Linear(hidden_dim, env.action_space.shape[0])
        self.env = env
        self.reparam_noise = 1e-6
        seed = 0
        seed_all(seed, self.env)

        # Apply Xavier initialization to all layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mean.weight)
        nn.init.xavier_uniform_(self.std.weight)
        # Optional: Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.std.bias)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)
        std = F.softplus(std) + 1e-6
        if torch.isnan(mean).any() or torch.isnan(std).any():
            mean = torch.nan_to_num(mean, nan=1e-6)
            std = torch.nan_to_num(std, nan=1e-6)


        return mean, std

    def sample_action(self, state):
        mu, sigma = self.forward(state)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        probabilities = Normal(mu, sigma)
        actions = probabilities.sample()
        action = torch.tanh(actions) * torch.tensor(self.env.action_space.high, dtype=torch.float32)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def select_best_action(self, state):
        mu, sigma = self.forward(state)
        action = mu
        return action