import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from train_setup.seed_all import seed_all

class Actor(nn.Module):
    def __init__(self, env, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, env.action_space.shape[0])
        self.std = nn.Linear(hidden_dim, env.action_space.shape[0])
        self.env = env
        self.reparam_noise = 0.1
        seed = 0
        seed_all(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

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
        # Convert all parameters to double precision
        self.fc1 = self.fc1.to(dtype=torch.float64)
        self.fc2 = self.fc2.to(dtype=torch.float64)
        self.mean = self.mean.to(dtype=torch.float64)
        self.std = self.std.to(dtype=torch.float64)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)
        std = F.softplus(std) + 1e-6 # Limit log_std to stabilize learning

        return mean, std

    def sample_action(self, state):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        actions = probabilities.sample()
        action = torch.tanh(actions) * self.env.action_space.high
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def select_best_action(self, state):
        mu, sigma = self.forward(state)
        action = mu
        return action