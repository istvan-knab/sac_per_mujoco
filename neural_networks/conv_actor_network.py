import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from train_setup.seed_all import seed_all

class ConvActor(nn.Module):
    def __init__(self, env, config, hidden_dim=128):
        super(ConvActor, self).__init__()
        self.device = config["DEVICE"]
        # RGB input, so in_channels=3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)  # Conv layer 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # Conv layer 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # Conv layer 3

        # After convolutions, the output will be flattened and fed into fully connected layers.
        # You will need to calculate the correct input size after convolutions.
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_dim)  # This size depends on the input image size.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, env.action_space.shape[0])
        self.std = nn.Linear(hidden_dim, env.action_space.shape[0])

        self.reparam_noise = 1e-6

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

        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device).float() / 255.0
        x = F.relu(self.conv1(state))  # Pass through conv layers
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the conv output
        x = F.relu(self.fc1(x))  # Fully connected layers
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        std = self.std(x)
        std = F.softplus(std) + 1e-6  # Ensure std is positive

        return mean, std

    def sample_action(self, state):
        mu, sigma = self.forward(state)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        probabilities = Normal(mu, sigma)
        actions = probabilities.sample()
        action = torch.tanh(actions) * torch.tensor(self.env.action_space.high, dtype=torch.float32).to(self.device)
        log_probs = probabilities.log_prob(actions)
        argument = 1 - action.pow(2) + self.reparam_noise
        log_probs -= torch.log(argument)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def select_best_action(self, state):
        mu, sigma = self.forward(state)
        action = mu
        return action