import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvCritic(nn.Module):
    def __init__(self, env, config, hidden_dim=128):
        super(ConvCritic, self).__init__()
        self.device = config["DEVICE"]

        # Convolutional layers for processing the state (RGB image)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)  # Conv layer 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # Conv layer 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # Conv layer 3

        # After convolutions, we need to flatten the state features
        # Fully connected layers after flattening the state feature maps
        self.fc_state = nn.Linear(64 * 7 * 7, hidden_dim)  # Assuming 7x7 output size from conv layers

        # Fully connected layers for processing the action input
        self.fc_action = nn.Linear(env.action_space.shape[0], hidden_dim)

        # Combine the processed state and action and pass through two fully connected layers
        self.fc1 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

        # Apply Xavier initialization to all layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.q_value.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.q_value.bias)

        self.to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)

        # Process state (image) through convolutional layers
        x_state = F.relu(self.conv1(state))
        x_state = F.relu(self.conv2(x_state))
        x_state = F.relu(self.conv3(x_state))

        # Flatten the output of the convolutional layers
        x_state = x_state.view(x_state.size(0), -1)
        x_state = F.relu(self.fc_state(x_state))  # Fully connected layer for state

        # Process action through a fully connected layer
        x_action = F.relu(self.fc_action(action))  # Fully connected layer for action

        # Concatenate the processed state and action
        x = torch.cat([x_state, x_action], dim=-1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output Q-value
        q_value = self.q_value(x)
        return q_value