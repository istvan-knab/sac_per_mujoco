import numpy as np
import torch
from numpy import dtype


class ReplayMemory:

    def __init__(self, config):
        self.buffer_size = config["BUFFER_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.device = config["DEVICE"]
        self.state = np.zeros([self.buffer_size, config["INPUT_FEATURES"]])
        self.action = np.zeros([self.buffer_size, config["ACTION_SPACE"]])
        self.next_state = np.zeros([self.buffer_size, config["INPUT_FEATURES"]])
        self.reward = np.zeros(self.buffer_size)
        self.done = np.zeros([self.buffer_size])
        self.queue_length = 0
        self.pointer = 0
        self.indicies = np.zeros(self.batch_size)

    def add_element(self, state, action, next_state, reward, done):
        state = state.cpu().numpy()
        action = action.cpu().numpy()
        next_state = next_state.cpu().numpy()
        reward = reward.cpu().numpy()

        if self.queue_length < self.buffer_size:
            self.queue_length += 1
        self.state = np.roll(self.state, (1, 0), (0, 1))
        self.action = np.roll(self.action, (1, 0), (0, 1))
        self.next_state = np.roll(self.next_state, (1, 0), (0, 1))
        self.reward = np.roll(self.reward, 1)
        self.done = np.roll(self.done, 1)
        self.state[0] = state
        self.action[0] = action
        self.next_state[0] = next_state
        self.reward[0] = reward
        self.done[0] = done

    def sample(self):
        self.indicies = np.random.choice(range(self.queue_length), self.batch_size)
        state = torch.tensor(self.state[self.indicies], dtype=torch.float32).to(self.device)
        action = torch.tensor(self.action[self.indicies], dtype=torch.float32).to(self.device)
        next_state = torch.tensor(self.next_state[self.indicies], dtype=torch.float32).to(self.device)
        reward = torch.tensor(self.reward[self.indicies], dtype=torch.float32).to(self.device).unsqueeze_(dim=1)
        done = torch.tensor(self.done[self.indicies], dtype=torch.bool).to(self.device).unsqueeze_(dim=1)

        return state, action, next_state, reward, done


    def __len__(self):
        return self.queue_length