from collections import deque, namedtuple
import numpy as np
import torch

from replay_memory.replay_memory import ReplayMemory

class UCBMemory(ReplayMemory):
    def __init__(self, config):
        self.buffer_size = config["BUFFER_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.init_td_error = config["INIT_TD_ERROR"]
        self.init_weight = config["INIT_WEIGHT"]
        self.memory = deque([], maxlen=self.buffer_size)
        self.td_errors = deque([], maxlen=self.buffer_size)
        self.weights = deque([], maxlen=self.buffer_size)
        self.fit_counts = deque([], maxlen=self.buffer_size)

    def update_priorities(self, td_errors):
        indicies_list = self.sample_indices.tolist()
        td_errors = np.array(td_errors.unsqueeze(0).detach().cpu().numpy().flatten())
        for count, index in enumerate(indicies_list):
            self.td_errors[index] = np.abs(td_errors[count])
        for count, index in enumerate(indicies_list):
            self.weights[index] = ((self.td_errors[count] + self.init_td_error) /
                                   sum(self.td_errors))
            self.fit_counts[index] += 1

    def add_element(self, *args):
        transition = namedtuple('transition', ('state', 'action',
                                               'next_state', 'reward', 'done'))
        self.memory.append(transition(*args))
        self.td_errors.append(self.init_td_error)
        self.weights.append(self.init_weight)
        self.fit_counts.append(0)

    def sample(self):
        w = torch.tensor(self.weights)
        w = F.softmax(w, dim=0)
        self.sample_indices = np.random.choice(range(len(self.memory)), self.batch_size,
                                               p=w)
        sampled_elements = [self.memory[i] for i in self.sample_indices]
        batch = zip(*sampled_elements)
        return [torch.cat(items) for items in batch]


def __len__(self):
    return len(self.memory)