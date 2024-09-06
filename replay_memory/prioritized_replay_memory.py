from collections import deque, namedtuple
import random
import torch

from replay_memory.replay_memory import ReplayMemory

class PER(ReplayMemory):
    def __init__(self, config):
        self.buffer_size = config["BUFFER_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.memory = deque([], maxlen=self.buffer_size)
        self.td_errors = deque([], maxlen=self.buffer_size)
        self.weights = deque([], maxlen=self.buffer_size)
        self.fit_counts = deque([], maxlen=self.buffer_size)

    def update_priorities(self):
        pass

    def add_element(self, *args):
        transition = namedtuple('transition', ('state', 'action',
                                               'next_state', 'reward', 'done'))

        self.memory.append(transition(*args))

    def sample(self):
        batch = zip(*(random.sample(self.memory, self.batch_size)))
        return [torch.cat(items) for items in batch]

    def __len__(self):
        return len(self.memory)