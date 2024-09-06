from collections import deque, namedtuple
import random
import torch

class ReplayMemory:
    def __init__(self, config):
        self.buffer_size = config["BUFFER_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.memory = deque([], maxlen=self.buffer_size)

    def add_element(self, *args):
        Transition = namedtuple('Transition',('state', 'action',
                                              'next_state', 'reward', 'done'))
        self.memory.append(Transition(*args))

    def sample(self):
        batch = zip(*(random.sample(self.memory, self.batch_size)))
        return [torch.cat(items) for items in batch]

    def __len__(self):
        return len(self.memory)