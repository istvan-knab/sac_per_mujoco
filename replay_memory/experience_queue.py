import numpy as np
from numba import njit

class ExperienceQueue:
    def __init__(self, config):
        self.buffer_size = config["BUFFER_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.queue = np.zeros(self.buffer_size)
        self.queue_length = 0
        self.pointer = 0

    @njit
    def add_element(self, element):
        if self.queue_length < self.buffer_size:
            self.queue_length += 1
            self.pointer += 1
            self.queue[self.pointer] = element
        else:
            self.queue = np.roll(self.queue, 1)
            self.queue[0] = element



    def __len__(self):
        return self.queue_length
