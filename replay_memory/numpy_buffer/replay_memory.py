import numpy as np


class ReplayMemory:

    def __init__(self, config):
        self.buffer_size = config["BUFFER_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.device = config["DEVICE"]
        self.state = np.zeros([self.buffer_size, config["INPUT_FEATURES"]])
        self.action = np.zeros([self.buffer_size, config["ACTION_SPACE"]])
        self.next_state = np.zeros([self.buffer_size, config["INPUT_FEATURES"]])
        self.reward = np.zeros(self.buffer_size)
        self.done = np.zeros(self.buffer_size)
        self.queue_length = 0
        self.pointer = 0
        self.indicies = np.zeros(self.batch_size)

    def add_element(self, state, action, next_state, reward, done):
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
        print("Hello")


    def __len__(self):
        return self.queue_length