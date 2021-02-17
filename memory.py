import random
from collections import deque


class ReplayMemory(object):
    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.memory = deque([], maxlen=int(capacity))

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory.clear()

    def size(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)
