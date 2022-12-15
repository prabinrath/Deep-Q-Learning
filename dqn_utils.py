import random

class ReplayMemory():
    def __init__(self, max_len=10000):
        self.queue = []
        self.max_len = max_len
    
    def push(self, data):
        self.queue.append(data)
        if len(self.queue)>self.max_len:
            self.queue.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.queue, batch_size)
    
    def length(self):
        return len(self.queue)