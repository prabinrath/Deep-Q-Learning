import random
import numpy as np

class ReplayMemory():
    def __init__(self, max_len=10000):
        print('Replay Memory')
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

class BatchReplayMemory():
    def __init__(self, n_buffer, max_len=10000):
        print('Batched Replay Memory')
        self.queue = []
        self.max_len = max_len
        self.n_buffer = n_buffer
    
    def push(self, data):
        self.queue.append(data)
        if len(self.queue)>self.max_len:
            self.queue.pop(0)
    
    def sample(self, batch_size):
        batch = []
        while len(batch)<batch_size:
            idx = random.sample(range(len(self.queue)),1)[0]
            if idx >= self.n_buffer:
                s, a, r, n, t = zip(*self.queue[idx-self.n_buffer:idx])
                interm_t = False
                for i in t[:-1]:
                    interm_t = interm_t or i
                    
                # Abrupt transitions should not be sampled
                if not interm_t:
                    batch.append((np.expand_dims(np.vstack(s),0),a[-1],r[-1],np.expand_dims(np.vstack(n),0),t[-1]))

        return batch
    
    def length(self):
        return len(self.queue)