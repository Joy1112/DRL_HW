import random
import collections
import numpy as np

import torch


class ReplayBuffer():
    def __init__(self, buffer_size):
        self._buffer = collections.deque(maxlen=buffer_size)

    def insert(self, transition):
        self._buffer.append(transition)

    def size(self):
        return len(self._buffer)

    def _canSample(self, batch_size):
        return self.size() >= batch_size

    def sample(self, batch_size):
        """
        Sample the mini-batch and return the tensor.
        """
        assert self._canSample(batch_size)

        mini_batch = random.sample(self._buffer, batch_size)
        state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, next_s, done_mask = transition
            state_lst.append(s)
            action_lst.append([a])
            reward_lst.append([r])
            next_state_lst.append(next_s)
            done_mask_lst.append([done_mask])

        return torch.tensor(state_lst, dtype=torch.float), \
               torch.tensor(action_lst), \
               torch.tensor(reward_lst), \
               torch.tensor(next_state_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
