from copy import copy
import numpy as np

INPUT_DIM = 4
ACTION_DIM = 2

BATCH_SIZE = 64

class ReplayMemory:
    def __init__(self, size):
        self.size = size

        self.states = np.empty((self.size, INPUT_DIM), dtype = np.float32)
        self.actions = np.empty((self.size), dtype = np.int16)
        self.rewards = np.empty((self.size), dtype = np.int16)
        self.next_states = np.empty((self.size, INPUT_DIM), dtype = np.float32)
        self.dones = np.empty((self.size), dtype = np.bool)

        self.count = 0
        self.current = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.current, ...] = state
        self.actions[self.current] =  action
        self.rewards[self.current] = reward
        self.next_states[self.current, ...] = next_state
        self.dones[self.current] = done 

        self.count = max(self.count, self.current + 1) 
        self.current = (self.current + 1) % self.size 

    def sample_batch(self, type='all'):
        shuffle_idx = np.arange(0, self.count)
        np.random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[:BATCH_SIZE]
        
        states = copy(self.states[shuffle_idx])
        actions = self.actions[shuffle_idx]
        rewards = self.rewards[shuffle_idx]
        next_states = copy(self.next_states[shuffle_idx])
        dones = self.dones[shuffle_idx]

        return states, actions, rewards, next_states, dones

