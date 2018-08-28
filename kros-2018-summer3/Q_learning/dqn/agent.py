import numpy as np
import random

INPUT_DIM = 4
ACTION_DIM = 2

EPSILON_DECAY = 0.999 
EPSILON_MIN = 0.1

class Agent(object):
    def __init__(self, network):
        self.network = network
        self.epsilon = 1.0
        self.epsilon_decay = EPSILON_DECAY 
        self.epsilon_min = EPSILON_MIN 
       
    def train_network(self, states, actions, rewards, next_states, dones):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.network.train(states, actions, rewards, next_states, dones)

    def update_target_q_network(self):
        self.network.update_target_q_network()

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        else:
            q = self.network.predict_q_single(state)
            return np.argmax(q)