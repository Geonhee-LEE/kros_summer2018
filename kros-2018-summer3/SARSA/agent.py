from __future__ import print_function

import numpy as np
import random
from collections import defaultdict

from grid_env import GridEnv

class Agent(object):
    def __init__(self, env):
        self.env = env

        self.learning_rate = 0.1
        self.discount_factor = 0.99 
        self.epsilon = 0.9

        self.q_table = np.zeros(env.state_action_dim, dtype=np.float)

    def update_table(self, state, action, reward, next_state, next_action):
        q_prev = self.q_table[state[0], state[1], action]
        q_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], next_action]
        self.q_table[state[0], state[1], action] += self.learning_rate * (q_target - q_prev)

    def get_action(self, state, greedy=False):
        if greedy:
            epsilon = 0
        else:
            epsilon = self.epsilon

        valid_actions = self.env.get_valid_actions()
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            Q_s = self.q_table[state[0], state[1], valid_actions]
            action = random.choice(valid_actions[np.flatnonzero(Q_s == np.max(Q_s))])
        return action
