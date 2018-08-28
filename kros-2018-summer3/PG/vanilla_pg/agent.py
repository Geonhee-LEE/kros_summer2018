import numpy as np

INPUT_DIM = 4
ACTION_DIM = 2

class Agent(object):
    def __init__(self, network):
        self.network = network 
        self.actions = np.arange(ACTION_DIM)
        self.discount_factor = 0.99

    def train_network(self, x_, r_, a_):
        self.network.train(x_, r_, a_)

    def predict(self, x):
        return self.network.predict_p_single(x)

    def select_action(self, prediction):
        action = np.random.choice(self.actions, p=prediction)
        return action

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(ACTION_DIM)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    @staticmethod
    def accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            reward_sum = discount_factor * reward_sum + experiences[t].reward
            experiences[t].reward = reward_sum
        return experiences[:-1]