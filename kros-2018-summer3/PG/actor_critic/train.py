import matplotlib.pyplot as plt

import gym
import numpy as np

from network import Network
from agent import Agent

NUM_EPISODE = 1000
RENDER = False
REWARD_SUM_QUEUE_SIZE = 100

class Experience(object):
    def __init__(self, state, action, prediction, reward, done):
        self.state = state
        self.action = action
        self.prediction = prediction
        self.reward = reward
        self.done = done

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    network = Network("cpu:0")
    agent = Agent(network)

    reward_sum_queue = []
    reward_sum_history = []
    reward_sum_avg_history = []
    for n_episode in range(NUM_EPISODE):
        state = env.reset()

        done = False
        reward_sum = 0.0

        experiences = []
        while not done:
            if RENDER:
                env.render()

            prediction, value = agent.predict(state)
            action = agent.select_action(prediction)

            prev_state = state 
            state, reward, done, _ = env.step(action)
            reward_sum += reward

            exp = Experience(prev_state, action, prediction, reward, done)
            experiences.append(exp)

            if done:
                if len(reward_sum_queue) > REWARD_SUM_QUEUE_SIZE:
                    reward_sum_queue.append(reward_sum)
                    reward_sum_queue.pop(0)
                else:
                    reward_sum_queue.append(reward_sum)
                reward_sum_history.append(reward_sum)
                reward_sum_avg_history.append(np.mean(reward_sum_queue))

                terminal_reward = 0
                updated_exps = Agent.accumulate_rewards(
                    experiences,
                    agent.discount_factor,
                    terminal_reward)
                x_, r_, a_ = agent.convert_data(updated_exps)

                agent.train_network(x_, r_, a_)

                print("episode: %d, avg_score: %.2f"
                % (n_episode, np.mean(reward_sum_queue)))

    plt.plot(reward_sum_history, 'b')
    plt.plot(reward_sum_avg_history, 'r')
    plt.xlabel('Episode')
    plt.ylabel('Averaged sum of rewards per episode')
    plt.show()