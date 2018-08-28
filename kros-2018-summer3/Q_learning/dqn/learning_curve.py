from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import gym

from network import Network
from agent import Agent
from replay_memory import ReplayMemory


NUM_EPISODE = 150
RENDER = False
REWARD_SUM_QUEUE_SIZE = 100

MEMORY_SIZE = 2000
TRAIN_START = 1000


if __name__ == "__main__":
    all_reward_sum_history = []
    for k in range(10):
        env = gym.make('CartPole-v0')
        network = Network("cpu:0")
        agent = Agent(network)
        replay_memory = ReplayMemory(MEMORY_SIZE)

        reward_sum_queue = []
        reward_sum_history = []
        for n_episode in range(NUM_EPISODE):
            state = env.reset()

            done = False 
            reward_sum = 0.0
            while not done:
                if RENDER:
                    env.render()

                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward 

                replay_memory.add(state, action, reward, next_state, done)

                if replay_memory.count > TRAIN_START:
                    agent.train_network(*replay_memory.sample_batch())

                state = next_state # change current state
                
                if done:
                    if len(reward_sum_queue) > REWARD_SUM_QUEUE_SIZE:
                        reward_sum_queue.append(reward_sum)
                        reward_sum_queue.pop(0)
                    else:
                        reward_sum_queue.append(reward_sum)
                    reward_sum_history.append(reward_sum)

                    agent.update_target_q_network()
                    print("episode: %d, avg_score: %.2f, memory_count: %d, epsilon: %.2f" 
                        % (n_episode, np.mean(reward_sum_queue), replay_memory.count, agent.epsilon))
        all_reward_sum_history.append(reward_sum_history)
    sns.tsplot(all_reward_sum_history)
    plt.xlim([0, 150])
    plt.xlabel('Episode')
    plt.ylabel('Averaged sum of rewards per episode')
    plt.show()

