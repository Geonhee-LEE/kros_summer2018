import os
import numpy as np
import time

from grid_env import GridEnv
from agent import Agent


if __name__ == "__main__":
    env = GridEnv()
    agent = Agent(env)

    # load table 
    agent.q_table = np.load("q_table.npy")

    state = env.reset()
    state_history = [] 

    while True:
        action = agent.get_action(state, greedy=True)
        state_history.append(state.tolist())
        next_state, reward, done = env.step(action)

        state = next_state

        if done:
            break
    state_history.append(next_state.tolist())

    for t in range(len(state_history)):
        debug_str = ""
        for h in range(env.height):

            for w in range(env.width):
                debug_str += '****************'
            debug_str += "*\n"
            for w in range(env.width):
                debug_str += "#               "
            debug_str += "#\n"
            for w in range(env.width):
                if [h, w] in state_history[:t + 1]:
                    debug_str += "#     @@@@      "
                else:
                    debug_str += "#               "
            debug_str += "#\n"
            for w in range(env.width):
                if [h, w] in state_history[:t + 1]:
                    debug_str += "#     @@@@      "
                else:
                    debug_str += "#               "
            debug_str += "#\n"
            for w in range(env.width):
                debug_str += "#               "
            debug_str += "#\n"
        for c in range(env.width):
            debug_str += '****************'
        debug_str += "*\n"

        os.system("clear")
        print(debug_str)

        # delay
        time.sleep(0.3)