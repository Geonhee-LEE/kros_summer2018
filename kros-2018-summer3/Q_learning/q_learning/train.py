import os
import numpy as np
import time

from grid_env import GridEnv
from agent import Agent

NUM_EPISODE = 1000

if __name__ == "__main__":
    env = GridEnv()
    agent = Agent(env)

    for n_episode in range(NUM_EPISODE):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update_table(state, action, reward, next_state)
            state = next_state

            if done:
                break

        debug_str = ""
        for h in range(env.height):
            for w in range(env.width):
                debug_str += '****************'
            debug_str += "*\n"
            for w in range(env.width):
                debug_str += '# up:' + str('%.2f ' % (agent.q_table[h, w, 0])).rjust(11)
            debug_str += "#\n"
            for w in range(env.width):
                debug_str += '# right:' + str('%.2f ' % (agent.q_table[h, w, 1])).rjust(8)
            debug_str += "#\n"
            for w in range(env.width):
                debug_str += '# down:' + str('%.2f ' % (agent.q_table[h, w, 2])).rjust(9)
            debug_str += "#\n"
            for w in range(env.width):
                debug_str += '# left:' + str('%.2f ' % (agent.q_table[h, w, 3])).rjust(9)
            debug_str += "#\n"
        for c in range(env.width):
            debug_str += '****************'
        debug_str += "*\n"
        debug_str += "num_episode=%d" % n_episode

        os.system("clear")
        print(debug_str)

    # save table
    np.save("q_table.npy", agent.q_table)

