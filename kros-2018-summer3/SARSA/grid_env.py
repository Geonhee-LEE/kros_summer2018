import numpy as np

N_HEIGHT = 7 # grid height
N_WIDTH = 7 # grid width 


class GridEnv(object):
    def __init__(self):
        self.height = N_HEIGHT
        self.width = N_WIDTH

        # State and Action space
        self.action_dict = {"up":0, "right": 1, "down": 2, "left": 3}
        self.action_coords = np.array([[-1,0], [0,1], [1,0], [0,-1]], dtype=np.int)
        self.num_actions = len(self.action_dict.keys())

        self.state_dim = (self.height, self.width)
        self.action_dim = (self.num_actions,)
        self.state_action_dim = self.state_dim + self.action_dim

        self.obstacles = []
        self.add_obstacle(6, 5)
        
        self.reset()

    def add_obstacle(self, h, w):
        self.obstacles.append([h, w])

    def get_valid_actions(self):
        actions = []
        h = self.current_state[0]
        w = self.current_state[1]
        if (h > 0): actions.append(self.action_dict["up"])
        if (h < self.height-1): actions.append(self.action_dict["down"])
        if (w > 0): actions.append(self.action_dict["left"])
        if (w < self.width-1): actions.append(self.action_dict["right"])
        actions = np.array(actions, dtype=np.int)
        return actions

    def reset(self):
        self.current_state = np.array([0, 0], dtype=np.int)
        return self.current_state

    def step(self, action):
        self.current_state = np.add(self.current_state, self.action_coords[action])
        if np.array_equal(self.current_state, [self.height-1, self.width-1]):
            reward = 100
            done = True
        elif list(self.current_state) in self.obstacles:
            reward = -10
            done = False
        else:
            reward = -1
            done = False
        return self.current_state, reward, done

