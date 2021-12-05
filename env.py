import gym
import numpy as np
from gym import spaces


class GridEnv(gym.Env):
    def __init__(self):
        self.world_shape = [11, 11]
        self.init_agent_pos = [10, 5]
        self.init_ghost_pos = [[1, 5], [6, 5]]
        self.agent_pos = self.init_agent_pos[:]
        self.ghost_pos = self.init_ghost_pos[:]

        self.action_space = spaces.Discrete(4)
        self.obs_space = spaces.Discrete(11*11)

        self.wall_pos = [[0, 5], [1, 1], [1, 2], [1, 3], [1, 7],
                         [1, 8], [1, 9], [2, 5], [3, 1], [3, 3],
                         [3, 5], [3, 7], [3, 9], [4, 1], [4, 3],
                         [4, 7], [4, 9], [5, 1], [5, 3], [5, 4],
                         [5, 5], [5, 6], [5, 7], [5, 9], [6, 1],
                         [6, 9], [7, 3], [7, 4], [7, 6], [7, 7],
                         [8, 1], [8, 4], [8, 6], [8, 9], [9, 1],
                         [9, 2], [9, 4], [9, 6], [9, 8], [9, 9]]
        self.ghost_road = [[1, 4], [1, 5], [1, 6], [6, 2], [6, 3],
                           [6, 4], [6, 5], [6, 6], [6, 7], [6, 8]]
        self.star_pos = [[0, 0], [0, 10]]
        self.coin_pos = self.set_coin_pos()

    def set_coin_pos(self):
        coin_pos = [[i, j] for i in range(11) for j in range(11)]
        for item in self.wall_pos:
            coin_pos.remove(item)
        for item in self.ghost_road:
            coin_pos.remove(item)
        for item in self.star_pos:
            coin_pos.remove(item)
        coin_pos.remove(self.init_agent_pos)
        return coin_pos

    def step(self, action):
        self.agent_pos = self.step_agent(action)

        info = {}
        results = (self._get_obs(), self._get_reward(), self._is_done(), info)

        self.world[np.where(self.world == -1)] = 0
        self.world[self.agent_pos[0], self.agent_pos[1]] = -1

        return results

    def step_ghost(self):
        ghost_pos = self.ghost_pos
        ghost_prime = []

        if not (self.agent_pos in self.ghost_pos):
            for i in range(len(ghost_pos)):
                ghost = ghost_pos[i]
                if [ghost[0], ghost[1] - 1] in self.wall_pos:
                    ghost = [ghost[0], ghost[1] + 1]
                elif [ghost[0], ghost[1] + 1] in self.wall_pos:
                    ghost = [ghost[0], ghost[1] - 1]
                else:
                    move = np.random.choice([1, -1], 1, p=[0.5, 0.5])[0]
                    ghost = [ghost[0], ghost[1] + move]
                ghost_prime.append(ghost)

            # update world table
            for v in ghost_prime:
                self.world[v[0], v[1]] = 3
            for v in ghost_pos:
                self.world[v[0], v[1]] = 0

            self.ghost_pos = ghost_prime
        return ghost_prime

    def step_agent(self, action):
        agent_pos = self.agent_pos
        pos = agent_pos[:]

        if pos in self.ghost_pos:
            pass
        else:
            if action == 0:     # up
                agent_pos[0] -= 1
            elif action == 1:   # down
                agent_pos[0] += 1
            elif action == 2:   # left
                agent_pos[1] -= 1
            elif action == 3:   # right
                agent_pos[1] += 1
            else:
                raise Exception("the action is not defined")

            # out of the map
            if agent_pos[0] < 0 or agent_pos[0] > 10:
                agent_pos = pos
            if agent_pos[1] < 0 or agent_pos[1] > 10:
                agent_pos = pos
            # wall
            if agent_pos in self.wall_pos:
                agent_pos = pos

        return agent_pos

    def reset(self):
        self.world = np.zeros(self.world_shape)
        self.world[self.init_agent_pos] = -1
        for v in self.coin_pos:     # coin
            self.world[v[0], v[1]] = 1
        for v in self.star_pos:     # star
            self.world[v[0], v[1]] = 2
        for v in self.init_ghost_pos:   # ghost
            self.world[v[0], v[1]] = 3
        for v in self.wall_pos:     # wall
            self.world[v[0], v[1]] = 4
        self.agent_pos = self.init_agent_pos[:]
        self.ghost_pos = self.init_ghost_pos[:]
        return self._get_obs()

    def render(self):
        render_str = ""
        for i in range(self.world_shape[0]):
            for j in range(self.world_shape[1]):
                if self.world[i, j] == 1:
                    item = "$ "
                elif self.world[i, j] == 2:
                    item = "* "
                elif self.world[i, j] == 3:
                    item = "G "
                elif self.world[i, j] == 4:
                    item = "# "
                elif self.world[i, j] == -1:
                    if self.agent_pos in self.ghost_pos:
                        item = "X "
                    else:
                        item = "C "
                else:
                    item = "0 "
                render_str += "{}".format(item)
            render_str += "\n"

        print(render_str)

    def close(self):
        pass

    def _get_obs(self):
        return self.agent_pos

    def _get_reward(self):
        pos = self.agent_pos
        if self.world[pos[0], pos[1]] == 1:
            return 1    # coin point
        elif self.world[pos[0], pos[1]] == 2:
            return 50   # star point
        elif self.world[pos[0], pos[1]] == 3:
            return -50  # meet ghost
        else:
            return 0

    def _is_done(self):
        if self.agent_pos in self.ghost_pos:    # meet ghost
            return True
        if not np.any(self.world == 2):      # no star
            return True
        else:
            return False
