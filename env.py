import gym
import numpy as np
from gym import spaces


class SmallGridEnv(gym.Env):
    def __init__(self):
        self.world_shape = [5, 5]
        self.init_agent_pos = [4, 0]
        self.init_star_pos = [0, 2]
        self.agent_pos = self.init_agent_pos[:]
        self.ghost_pos = [[1, 1], [0, 4]]
        self.wall_pos = [[3, 0], [3, 1], [3, 3], [1, 2], [2, 3]]
        self.star_pos = self.init_star_pos[:]

        n = 5 * 5 - len(self.wall_pos)
        self.observation_space = spaces.Discrete(n)
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        self.agent_pos = self.step_agent(action)

        info = {}
        results = (self._get_obs(), self._get_reward(), self._is_done(), info)

        self.world[np.where(self.world == -1)] = 0
        self.world[self.agent_pos[0], self.agent_pos[1]] = -1

        return results

    def step_agent(self, action):
        agent_pos = self.agent_pos
        pos = agent_pos[:]

        if action == 0:  # up
            agent_pos[0] -= 1
        elif action == 1:  # down
            agent_pos[0] += 1
        elif action == 2:  # left
            agent_pos[1] -= 1
        elif action == 3:  # right
            agent_pos[1] += 1
        else:
            raise Exception("the action is not defined")

        # out of the map
        if agent_pos[0] < 0 or agent_pos[0] > 4:
            agent_pos = pos
        if agent_pos[1] < 0 or agent_pos[1] > 4:
            agent_pos = pos
        # wall
        if agent_pos in self.wall_pos:
            agent_pos = pos

        return agent_pos

    def reset(self):
        self.world = np.zeros(self.world_shape)
        self.world[self.init_agent_pos[0], self.init_agent_pos[1]] = -1
        self.world[self.init_star_pos[0], self.init_star_pos[1]] = 1
        for v in self.ghost_pos:  # ghost
            self.world[v[0], v[1]] = 2
        for v in self.wall_pos:  # wall
            self.world[v[0], v[1]] = 3
        self.agent_pos = self.init_agent_pos[:]
        self.star_pos = self.init_star_pos[:]
        return self._get_obs()

    def render(self):
        render_str = ""
        for i in range(self.world_shape[0]):
            for j in range(self.world_shape[1]):
                if self.world[i, j] == 1:
                    item = "* "
                elif self.world[i, j] == 2:
                    item = "G "
                elif self.world[i, j] == 3:
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
        obs_agent = self.agent_pos[0] * self.world_shape[1] + self.agent_pos[1]
        obs_wall = len([item for item in self.wall_pos if (item[0] < self.agent_pos[0]) or
                        ((item[0] == self.agent_pos[0]) and (item[1] < self.agent_pos[1]))])
        obs = obs_agent - obs_wall
        return obs

    def _get_reward(self):
        pos = self.agent_pos
        if self.world[pos[0], pos[1]] == 1:
            self.star_pos.remove(pos)
            return 50  # star point
        elif self.world[pos[0], pos[1]] == 2:
            return -100  # meet ghost
        else:
            return -1

    def _is_done(self):
        if self.agent_pos in self.ghost_pos:  # meet ghost
            return True
        if not np.any(self.world == 1):  # no star
            return True
        else:
            return False


class BigGridEnv(gym.Env):
    def __init__(self):
        self.world_shape = [11, 11]
        self.init_agent_pos = [10, 5]
        self.init_ghost_pos = [[1, 5], [6, 5]]
        self.init_star_pos = [[0, 0], [2, 10], [4, 4], [10, 2]]
        self.agent_pos = self.init_agent_pos[:]
        self.ghost_pos = self.init_ghost_pos[:]
        self.star_pos = self.init_star_pos[:]
        self.wall_pos = [[0, 5], [1, 1], [1, 2], [1, 3], [1, 7],
                         [1, 8], [1, 9], [2, 5], [3, 1], [3, 3],
                         [3, 5], [3, 7], [3, 9], [4, 1], [4, 3],
                         [4, 7], [4, 9], [5, 1], [5, 3], [5, 4],
                         [5, 5], [5, 6], [5, 7], [5, 9], [6, 1],
                         [6, 9], [7, 3], [7, 4], [7, 6], [7, 7],
                         [8, 1], [8, 4], [8, 6], [8, 9], [9, 1],
                         [9, 2], [9, 4], [9, 6], [9, 8], [9, 9]]
        self.ghost_a_road = [[1, 4], [1, 5], [1, 6]]
        self.ghost_b_road = [[6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8]]

        n = (11 * 11 - len(self.wall_pos)) * (2 ** len(self.star_pos)) * len(self.ghost_a_road) * len(self.ghost_b_road)
        self.observation_space = spaces.Discrete(n)
        self.action_space = spaces.Discrete(4)

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
                self.world[v[0], v[1]] = 2
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
            if action == 0:  # up
                agent_pos[0] -= 1
            elif action == 1:  # down
                agent_pos[0] += 1
            elif action == 2:  # left
                agent_pos[1] -= 1
            elif action == 3:  # right
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
        self.world[self.init_agent_pos[0], self.init_agent_pos[1]] = -1
        for v in self.init_star_pos:  # star
            self.world[v[0], v[1]] = 1
        for v in self.init_ghost_pos:  # ghost
            self.world[v[0], v[1]] = 2
        for v in self.wall_pos:  # wall
            self.world[v[0], v[1]] = 3
        self.agent_pos = self.init_agent_pos[:]
        self.ghost_pos = self.init_ghost_pos[:]
        self.star_pos = self.init_star_pos[:]
        return self._get_obs()

    def render(self):
        render_str = ""
        for i in range(self.world_shape[0]):
            for j in range(self.world_shape[1]):
                if self.world[i, j] == 1:
                    item = "* "
                elif self.world[i, j] == 2:
                    item = "G "
                elif self.world[i, j] == 3:
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
        obs_agent = self.agent_pos[0] * self.world_shape[1] + self.agent_pos[1]
        obs_wall = len([item for item in self.wall_pos if (item[0] < self.agent_pos[0]) or
                      ((item[0] == self.agent_pos[0]) and (item[1] < self.agent_pos[1]))])
        obs_star = self._get_obs_star()
        obs_ghost = self._get_obs_star()

        n_grid = self.world_shape[0] * self.world_shape[1]
        n_wall = len(self.wall_pos)
        n_obs_star = 2 ** len(self.init_star_pos)
        n_obs_ghost = len(self.ghost_a_road) * len(self.ghost_b_road)

        obs = np.ravel_multi_index((obs_agent - obs_wall, obs_star, obs_ghost),
                                   (n_grid - n_wall, n_obs_star, n_obs_ghost))
        return obs

    def _get_obs_star(self):
        tuple_obs = tuple([int(item in self.star_pos) for item in self.init_star_pos])
        tuple_obs_all = tuple([2 for _ in range(len(self.init_star_pos))])
        obs_star = np.ravel_multi_index(tuple_obs, tuple_obs_all)
        return obs_star

    def _get_obs_ghost(self):
        ghost_a_idx = self.ghost_a_road.index(self.ghost_pos[0])
        ghost_b_idx = self.ghost_b_road.index(self.ghost_pos[1])
        obs_ghost = np.ravel_multi_index((ghost_a_idx, ghost_b_idx),
                                         (len(self.ghost_a_road), len(self.ghost_b_road)))
        return obs_ghost

    def _get_reward(self):
        pos = self.agent_pos
        if self.world[pos[0], pos[1]] == 1:
            self.star_pos.remove(pos)
            return 50  # star point
        elif self.world[pos[0], pos[1]] == 2:
            return -100  # meet ghost
        else:
            return -1

    def _is_done(self):
        if self.agent_pos in self.ghost_pos:  # meet ghost
            return True
        if not np.any(self.world == 1):  # no star
            return True
        else:
            return False


class UnistEnv(gym.Env):
    def __init__(self):
        self.world_shape = [11, 11]
        self.init_agent_pos = [10, 5]
        self.init_ghost_pos = [[0, 3], [5, 8]]
        self.init_star_pos = [[1, 8], [3, 2], [3, 5], [7, 2], [7, 7]]
        self.agent_pos = self.init_agent_pos[:]
        self.ghost_pos = self.init_ghost_pos[:]
        self.star_pos = self.init_star_pos[:]
        self.wall_pos = [[0, 7], [0, 8], [0, 9], [1, 1], [1, 3],
                         [1, 5], [1, 7], [2, 1], [2, 3], [2, 5],
                         [2, 7], [2, 8], [2, 9], [3, 1], [3, 3],
                         [3, 9], [4, 1], [4, 2], [4, 3], [4, 5],
                         [4, 7], [4, 8], [4, 9], [5, 5], [6, 1],
                         [6, 2], [6, 3], [6, 5], [6, 7], [6, 8],
                         [6, 9], [7, 1], [7, 3], [7, 5], [7, 8],
                         [8, 1], [8, 3], [8, 5], [8, 8], [9, 1],
                         [9, 3], [9, 5], [9, 8]]
        self.ghost_a_road = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]]
        self.ghost_b_road = [[5, 6], [5, 7], [5, 8], [5, 9], [5, 10]]

        n = (11 * 11 - len(self.wall_pos)) * (2 ** len(self.star_pos)) * len(self.ghost_a_road) * len(self.ghost_b_road)
        self.observation_space = spaces.Discrete(n)
        self.action_space = spaces.Discrete(4)

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
                if ([ghost[0], ghost[1] - 1] in self.wall_pos) or (ghost[1] - 1 < 0):
                    ghost = [ghost[0], ghost[1] + 1]
                elif ([ghost[0], ghost[1] + 1] in self.wall_pos) or (ghost[1] + 1 > 10):
                    ghost = [ghost[0], ghost[1] - 1]
                else:
                    move = np.random.choice([1, -1], 1, p=[0.5, 0.5])[0]
                    ghost = [ghost[0], ghost[1] + move]
                ghost_prime.append(ghost)

            # update world table
            for v in ghost_prime:
                self.world[v[0], v[1]] = 2
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
            if action == 0:  # up
                agent_pos[0] -= 1
            elif action == 1:  # down
                agent_pos[0] += 1
            elif action == 2:  # left
                agent_pos[1] -= 1
            elif action == 3:  # right
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
        self.world[self.init_agent_pos[0], self.init_agent_pos[1]] = -1
        for v in self.init_star_pos:  # star
            self.world[v[0], v[1]] = 1
        for v in self.init_ghost_pos:  # ghost
            self.world[v[0], v[1]] = 2
        for v in self.wall_pos:  # wall
            self.world[v[0], v[1]] = 3
        self.agent_pos = self.init_agent_pos[:]
        self.ghost_pos = self.init_ghost_pos[:]
        self.star_pos = self.init_star_pos[:]
        return self._get_obs()

    def render(self):
        render_str = ""
        for i in range(self.world_shape[0]):
            for j in range(self.world_shape[1]):
                if self.world[i, j] == 1:
                    item = "* "
                elif self.world[i, j] == 2:
                    item = "G "
                elif self.world[i, j] == 3:
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
        obs_agent = self.agent_pos[0] * self.world_shape[1] + self.agent_pos[1]
        obs_wall = len([item for item in self.wall_pos if (item[0] < self.agent_pos[0]) or
                        ((item[0] == self.agent_pos[0]) and (item[1] < self.agent_pos[1]))])
        obs_star = self._get_obs_star()
        obs_ghost = self._get_obs_star()

        n_grid = self.world_shape[0] * self.world_shape[1]
        n_wall = len(self.wall_pos)
        n_obs_star = 2 ** len(self.init_star_pos)
        n_obs_ghost = len(self.ghost_a_road) * len(self.ghost_b_road)

        obs = np.ravel_multi_index((obs_agent - obs_wall, obs_star, obs_ghost),
                                   (n_grid - n_wall, n_obs_star, n_obs_ghost))
        return obs

    def _get_obs_star(self):
        tuple_obs = tuple([int(item in self.star_pos) for item in self.init_star_pos])
        tuple_obs_all = tuple([2 for _ in range(len(self.init_star_pos))])
        obs_star = np.ravel_multi_index(tuple_obs, tuple_obs_all)
        return obs_star

    def _get_obs_ghost(self):
        ghost_a_idx = self.ghost_a_road.index(self.ghost_pos[0])
        ghost_b_idx = self.ghost_b_road.index(self.ghost_pos[1])
        obs_ghost = np.ravel_multi_index((ghost_a_idx, ghost_b_idx),
                                         (len(self.ghost_a_road), len(self.ghost_b_road)))
        return obs_ghost

    def _get_reward(self):
        pos = self.agent_pos
        if self.world[pos[0], pos[1]] == 1:
            self.star_pos.remove(pos)
            return 50  # star point
        elif self.world[pos[0], pos[1]] == 2:
            return -100  # meet ghost
        else:
            return -1

    def _is_done(self):
        if self.agent_pos in self.ghost_pos:  # meet ghost
            return True
        if not np.any(self.world == 1):  # no star
            return True
        else:
            return False
