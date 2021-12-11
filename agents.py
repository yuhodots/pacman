import numpy as np


class MCAgent(object):
    def __init__(self, n_state, n_action, epsilon=1.0, alpha=0.1, gamma=0.995, seed=42):
        np.random.seed(seed)
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = epsilon      # epsilon greediness
        self.alpha = alpha          # Q mixing rate
        self.gamma = gamma          # discount factor

        self.Q = np.zeros([n_state, n_action])
        self.samples = []

    def save_sample(self, state, action, reward, done):
        self.samples.append([state, action, reward, done])

    def update_q(self):
        Q_old = self.Q
        g = 0
        G = Q_old

        for t in reversed(range(len(self.samples))):    # for all samples in a reversed way
            state, action, reward, _ = self.samples[t]
            g = reward + self.gamma * g                 # g = r + gamma * g
            G[state][action] = g                        # update G

        self.Q = Q_old + self.alpha * (G - Q_old)       # Update Q
        self.samples = []                               # Empty memory

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 1.0])

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:      # random with epsilon probability
            action = np.random.randint(0, high=self.n_action)
        else:
            action = np.argmax(self.Q[state])       # greedy action
        return action


class SARSAAgent(object):
    def __init__(self, n_state, n_action, epsilon=1.0, alpha=0.1, gamma=0.99, seed=42):
        np.random.seed(seed)
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros([n_state, n_action])

    def update_q(self, state, action, reward, state_prime, action_prime, done):
        Q_old = self.Q[state][action]
        if done:
            td_target = reward      # for the last step, Q = reward
        else:
            td_target = reward + self.gamma * self.Q[state_prime][action_prime]
        td_error = td_target - Q_old
        self.Q[state, action] = Q_old + self.alpha * td_error

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 1.0])       # decay

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:      # random with epsilon probability
            action = np.random.randint(0, high=self.n_action)
        else:
            action = np.argmax(self.Q[state])       # greedy action
        return action
