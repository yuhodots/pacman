import numpy as np
from gym import spaces
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing


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


class QlearningAgent(object):
    def __init__(self, n_state, n_action, alpha=0.5, epsilon=1.0, gamma=0.999, seed=42):
        np.random.seed(seed)
        self.n_state = n_state
        self.n_action = n_action
        self.alpha_init = alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros([n_state, n_action])

    def update_q(self, state, action, reward, state_prime, done):
        Q_old = self.Q[state][action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[state_prime])
        td_error = td_target - Q_old
        self.Q[state, action] = Q_old + self.alpha * td_error

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 1.0])

    def update_alpha(self, alpha):
        self.alpha = np.min([alpha, self.alpha_init])

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, high=self.n_action)
        else:
            action = np.argmax(self.Q[state])
        return action


class DoubleQlearningAgent(object):
    def __init__(self, n_state, n_action, alpha=0.5, epsilon=1.0, gamma=0.999, seed=42):
        np.random.seed(seed)
        self.n_state = n_state
        self.n_action = n_action
        self.alpha_init = alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.Q_A = np.zeros([n_state, n_action])
        self.Q_B = np.zeros([n_state, n_action])

    def update_q(self, state, action, reward, state_prime, done):
        if np.random.uniform() < 0.5:
            Q_update = self.Q_A
            Q_target = self.Q_B
        else:
            Q_update = self.Q_B
            Q_target = self.Q_A

        action_prime = np.argmax(Q_update[state_prime])

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * Q_target[state_prime][action_prime]

        Q_old = Q_update[state][action]
        td_error = td_target - Q_old
        Q_update[state, action] = Q_old + self.alpha * td_error

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 1.0])

    def update_alpha(self, alpha):
        self.alpha = np.min([alpha, self.alpha_init])

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, high=self.n_action)
        else:
            action = np.argmax(self.Q_A[state])
        return action


class LinearApprox(object):
    def __init__(self, n_state, n_action, alpha=0.01, epsilon=0.1, gamma=1.0, seed=42):
        np.random.seed(seed)
        self.n_action = n_action
        self.w = np.zeros((n_action, 400))
        self.set_featurizer(n_state)
        self.epsilon = alpha
        self.alpha = epsilon
        self.gamma = gamma

    def set_featurizer(self, n_state):
        obs_space = spaces.Discrete(n_state)
        observation_examples = np.expand_dims(np.array([obs_space.sample() for _ in range(10**6)]), -1)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def featurize_state(self, state):
        scaled = self.scaler.transform(np.expand_dims([state], -1))
        featurized = self.featurizer.transform(scaled)
        return featurized

    def Q(self, state, action):
        value = state.dot(self.w[action])
        return value

    def get_action(self, state, test=False):
        probs = np.ones(self.n_action, dtype=float) * self.epsilon / self.n_action
        best_action = np.argmax([self.Q(state, action) for action in range(self.n_action)])
        probs[best_action] += (1.0 - self.epsilon)
        if test:
            probs = np.zeros(self.n_action, dtype=float)
            probs[best_action] += 1.0
        action = np.random.choice(self.n_action, p=probs)
        return action

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 0.1])

    def update_alpha(self, alpha):
        self.alpha = np.min([alpha, 0.01])