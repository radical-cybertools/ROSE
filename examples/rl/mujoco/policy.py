import numpy as np

class ReinforcePolicy:
    def __init__(self, state_dim, learning_rate=0.01):
        self.state_dim = state_dim
        self.weights = np.random.randn(state_dim) * 0.1
        self.lr = learning_rate

    def select_action(self, state, deterministic=False):
        z = np.dot(self.weights, state)
        mean = np.tanh(z)
        if deterministic:
            return np.clip(mean, -1.0, 1.0)
        else:
            return np.clip(mean + np.random.randn() * 0.1, -1.0, 1.0)

    def log_prob(self, state, action):
        z = np.dot(self.weights, state)
        mean = np.tanh(z)
        std = 0.1
        return -0.5 * ((action - mean)**2 / std**2 + np.log(2 * np.pi * std**2))

    def update(self, episode):
        G = 0
        gamma = 0.99
        returns = []
        for exp in reversed(episode):
            G = exp.reward + gamma * G
            returns.insert(0, G)

        for exp, R in zip(episode, returns):
            state = np.array(exp.state)
            z = np.dot(self.weights, state)
            grad = (1 - np.tanh(z)**2) * state
            logp_grad = grad * (exp.action - np.tanh(z)) / (0.1**2)
            self.weights += self.lr * logp_grad * R
