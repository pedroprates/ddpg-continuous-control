import random
import copy
import numpy as np

class OrsnteinUhlenbeck:
    """ Ornstein-Uhlenbeck noise process """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        
        random.seed(seed)

        self.reset()

    def reset(self):
        """ Reset the internal state (= noise) to mean (mu) """
        self.state = copy.copy(self.mu)

    def sample(self):
        """ Update the internal state and return it as a noise sample """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random()for i in range(len(x))])
        self.state = x + dx

        return self.state