import random

import numpy as np
import torch
from torch.autograd import Variable


class Agent(object):
    def __init__(self, epsilon, gamma, state_space, action_space, model):
        # Chance of random action
        self.epsilon = epsilon
        self.state_space = state_space
        self.action_space = action_space
        self.model = model
        # Reward Discount
        self.gamma = gamma
        self.Q = None

    def select_action(self, state):
        self.Q = self.model(state)
        if np.random.rand(1) < self.epsilon:
            action = random.randrange(self.action_space)
        else:
            _, action = torch.max(self.Q, 1)
            action = action.data[0].item()
        return action

    def remember(self, Q1, action, reward):
        maxQ1, _ = torch.max(Q1.data, 1)
        maxQ1 = torch.FloatTensor(maxQ1)

        targetQ = Variable(self.Q.data, requires_grad=False)
        targetQ[0, action] = reward + torch.mul(maxQ1, self.gamma)
        return targetQ

    def save_model(self, param):
        torch.save(self.model.state_dict(), param)

    def load_model(self, param):
        self.model.load_state_dict(torch.load(param))

    def train(self, train):
        if train:
            self.model.train()
        else:
            self.model.eval()
