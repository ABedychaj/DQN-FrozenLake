import argparse

import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange

from dqn.agent import Agent
from model.architecture import Net
from utils import OH, plot_results

parser = argparse.ArgumentParser(description='DQN on FrozenLake8x8 gym')
parser.add_argument('--e', type=float, default=0.01, help='chance of random action')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--episodes', type=int, default=2000, help='number of episodes')
parser.add_argument('--steps', type=int, default=100, help='max steps in episode')
args = parser.parse_args()

if __name__ == "__main__":
    env = gym.make('FrozenLake8x8-v0')

    # Initialize history memory
    step_list = []
    reward_list = []
    loss_list = []
    e_list = []

    state_space = env.observation_space.n
    action_space = env.action_space.n

    agent = Agent(args.e, args.gamma, state_space, action_space, Net(state_space, action_space))
    agent.train(True)
    # model.load_state_dict(torch.load('net_params.pkl'))
    loss = nn.MSELoss()
    optimizer = optim.Adam(agent.model.parameters(), lr=args.lr)

    for i in trange(args.episodes):
        state = int(env.reset())
        reward_all = 0
        done = False
        s = 0
        total_loss = 0

        for s in range(args.steps):
            state = Variable(OH(state, state_space))

            # propose an action
            action = agent.select_action(state)

            # what are the consequences of taking that action?
            new_state, reward, done, _ = env.step(action)

            # if we're dead
            if done and reward == 0.0:
                reward = -1

            # store memories for experience replay
            Q1 = agent.model(Variable(OH(new_state, state_space)))
            targetQ = agent.remember(Q1, action, reward)

            # optimize predicting rewards
            output = agent.model(state)
            train_loss = loss(output, targetQ)
            total_loss += train_loss.data

            agent.model.zero_grad()
            train_loss.backward()
            optimizer.step()

            # move to next state
            reward_all += reward
            state = new_state

            # decrease epsilon after success
            if done:
                if reward > 0:
                    agent.epsilon *= 0.9 + 1E-6  # always explore a bit during training
                break

        # logging epochs
        loss_list.append(total_loss / s)
        step_list.append(s)
        reward_list.append(reward_all)
        e_list.append(agent.epsilon)

    print('\nSuccessful episodes: {}'.format(np.sum(np.array(reward_list) > 0.0) / args.episodes))
    plot_results(args.episodes, Steps=step_list, Rewards=reward_list, Loss=loss_list, Epsilon=e_list)
    agent.save_model('net_params.pkl')
