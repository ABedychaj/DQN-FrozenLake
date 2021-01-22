import argparse

import gym
import numpy as np
from torch.autograd import Variable

from dqn.agent import Agent
from model.architecture import Net
from utils import OH, plot_results

parser = argparse.ArgumentParser(description='DQN on FrozenLake8x8 gym testing')
parser.add_argument('--steps', type=int, default=100, help='max steps in episode')
args = parser.parse_args()

if __name__ == "__main__":
    env = gym.make('FrozenLake8x8-v0')
    state_space = env.observation_space.n
    action_space = env.action_space.n

    # no exploration and no scaling the memory
    agent = Agent(None, None, state_space, action_space, Net(state_space, action_space))
    agent.load_model('net_params.pkl')

    env.reset()
    total_test_episodes = 100
    rewards = []
    # do not explore in test
    agent.epsilon = 0

    step_list = []
    reward_list = []

    for episode in range(total_test_episodes):
        state = int(env.reset())
        done = False
        total_rewards = 0
        total_loss = 0
        step = 0

        for step in range(args.steps):
            state = Variable(OH(state, state_space))
            action = agent.select_action(state)

            new_state, reward, done, _ = env.step(action)

            total_rewards += reward

            # If done (if we're dead)
            if done:
                rewards.append(total_rewards)
                break
            state = new_state

        step_list.append(step)
        reward_list.append(total_rewards)
    env.close()

    print('\nSuccessful episodes: {}'.format(np.sum(np.array(reward_list) > 0.0) / total_test_episodes))
    plot_results(total_test_episodes, Steps=step_list, Rewards=reward_list)
