# DQN-FrozenLake
An aproach to create Q-learning reinforcement learning algorithm for FrozenLake gym. 

## Getting started

Install all required libraries:

```pip install -r requirement.txt```

## Training
How to run training:

```train.py --e 0.9 --lr 0.8 --gamma 0.95 --episodes 25000 --steps 100```

Full list of parameters:

```
optional arguments:
  -h, --help           show this help message and exit
  --e E                chance of random action
  --lr LR              learning rate
  --gamma GAMMA        gamma
  --episodes EPISODES  number of episodes
  --steps STEPS        max steps in episode
```

## Testing
How to run testing:

```test.py --steps 100```

Full list of parameters:
```
DQN on FrozenLake8x8 gym testing

optional arguments:
  -h, --help     show this help message and exit
  --steps STEPS  max steps in episode

```

Already trained model with the architecture declared in `net_params.pkl`

## Inspired by
* https://cwong8.github.io/projects/FrozenLake/
* https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
* https://github.com/mynameisvinn/DQN