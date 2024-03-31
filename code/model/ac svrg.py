import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple, deque
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

gamma = 0.95

env = gym.make('CartPole-v1')
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

action_dim = env.action_space.n
state_dim  = env.observation_space.shape[0]

steps = []
rewards = []

class Actor(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        probs = F.softmax(self.output(outs))
        return probs

class Value(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value

actor = Actor()
value = Value()

actor_optimizer = optim.Adam(actor.parameters(), lr=3e-2)
value_optimizer = optim.Adam(value.parameters(), lr=3e-2)

def select_action(state):
    state = torch.from_numpy(state).float()
    prob, val = actor(state), value(state)

    m = Categorical(prob)
    action = m.sample()
    steps.append(SavedAction(m.log_prob(action), val))

    return action.item()


def finish_episode():
    R = 0
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = deque()

    global rewards, steps

    # calculate the true value using rewards returned from the environment
    for r in rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.appendleft(R)

    # normalization deleted

    for (log_prob, value), R in zip(steps, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    actor_optimizer.zero_grad()
    value_optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    actor_loss = torch.stack(policy_losses).sum()
    value_loss = torch.stack(value_losses).sum()

    # perform backprop
    actor_loss.backward()
    value_loss.backward()
    actor_optimizer.step()
    value_optimizer.step()

    rewards = []
    steps = []

def main():
    ep_rewards = []

    for i_episode in range(300):
        state, _ = env.reset()
        ep_reward = 0

        while True:
            action = select_action(state)

            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            rewards.append(reward)
            ep_reward += reward
            if done:
                break
        ep_rewards.append(ep_reward)
        finish_episode()

        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}'.format(i_episode, ep_reward))
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(ep_rewards)
    plt.savefig('ac cartpole.png')
    
if __name__ == '__main__':
    main()