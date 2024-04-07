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
import random
import os
import time
import wandb
import math

# Cart Pole

gamma = 0.95

# taskname = 'CartPole-v1'
taskname = 'Acrobot-v1'
if taskname == 'Acrobot-v1':
    displayname = 'acrobot'
    LR = 1e-3
elif taskname == 'CartPole-v1':
    displayname = 'cartpole'
    LR = 1e-2

debug = False

env = gym.make(taskname)
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

action_dim = env.action_space.n
state_dim  = env.observation_space.shape[0]

if debug:
    random.seed(33)
    np.random.seed(33)
    os.environ['PYTHONHASHSEED'] = str(33)
    torch.manual_seed(33)
    torch.cuda.manual_seed(33)
    torch.backends.cudnn.deterministic = True
    
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

class Agent():
    def __init__(self, lr=LR):
        self.actor = Actor()
        self.value = Value()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LR)

        self.steps = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        prob, val = self.actor(state), self.value(state)

        m = Categorical(prob)
        action = m.sample()
        self.steps.append(SavedAction(m.log_prob(action), val))

        return action.item()

    def finish_episode(self):
        R = 0
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = deque()

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)

        # normalization deleted

        for (log_prob, val), R in zip(self.steps, returns):
            advantage = R - val.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(val, torch.tensor([R])))

        # reset gradients
        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        actor_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()

        # perform backprop
        actor_loss.backward()
        value_loss.backward()

        self.actor_optimizer.step()
        self.value_optimizer.step()

        self.rewards = []
        self.steps = []

def experiment(episodes=1000, lr=LR):
    agent = Agent()
    ep_rewards = []

    for i_episode in range(episodes):
        state, _ = env.reset(seed=33)

        ep_reward = 0

        while True:
            action = agent.select_action(state)

            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        ep_rewards.append(ep_reward)
        agent.finish_episode()

        if i_episode % 5 == 0:
            print('Episode {}\tLast reward: {:.2f}'.format(i_episode, ep_reward))
    return ep_rewards
    
if __name__ == '__main__':
    # wandb.init(project="Comp579")
    all_rewards = []
    for k in range(10):
        all_rewards.append(experiment(1000))
    np.savetxt(f"ac ADAM {displayname} lr={LR}.txt", np.array(all_rewards))
    
    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)
    # for m in mean:
    #     wandb.log(f"reward: {m}")
    plt.figure(figsize=(30, 15))
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.savefig(f'ac ADAM {displayname} lr={LR}.png')