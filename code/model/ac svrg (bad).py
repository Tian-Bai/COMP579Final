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
import svrg

# Cart Pole

gamma = 0.95

env = gym.make('CartPole-v1')
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'value_snapshot'])

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
value_snapshot = Value()
value = Value()

actor_optimizer = optim.Adam(actor.parameters(), lr=3e-2)

value_optimizer_snapshot = svrg.SVRG_Snapshot(value_snapshot.parameters())
value_optimizer = svrg.SVRG_k(value.parameters(), lr=3e-2, weight_decay=1e-4)

def select_action(state):
    state = torch.from_numpy(state).float()
    prob, val, val_snapshot = actor(state), value(state), value_snapshot(state)

    m = Categorical(prob)
    action = m.sample()
    steps.append(SavedAction(m.log_prob(action), val, val_snapshot))

    return action.item()

def finish_episode(groupsize):
    # now we have *groupsize* trajectories
    # accumulate their gradients

    global rewards, steps

    for j_groupidx in range(groupsize):
        R = 0
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = deque()

        # calculate the true value using rewards returned from the environment
        for r in rewards[j_groupidx][::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.appendleft(R)

        # normalization deleted

        for (log_prob, value, value_snapshot), R in zip(steps, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value_snapshot, torch.tensor([R])))

        # sum up all the values of policy_losses and value_losses
        actor_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()

        # accumulate the gradients to get the mean grad mu
        snapshot_loss = value_loss / groupsize
        snapshot_loss.backward(retain_graph=True)

    # pass the current optimizer parameters
    u = value_optimizer_snapshot.get_param_groups()
    value_optimizer.set_u(u)

    for j_groupidx in range(groupsize):
        R = 0
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        value_snapshot_losses = []
        returns = deque()

        # calculate the true value using rewards returned from the environment
        for r in rewards[j_groupidx][::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.appendleft(R)

        # normalization deleted

        for (log_prob, value, value_snapshot), R in zip(steps, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            value_snapshot_losses.append(F.smooth_l1_loss(value_snapshot, torch.tensor([R])))

        # sum up all the values of policy_losses and value_losses
        actor_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        value_snapshot_loss = torch.stack(value_snapshot_losses).sum()

        value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)

        value_optimizer_snapshot.zero_grad()
        value_snapshot_loss.backward(retain_graph=True)

        value_optimizer.step(value_optimizer_snapshot.get_param_groups())

        # for the actor - unmodified
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()

    value_optimizer_snapshot.set_param_groups(value_optimizer.get_param_groups())
        
    rewards = []
    steps = []

def main():
    ep_rewards = []
    groupsize = 10

    for i_episode in range(30):
        for j_groupidx in range(groupsize):
            state, _ = env.reset()
            ep_reward = 0

            while True:
                rewards_j = []
                action = select_action(state)

                state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                rewards_j.append(reward)
                ep_reward += reward
                if done:
                    rewards.append(rewards_j)
                    break
                ep_rewards.append(ep_reward)
        finish_episode(groupsize)

        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}'.format(i_episode, ep_reward))

    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(ep_rewards)
    plt.savefig('ac cartpole.png')
    
if __name__ == '__main__':
    main()