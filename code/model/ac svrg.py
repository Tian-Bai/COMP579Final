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
from torchviz import make_dot

# Cart Pole

gamma = 0.95

env = gym.make('CartPole-v1')
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['action', 'log_prob', 'state', 'val'])

action_dim = env.action_space.n
state_dim  = env.observation_space.shape[0]

# these contains steps/rewards for many episodes
steps = []
rewards = []

# these contains steps/rewards for only the latest episode
latest_steps = []
latest_rewards = []

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

# value optimizer?

value_pass_grad = [] # previous gradients of value model
actor_pass_grad = [] # previous gradients of actor model

def select_action(state):
    state = torch.from_numpy(state).float()
    prob, val = actor(state), value(state)

    m = Categorical(prob)
    action = m.sample()
    latest_steps.append(SavedAction(action, m.log_prob(action), state, val))

    return action.item()

def finish_episode():
    '''
    The procedure after an episode. We record the trajectories, 
    accumulate the mean gradients w.r.t the snapshot model, without updating the parameters.
    '''
    R = 0
    policy_losses = []
    value_snapshot_losses = []
    returns = deque()

    global latest_rewards, latest_steps

    # calculate the true value using rewards returned from the environment
    for r in latest_rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.appendleft(R)

    for (_, log_prob, _, val), R in zip(latest_steps, returns):
        advantage = R - val.item()
        policy_losses.append(-log_prob * advantage)
        # since we don't update now, value is w.r.t the snapshot model
        value_snapshot_losses.append(F.smooth_l1_loss(val, torch.tensor([R])))
    
    actor_loss = torch.stack(policy_losses).sum()
    value_snapshot_loss = torch.stack(value_snapshot_losses).sum()
    actor_loss.backward()
    value_snapshot_loss.backward()

    # remember the past gradients
    value_grad = [param.grad.clone() for param in value.parameters()]
    value_pass_grad.append(value_grad)
    actor_grad = [param.grad.clone() for param in actor.parameters()]
    actor_pass_grad.append(actor_grad)

    # remember the trajectories
    rewards.append(latest_rewards)
    steps.append(latest_steps)

    # clear the buffer for this episode
    latest_rewards = []
    latest_steps = []


def finish_step(update_time=20, lr=3e-2):
    '''
    The procedure after a step.
    Now we can sample past episodes and do the corresponding updates.
    '''
    n = len(value_pass_grad)

    # we first calculate mu.
    value_mu = [0 for param in value.parameters()]
    for p in value_pass_grad:
        for i, g in enumerate(p):
            value_mu[i] += g / n
    actor_mu = [0 for param in actor.parameters()]
    for p in actor_pass_grad:
        for i, g in enumerate(p):
            actor_mu[i] += g / n

    for i_update in range(update_time):
        # pick a random previous episode t
        t = np.random.randint(0, n)

        # calculate the current gradient
        R = 0
        policy_losses = []
        value_losses = []
        returns = deque()
        for r in rewards[t][::-1]:
            R = r + gamma * R
            returns.appendleft(R)

        for (action, _, state, _), R in zip(steps[t][::-1], returns):
            cur_val = value(state)
            cur_advantage = R - cur_val
            m = Categorical(actor(state))
            cur_log_prob = m.log_prob(action)
            value_losses.append(F.smooth_l1_loss(cur_val, torch.tensor([R])))
            policy_losses.append(-cur_log_prob * cur_advantage)

        '''TODO'''
        # cannot get current gradient???
        value_loss = torch.stack(value_losses).sum()
        value_loss.retain_grad()
        value_loss.backward(retain_graph=True)
        print(value_loss.grad)
        policy_loss = torch.stack(policy_losses).sum()
        policy_loss.retain_grad()
        policy_loss.backward(retain_graph=True)
        print(policy_loss.grad)
        
        # add or subtract?
        for i, param in enumerate(value.parameters()):
            param.data -= lr * (value_mu[i] - value_pass_grad[t][i] + value_loss.grad[i])
        
        for i, param in enumerate(actor.parameters()):
            param.data -= lr * (actor_mu[i] - actor_pass_grad[t][i] + policy_loss.grad[i])


def main():
    ep_rewards = []

    # we first freeze the model to get a 'full batch' of gradients
    # Then we use SVRG to update the model param multiple times
    for i_step in range(300):
        for j_episode in range(10):
            # could change 10 to a decreasing number?
            state, _ = env.reset()
            ep_reward = 0

            while True:
                action = select_action(state)

                state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                latest_rewards.append(reward)
                ep_reward += reward
                if done:
                    break
            ep_rewards.append(ep_reward)
            finish_episode()
        finish_step()

        if i_step % 10 == 0:
            print('Step {}\tLast reward: {:.2f}'.format(i_step, ep_reward))
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(ep_rewards)
    plt.savefig('ac svrg cartpole.png')
    
if __name__ == '__main__':
    main()