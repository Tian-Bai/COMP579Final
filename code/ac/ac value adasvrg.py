import argparse
import gymnasium as gym
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchviz import make_dot
import time
import random
import os
import math
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('task', action='store')
parser.add_argument('LR', action='store', type=float)
parser.add_argument('groupsize', action='store', type=int)
parser.add_argument('update', action='store', type=int)
parser.add_argument('runs', action='store', type=int)
parser.add_argument('-e', dest='episodes', action='store', type=int, default=1000)
args = parser.parse_args()

gamma = 0.95

CONTINUOUS_ACTION = False

if args.task == 'acrobot':
    taskname = 'Acrobot-v1'
elif args.task == 'cartpole':
    taskname = 'CartPole-v1'
elif args.task == 'mountaincar':
    taskname = 'MountainCar-v0'
elif args.task == 'pendulum':
    taskname = 'Pendulum-v1'
    CONTINUOUS_ACTION = True
LR = args.LR

sample_env = gym.make(taskname)
eps = np.finfo(np.float32).eps.item()

# SavedAction works as a replay buffer
SavedAction = namedtuple('SavedAction', ['s', 'v', 'a', 'log_p'])

action_dim = sample_env.action_space.n
state_dim  = sample_env.observation_space.shape[0]
del sample_env

random.seed(33)
np.random.seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
torch.manual_seed(33)
torch.cuda.manual_seed(33)
torch.backends.cudnn.deterministic = True

# 2 simple 2-layer NNs for the actor and critic
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
    def __init__(self, lr=LR, beta1=0.9, beta2=0.999):
        self.actor = Actor()
        self.value = Value()
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=LR)

        self.steps = []
        self.rewards = []
        self.latest_steps = []
        self.latest_rewards = []
        self.value_past_grad = []

        self.env = gym.make(taskname)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        prob, val = self.actor(state), self.value(state)

        m = Categorical(prob)
        action = m.sample()
        self.latest_steps.append(SavedAction(state, val, action, m.log_prob(action)))

        return action.item()
    
    def finish_episode(self):
        '''
        The procedure after an episode. We record the trajectories, 
        accumulate the mean gradients w.r.t the snapshot model, without updating the parameters.
        '''
        R = 0
        policy_losses = []
        value_snapshot_losses = []
        returns = deque()

        # calculate the true value using rewards returned from the environment
        for r in self.latest_rewards[::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)

        for (s, v, a, log_p), R in zip(self.latest_steps, returns):
            advantage = R - v.item()
            policy_losses.append(-log_p * advantage)
            # since we don't update now, value is w.r.t the snapshot model
            value_snapshot_losses.append(F.smooth_l1_loss(v, torch.tensor([R])))
        
        self.value.zero_grad()
        self.actor.zero_grad()

        actor_loss = torch.stack(policy_losses).sum()
        value_snapshot_loss = torch.stack(value_snapshot_losses).sum()

        actor_loss.backward()
        value_snapshot_loss.backward()

        # remember the past gradients
        value_grad = [param.grad.clone() for param in self.value.parameters()]
        self.value_past_grad.append(value_grad)

        # update directly for the policy
        self.actor_optimizer.step()

        # remember the trajectories in a group
        self.rewards.append(self.latest_rewards)
        self.steps.append(self.latest_steps)

        # clear the buffer for this episode
        self.latest_rewards = []
        self.latest_steps = []

    #TODO: change update to Adagrad style
    def finish_step(self, update_time, lr=LR):
        '''
        The procedure after a step.
        Now we can sample past episodes and do the corresponding updates.
        '''
        n = len(self.value_past_grad)

        # we first calculate mu.
        value_mu = [torch.zeros_like(param.grad) for param in self.value.parameters()]
        for p in self.value_past_grad:
            for i, g in enumerate(p):
                value_mu[i] += g / n

        # for Adagrad
        G = 0

        for i_update in range(update_time):
            # pick a random previous episode t
            t = np.random.randint(0, n)
            self.value.zero_grad()

            # calculate the current gradient
            R = 0
            value_losses = []
            returns = deque()
            for r in self.rewards[t][::-1]:
                R = r + gamma * R
                returns.appendleft(R)
            returns = torch.tensor(returns)

            for (s, v, a, log_p), R in zip(self.steps[t], returns):
                cur_val = self.value(s)
                value_losses.append(F.smooth_l1_loss(cur_val, torch.tensor([R])))

            value_loss = torch.stack(value_losses).sum()
            value_loss.backward()

            with torch.no_grad():
                update_step = [value_mu[i] - self.value_past_grad[t][i] + p.grad for i, p in enumerate(self.value.parameters())]

                # collect norm and update G
                squares = 0
                for g in update_step:
                    squares = squares + torch.square(g).sum().item()
                G = G + np.sqrt(squares)

                for i, p in enumerate(self.value.parameters()):
                    new_p = p - lr * update_step[i] / np.sqrt(G)
                    p.copy_(new_p)

        self.value_past_grad = []
        self.steps = []
        self.rewards = []


def experiment(episodes=50, groupsize=20, update=30, lr=LR):
    agent = Agent()

    ep_rewards = []

    # we first freeze the model to get a 'full batch' of gradients
    # Then we use SVRG to update the model param multiple times
    for i_step in range(episodes):
        for j_episode in range(groupsize):
            # could change it to a decreasing number?
            state, _ = agent.env.reset()
            ep_reward = 0

            while True:
                action = agent.select_action(state)

                state, reward, term, trunc, _ = agent.env.step(action)
                done = term or trunc
                agent.latest_rewards.append(reward)
                ep_reward += reward
                if done:
                    break
            ep_rewards.append(ep_reward)
            agent.finish_episode()
        agent.finish_step(update, lr)

        # if i_step % 5 == 0:
        #     print('Step {}\tLast reward: {:.2f}'.format(i_step, ep_reward))
    return ep_rewards
    
if __name__ == '__main__':
    # wandb.init(project="Comp579")
    runs = args.runs
    all_rewards = []
    total_episodes = args.episodes
    groupsize = args.groupsize
    episodes = int(math.ceil(total_episodes / groupsize))
    update = args.update

    with Pool() as p:
        all_rewards = p.starmap(experiment, [(episodes, groupsize, update, LR)] * runs)

    np.savetxt(f"ac value adasvrg {args.task} {groupsize} {update} {runs} lr={LR}.txt", np.array(all_rewards))
    
    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)
    # for m in mean:
        # wandb.log(f"reward: {m}")
    plt.figure(figsize=(30, 15))
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.savefig(f'ac value adasvrg {args.task} {groupsize} {update} {runs} lr={LR}.png')