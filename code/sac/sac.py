'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf

Discrete version reference: 
https://towardsdatascience.com/adapting-soft-actor-critic-for-discrete-action-spaces-a20614d4a50a
'''
 
# https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/sac_discrete.py

import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool
import os
import time
import cProfile

debug = False

GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('task', action='store')
parser.add_argument('LR', action='store', type=float)
parser.add_argument('update', action='store', type=int)
parser.add_argument('runs', action='store', type=int)
parser.add_argument('-e', dest='episodes', action='store', type=int, default=1000)
args = parser.parse_args()

if debug:
    random.seed(33)
    np.random.seed(33)
    os.environ['PYTHONHASHSEED'] = str(33)
    torch.manual_seed(33)
    torch.cuda.manual_seed(33)
    torch.backends.cudnn.deterministic = True

# choose env
if args.task == 'cartpole':
    sample_env = gym.make('CartPole-v1')
elif args.task == 'acrobot':
    sample_env = gym.make('Acrobot-v1')
state_dim  = sample_env.observation_space.shape[0]
action_dim = sample_env.action_space.n  # discrete
del sample_env

# q-network LR, policy-network LR, alpha LR (for entropy)
LR = [args.LR, args.LR, args.LR]

# hyper-parameters for RL training
max_episodes    = args.episodes
batch_size      = 256
update_itr      = args.update
AUTO_ENTROPY    = True
DETERMINISTIC   = False
hidden_dim      = 32
target_entropy  = -1. * action_dim
# target_entropy = 0.98 * -np.log(1 / action_dim)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, num_actions)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        x = self.linear4(x)
        return x
            
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, num_actions)

        self.num_actions = num_actions
        
    def forward(self, state, softmax_dim=-1):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        # x = F.tanh(self.linear4(x))

        probs = F.softmax(self.output(x), dim=softmax_dim)
        
        return probs
    
    def evaluate(self, state, epsilon=1e-8):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        probs = self.forward(state, softmax_dim=-1)
        log_probs = torch.log(probs)

        # Avoid numerical instability. Ref: https://github.com/ku2482/sac-discrete.pytorch/blob/40c9d246621e658750e0a03001325006da57f2d4/sacd/model.py#L98
        z = (probs == 0.0).float() * epsilon
        log_probs = torch.log(probs + z)

        return log_probs
        
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        dist = Categorical(probs)

        if deterministic:
            action = np.argmax(probs.detach().cpu().numpy())
        else:
            action = dist.sample().squeeze().detach().cpu().numpy()
        return action


class SAC_Trainer():
    def __init__(self, replay_buffer, hidden_dim):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.soft_q_optimizer1 = optim.SGD(self.soft_q_net1.parameters(), lr=LR[0])
        self.soft_q_optimizer2 = optim.SGD(self.soft_q_net2.parameters(), lr=LR[0])
        self.policy_optimizer = optim.SGD(self.policy_net.parameters(), lr=LR[1])
        self.alpha_optimizer = optim.SGD([self.log_alpha], lr=LR[2])

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.Tensor(action).to(torch.int64).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        predicted_q_value1 = self.soft_q_net1(state)
        predicted_q_value1 = predicted_q_value1.gather(1, action.unsqueeze(-1))
        predicted_q_value2 = self.soft_q_net2(state)
        predicted_q_value2 = predicted_q_value2.gather(1, action.unsqueeze(-1))
        log_prob = self.policy_net.evaluate(state)
        with torch.no_grad():
            next_log_prob = self.policy_net.evaluate(next_state)
        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

    # Training Q Function
        self.alpha = self.log_alpha.exp()
        target_q_min = (next_log_prob.exp() * (torch.min(self.target_soft_q_net1(next_state), self.target_soft_q_net2(next_state)) - self.alpha * next_log_prob)).sum(dim=-1).unsqueeze(-1)
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        with torch.no_grad():
            predicted_new_q_value = torch.min(self.soft_q_net1(state), self.soft_q_net2(state))
        policy_loss = (log_prob.exp() * (self.alpha * log_prob - predicted_new_q_value)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            self.alpha = 1.
            alpha_loss = 0
        
        # print('q loss: ', q_value_loss1.item(), q_value_loss2.item())
        # print('policy loss: ', policy_loss.item() )

    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
            
        return predicted_new_q_value.mean()

def experiment():
    if args.task == 'cartpole':
        env = gym.make('CartPole-v1')
    elif args.task == 'acrobot':
        env = gym.make('Acrobot-v1')

    replay_buffer_size = 1e6
    replay_buffer = ReplayBuffer(replay_buffer_size)
    sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim)
    rewards = []

    for eps in range(max_episodes):
        state, _ = env.reset(seed=33)
        episode_reward = 0
        
        while True:
            action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            # env.render()       
                
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    sac_trainer.update(batch_size, reward_scale=1., auto_entropy=AUTO_ENTROPY, target_entropy=target_entropy)

            if done:
                break

        print('Episode: ', eps, '| Episode Reward: ', episode_reward)
        rewards.append(episode_reward)
    return rewards

if __name__ == '__main__':
    all_rewards = []

    # for k in range(args.runs):
    #     all_rewards.append(experiment())
    with Pool(processes=10) as p:
        all_rewards = p.starmap(experiment, [()] * args.runs)

    np.savetxt(f'sac {args.task} update={args.update} {args.runs} lr={args.LR}.txt', np.array(all_rewards))

    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)

    plt.figure(figsize=(30, 15))
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.savefig(f'sac {args.task} update={args.update} {args.runs} lr={args.LR}.png')