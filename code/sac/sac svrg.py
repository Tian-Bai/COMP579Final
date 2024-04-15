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

if debug:
    random.seed(33)
    np.random.seed(33)
    os.environ['PYTHONHASHSEED'] = str(33)
    torch.manual_seed(33)
    torch.cuda.manual_seed(33)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('task', action='store')
parser.add_argument('LR', action='store', type=float)
parser.add_argument('groupsize', action='store', type=int)
parser.add_argument('update', action='store', type=int)
parser.add_argument('runs', action='store', type=int)
parser.add_argument('-e', dest='episodes', action='store', type=int, default=200)
args = parser.parse_args()

# choose env
if args.task == 'cartpole':
    sample_env = gym.make('CartPole-v1')
elif args.task == 'acrobot':
    sample_env = gym.make('Acrobot-v0')
state_dim  = sample_env.observation_space.shape[0]
action_dim = sample_env.action_space.n  # discrete
del sample_env

# q-network LR, policy-network LR, alpha LR (for entropy)
LR = [args.LR, args.LR, args.LR]

# hyper-parameters for RL training
max_episodes    = args.episodes
batch_size      = 256
groupsize       = args.groupsize
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

        self.soft_q1_past_grad = []
        self.soft_q2_past_grad = []
        self.policy_past_grad = []
        self.alpha_past_grad = []

        self.sampled_state = []
        self.sampled_next_state = []
        self.sampled_action = []
        self.sampled_reward = []
        self.sampled_done = []

    
    def calc_grad(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.Tensor(action).to(torch.int64).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # store them for the later svrg step
        self.sampled_state.append(state)
        self.sampled_next_state.append(next_state)
        self.sampled_action.append(action)
        self.sampled_reward.append(reward)
        self.sampled_done.append(done)

        predicted_q_value1 = self.soft_q_net1(state)
        predicted_q_value1 = predicted_q_value1.gather(1, action.unsqueeze(-1))
        predicted_q_value2 = self.soft_q_net2(state)
        predicted_q_value2 = predicted_q_value2.gather(1, action.unsqueeze(-1))
        log_prob = self.policy_net.evaluate(state)
        with torch.no_grad():
            next_log_prob = self.policy_net.evaluate(next_state)
        # reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

    # Calculating Q gradient
        self.alpha = self.log_alpha.exp()
        target_q_min = (next_log_prob.exp() * (torch.min(self.target_soft_q_net1(next_state), self.target_soft_q_net2(next_state)) - self.alpha * next_log_prob)).sum(dim=-1).unsqueeze(-1)
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        # calculate the gradients and store them
        self.soft_q_net1.zero_grad()
        q_value_loss1.backward()

        q_value_grad1 = [p.grad for p in self.soft_q_net1.parameters()]
        self.soft_q1_past_grad.append(q_value_grad1)

        self.soft_q_net2.zero_grad()
        q_value_loss2.backward()
        q_value_grad2 = [p.grad for p in self.soft_q_net2.parameters()] 
        self.soft_q2_past_grad.append(q_value_grad2)

    # Calculating policy gradient
        with torch.no_grad():
            predicted_new_q_value = torch.min(self.soft_q_net1(state), self.soft_q_net2(state))
        policy_loss = (log_prob.exp() * (self.alpha * log_prob - predicted_new_q_value)).sum(dim=-1).mean()
        
        self.policy_net.zero_grad()
        policy_loss.backward()
        policy_grad = [p.grad for p in self.policy_net.parameters()]
        # self.policy_optimizer.step()
        self.policy_past_grad.append(policy_grad)

    # Calculating alpha gradient
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.log_alpha.grad.zero_() # zero_grad() a single tensor
            alpha_loss.backward()
            alpha_grad = [self.log_alpha.grad]
            self.alpha_past_grad.append(alpha_grad)
        else:
            self.alpha = 1.
            alpha_loss = 0
            
        return predicted_new_q_value.mean()

    def update(self, update_itr, lr=LR, auto_entropy=True, gamma=0.99, soft_tau=1e-2):
        n = len(self.soft_q1_past_grad)

        q_value_mu1 = [torch.zeros_like(p.grad) for p in self.soft_q_net1.parameters()]
        q_value_mu2 = [torch.zeros_like(p.grad) for p in self.soft_q_net2.parameters()]
        policy_mu = [torch.zeros_like(p.grad) for p in self.policy_net.parameters()]
        log_alpha_mu = [torch.zeros_like(self.log_alpha.grad)]

        for p in self.soft_q1_past_grad:
            for i, g in enumerate(p):
                q_value_mu1[i] += g / n
        for p in self.soft_q2_past_grad:
            for i, g in enumerate(p):
                q_value_mu2[i] += g / n
        for p in self.policy_past_grad:
            for i, g in enumerate(p):
                policy_mu[i] += g / n
        for p in self.alpha_past_grad: # a trivial loop tho
            for i, g in enumerate(p):
                log_alpha_mu[i] += g / n
        
        for i_update in range(update_itr):
            t = np.random.randint(0, n)

            # Same process as in calc_grad()...
            # calculate the current gradient
            predicted_q_value1 = self.soft_q_net1(self.sampled_state[t].detach())
            predicted_q_value1 = predicted_q_value1.gather(1, self.sampled_action[t].unsqueeze(-1).detach())
            predicted_q_value2 = self.soft_q_net2(self.sampled_state[t].detach())
            predicted_q_value2 = predicted_q_value2.gather(1, self.sampled_action[t].unsqueeze(-1).detach())
            log_prob = self.policy_net.evaluate(self.sampled_state[t].detach())
            with torch.no_grad():
                next_log_prob = self.policy_net.evaluate(self.sampled_next_state[t].detach())

            self.alpha = self.log_alpha.exp()
            target_q_min = (next_log_prob.exp() * (torch.min(self.target_soft_q_net1(self.sampled_next_state[t]), self.target_soft_q_net2(self.sampled_next_state[t])) - self.alpha * next_log_prob)).sum(dim=-1).unsqueeze(-1)
            target_q_value = self.sampled_reward[t] + (1 - self.sampled_done[t]) * gamma * target_q_min # if done==1, only reward
            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
            q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

            # perform svrg update on the two q networks:
            # calculate the gradients and store them
            self.soft_q_net1.zero_grad()
            self.soft_q_net2.zero_grad()
            q_value_loss1.backward()
            q_value_loss2.backward()

            with torch.no_grad():
                for i, p in enumerate(self.soft_q_net1.parameters()):
                    new_p = p - lr[0] * (q_value_mu1[i] - self.soft_q1_past_grad[t][i] + p.grad)
                    p.copy_(new_p)

                for i, p in enumerate(self.soft_q_net2.parameters()):
                    new_p = p - lr[0] * (q_value_mu2[i] - self.soft_q2_past_grad[t][i] + p.grad)
                    p.copy_(new_p)
            
            # notice that here, the loss is w.r.t the two updated q networks, as in the vanilla sac
            with torch.no_grad():
                predicted_new_q_value = torch.min(self.soft_q_net1(self.sampled_state[t]), self.soft_q_net2(self.sampled_state[t]))
            policy_loss = (log_prob.exp() * (self.alpha * log_prob - predicted_new_q_value)).sum(dim=-1).mean()

            self.policy_net.zero_grad()
            policy_loss.backward()

            with torch.no_grad():
                for i, p in enumerate(self.policy_net.parameters()):
                    new_p = p - lr[1] * (policy_mu[i] - self.policy_past_grad[t][i] + p.grad)
                    p.copy_(new_p)
            
            if auto_entropy is True:
                alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
                self.log_alpha.grad.zero_() 
                alpha_loss.backward()
                
                new_log_alpha = self.log_alpha - lr[2] * (log_alpha_mu[0] - self.alpha_past_grad[t][0] + self.log_alpha.grad)
                self.log_alpha.data.copy_(new_log_alpha)

            # do the soft update now
            for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
            for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
        
        # clear the buffers now
        self.soft_q1_past_grad = []
        self.soft_q2_past_grad = []
        self.policy_past_grad = []
        self.alpha_past_grad = []
        self.sampled_state = []
        self.sampled_action = []
        self.sampled_next_state = []
        self.sampled_reward = []
        self.sampled_done = []

def experiment():
    if args.task == 'cartpole':
        env = gym.make('CartPole-v1')
    elif args.task == 'acrobot':
        env = gym.make('Acrobot-v0')

    replay_buffer_size = 1e6
    replay_buffer = ReplayBuffer(replay_buffer_size)
    sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim)
    rewards = []

    for eps in range(max_episodes):
        state, _ = env.reset(seed=33)
        episode_reward = 0
        
        while True:
            action = sac_trainer.policy_net.get_action(state, deterministic=DETERMINISTIC)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            # env.render()       
                
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) > batch_size:
                # here, use svrg style upate
                for i in range(groupsize):
                    _ = sac_trainer.calc_grad(batch_size, reward_scale=1., auto_entropy=AUTO_ENTROPY, target_entropy=target_entropy)
                sac_trainer.update(update_itr, auto_entropy=AUTO_ENTROPY)

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

    np.savetxt(f'sac svrg {args.task} groupsize={args.groupsize} update={args.update} {args.runs} lr={args.LR}.txt', np.array(all_rewards))

    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)

    plt.figure(figsize=(30, 15))
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.savefig(f'sac svrg {args.task} groupsize={args.groupsize} update={args.update} {args.runs} lr={args.LR}.png')