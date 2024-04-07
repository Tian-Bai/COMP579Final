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
import time
import random
import os
import wandb

'''
To avoid reruning the code for multiple times, 
we save the rewards of different models as .txt files and generate the plot here.
'''

task = 'acrobot' # cartpole, acrobot
algo = 'ac' # ac, ac value svrg
groupsize = 20
update = 40

if task == 'acrobot':
    LR = 1e-4
elif task == 'cartpole':
    LR = 1e-3

if __name__ == '__main__':
    ac = np.loadtxt(f"data\\{task}\\ac\\ac {task}.txt")
    ac_value_svrg = np.loadtxt(f"data\\{task}\\groupsize={groupsize}\\ac value svrg {groupsize} {update} {task}.txt")
    plt.figure(figsize=(10, 5))
    
    ac_mean = np.mean(ac, axis=0)
    ac_std = np.std(ac, axis=0)

    ac_value_mean = np.mean(ac_value_svrg, axis=0)
    ac_value_std = np.std(ac_value_svrg, axis=0)

    plt.plot(ac_mean, label="AC")
    plt.fill_between(range(len(ac_mean)), ac_mean + ac_std, ac_mean - ac_std, alpha=0.3, color='blue')

    plt.plot(ac_value_mean, label="SVRG")
    plt.fill_between(range(len(ac_value_mean)), ac_value_mean + ac_value_std, ac_value_mean - ac_value_std, alpha=0.3, color='red')
    
    plt.xlabel(f"AC vs. AC with SVRG on function value approximation (group size {groupsize}) on {task.capitalize()} task, lr = {LR}")
    plt.legend()
    plt.savefig(f"ac vs svrg {task} groupsize={groupsize}.png")