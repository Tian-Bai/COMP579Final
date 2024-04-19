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

task = 'cartpole' # cartpole, acrobot

if task == 'acrobot':
    groupsize = [20, 20]
    update = [40, 60]
    LR = [0.0001, 0.001, 0.0003, 0.0001] 
elif task == 'cartpole':
    groupsize = [30, 20]
    update = [60, 40]
    LR = [0.001, 0.01, 0.001, 0.001]
runs = 50


if __name__ == '__main__':
    ac = np.loadtxt(f"pic & data\\ac {task} {runs} lr={LR[0]}.txt")
    adam = np.loadtxt(f"pic & data\\ac ADAM {task} {runs} lr={LR[1]}.txt")
    ac_value_svrg = np.loadtxt(f"pic & data\\ac value svrg {task} {groupsize[0]} {update[0]} {runs} lr={LR[2]}.txt")
    ac_value_adasvrg = np.loadtxt(f"pic & data\\ac value adasvrg {task} {groupsize[1]} {update[1]} {runs} lr={LR[3]}.txt")
    plt.figure(figsize=(14, 7))

    ac_mean = np.mean(ac, axis=0)
    ac_std = np.std(ac, axis=0)

    adam_mean = np.mean(adam, axis=0)
    adam_std = np.std(adam, axis=0)

    ac_value_mean = np.mean(ac_value_svrg, axis=0)
    ac_value_std = np.std(ac_value_svrg, axis=0)

    ac_value_adasvrg_mean = np.mean(ac_value_adasvrg, axis=0)
    ac_value_adasvrg_std = np.std(ac_value_adasvrg, axis=0)

    plt.plot(ac_mean, label=f"SGD, lr={LR[0]}")
    plt.fill_between(range(len(ac_mean)), ac_mean + ac_std, ac_mean - ac_std, alpha=0.3)

    plt.plot(adam_mean, label=f"Adam, lr={LR[1]}")
    plt.fill_between(range(len(adam_mean)), adam_mean + adam_std, adam_mean - adam_std, alpha=0.3)

    plt.plot(ac_value_mean, label=f"SVRG, groupsize={groupsize[0]}, update={update[0]}, lr={LR[2]}")
    plt.fill_between(range(len(ac_value_mean)), ac_value_mean + ac_value_std, ac_value_mean - ac_value_std, alpha=0.3)
    
    plt.plot(ac_value_adasvrg_mean, label=f"AdaSVRG, groupsize={groupsize[1]}, update={update[1]}, lr={LR[3]}")
    plt.fill_between(range(len(ac_value_adasvrg_mean)), ac_value_adasvrg_mean + ac_value_adasvrg_std, ac_value_adasvrg_mean - ac_value_adasvrg_std, alpha=0.3)

    # plt.xlabel(f"Comparison of Actor-Critic with different optimizers on {task.capitalize()} task")
    plt.legend()
    plt.savefig(f"ac vs adam vs svrg vs adasvrg on {task}.png")