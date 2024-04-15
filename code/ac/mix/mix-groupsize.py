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
groupsize = [20, 20, 30]
update = [40, 60, 60]
runs = 50
LR = 0.01

if __name__ == '__main__':
    ac_value_svrg1 = np.loadtxt(f"pic & data\\ac value svrg {task} {groupsize[0]} {update[0]} {runs} lr={LR}.txt")
    ac_value_svrg2 = np.loadtxt(f"pic & data\\ac value svrg {task} {groupsize[1]} {update[1]} {runs} lr={LR}.txt")
    ac_value_svrg3 = np.loadtxt(f"pic & data\\ac value svrg {task} {groupsize[2]} {update[2]} {runs} lr={LR}.txt")
    ac_value_adasvrg1 = np.loadtxt(f"pic & data\\ac value adasvrg {task} {groupsize[0]} {update[0]} {runs} lr={LR}.txt")
    ac_value_adasvrg2 = np.loadtxt(f"pic & data\\ac value adasvrg {task} {groupsize[1]} {update[1]} {runs} lr={LR}.txt")
    ac_value_adasvrg3 = np.loadtxt(f"pic & data\\ac value adasvrg {task} {groupsize[2]} {update[2]} {runs} lr={LR}.txt")
    plt.figure(figsize=(20, 10))

    mean_1 = np.mean(ac_value_svrg1, axis=0)
    std_1 = np.std(ac_value_svrg1, axis=0)

    mean_2 = np.mean(ac_value_svrg2, axis=0)
    std_2 = np.std(ac_value_svrg2, axis=0)

    mean_3 = np.mean(ac_value_svrg3, axis=0)
    std_3 = np.std(ac_value_svrg3, axis=0)

    mean_4 = np.mean(ac_value_adasvrg1, axis=0)
    std_4 = np.std(ac_value_adasvrg1, axis=0)

    mean_5 = np.mean(ac_value_adasvrg2, axis=0)
    std_5 = np.std(ac_value_adasvrg2, axis=0)

    mean_6 = np.mean(ac_value_adasvrg3, axis=0)
    std_6 = np.std(ac_value_adasvrg3, axis=0)

    plt.plot(mean_1, label=f"SVRG, groupsize={groupsize[0]}, update={update[0]}")
    plt.fill_between(range(len(mean_1)), mean_1 + std_1, mean_1 - std_1, alpha=0.3)

    plt.plot(mean_2, label=f"SVRG, groupsize={groupsize[1]}, update={update[1]}")
    plt.fill_between(range(len(mean_2)), mean_2 + std_2, mean_2 - std_2, alpha=0.3)

    plt.plot(mean_3, label=f"SVRG, groupsize={groupsize[2]}, update={update[2]}")
    plt.fill_between(range(len(mean_3)), mean_3 + std_3, mean_3 - std_3, alpha=0.3)

    plt.plot(mean_4, label=f"AdaSVRG, groupsize={groupsize[0]}, update={update[0]}")
    plt.fill_between(range(len(mean_4)), mean_4 + std_4, mean_4 - std_4, alpha=0.3)

    plt.plot(mean_5, label=f"AdaSVRG, groupsize={groupsize[1]}, update={update[1]}")
    plt.fill_between(range(len(mean_5)), mean_5 + std_5, mean_5 - std_5, alpha=0.3)

    plt.plot(mean_6, label=f"AdaSVRG, groupsize={groupsize[2]}, update={update[2]}")
    plt.fill_between(range(len(mean_6)), mean_6 + std_6, mean_6 - std_6, alpha=0.3)
    
    

    plt.xlabel(f"Comparison of different groupsizes and updates on AdaSVRG Actor-Critic for {task.capitalize()} task, lr = {LR}")
    plt.legend()
    plt.savefig(f"comp {task} groupsizes & updates lr={LR}.png")