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
groupsize = 20

svrggroupsize = 20
update = 40
runs = 10
LR = '1e-4'

if __name__ == '__main__':
    ac = np.loadtxt(f"data\\{task}\\lr={LR}\\ac\\ac {task} {runs} lr={LR}.txt")
    adam = np.loadtxt(f"data\\{task}\\lr={LR}\\ADAM\\ac ADAM {task} {runs} lr={LR}.txt")
    ac_value_svrg = np.loadtxt(f"data\\{task}\\lr={LR}\\groupsize={groupsize}\\ac value svrg {task} {groupsize} {update} {runs} lr={LR}.txt")
    ac_value_adasvrg = np.loadtxt(f"data\\{task}\\lr={LR}\\adasvrg groupsize={svrggroupsize}\\ac value adasvrg {task} {svrggroupsize} {update} {runs} lr={LR}.txt")
    plt.figure(figsize=(10, 5))

    ac_mean = np.mean(ac, axis=0)
    ac_std = np.std(ac, axis=0)

    adam_mean = np.mean(adam, axis=0)
    adam_std = np.std(adam, axis=0)

    ac_value_mean = np.mean(ac_value_svrg, axis=0)
    ac_value_std = np.std(ac_value_svrg, axis=0)

    ac_value_adasvrg_mean = np.mean(ac_value_adasvrg, axis=0)
    ac_value_adasvrg_std = np.std(ac_value_adasvrg, axis=0)

    plt.plot(ac_mean, label="SGD")
    plt.fill_between(range(len(ac_mean)), ac_mean + ac_std, ac_mean - ac_std, alpha=0.3)

    plt.plot(adam_mean, label="Adam")
    plt.fill_between(range(len(adam_mean)), adam_mean + adam_std, adam_mean - adam_std, alpha=0.3)

    plt.plot(ac_value_mean, label="SVRG")
    plt.fill_between(range(len(ac_value_mean)), ac_value_mean + ac_value_std, ac_value_mean - ac_value_std, alpha=0.3)
    
    plt.plot(ac_value_adasvrg_mean, label="AdaSVRG")
    plt.fill_between(range(len(ac_value_adasvrg_mean)), ac_value_adasvrg_mean + ac_value_adasvrg_std, ac_value_adasvrg_mean - ac_value_adasvrg_std, alpha=0.3)

    plt.xlabel(f"Comparison of algorithms on function value approximation (group size {groupsize}) on {task.capitalize()} task, lr = {LR}")
    plt.legend()
    plt.savefig(f"ac vs svrg {task} groupsize={groupsize} vs ADAM vs AdaSVRG lr={LR}.png")