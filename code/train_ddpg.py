import os
import json
import pickle

import time
import datetime

from collections import deque

import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import helpers as fu
import control_helpers as ch

from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.99,
            help="discount factor.")
parser.add_argument("--tau", type=float, default=0.005,
            help="the target network updatting rate.")
parser.add_argument("--replay-size", type=int, default=20000,
            help="maximal replay buffer size.")
parser.add_argument("--batch-size", type=int, default=64,
            help="batch size of replay.")
parser.add_argument("--lr-actor", type=float, default=0.0005,
            help="learning rate of the actor network.")
parser.add_argument("--lr-critic", type=float, default=0.0005,
            help="learning rate of the critic network.")
parser.add_argument("--save-folder", type=str, default="logs/ddpg",
            help="Where to save the trained model.")
parser.add_argument("--load-folder", type=str, default='',
            help="Where to load trained model.")
parser.add_argument("--verbose-freq", type=int, default=100, 
                    help="show the training process.")
parser.add_argument("--clear-output", type=int, default=20,
                   help="clear console output.")
parser.add_argument("--max-episode", type=int, default=30000,
                   help="Maximal trained episodes.")
parser.add_argument("--train-critic-from", type=int, default=20,
                    help="train critic from episode n.")
parser.add_argument("--train-actor-from", type=int, default=40,
                    help="train actor from episode n.")

args = parser.parse_args()

gamma = args.gamma
tau = args.tau
lr_actor = args.lr_actor
lr_critic = args.lr_critic
max_episode = args.max_episode
train_critic_from = args.train_critic_from
train_actor_from = args.train_actor_from

replay_buffer = ReplayBuffer(max_size=args.replay_size, batch_size=args.batch_size)
