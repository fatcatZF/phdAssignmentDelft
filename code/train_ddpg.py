"""
Implementation of DDPG algorithm for
continuous traffic flow control.
"""

from ast import arg
import os
import json
import pickle
from sqlite3 import Timestamp

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
trained_from_scratch = True

if args.save_folder and args.load_folder == '':
    exp_count = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = "{}/exp{}/".format(arg.save_folder, timestamp)
    os.mkdir(save_folder)
    trained_from_scratch = True

if args.load_folder != '':
    """load trained models and continue training them"""
    save_folder = args.load_folder
    trained_from_scratch = False

info_file = os.path.join(save_folder, "info.json")
actor_file = os.path.join(save_folder, "actor.pt")
critic_file = os.path.join(save_folder, "critic.pt")
actor_target_file = os.path.join(save_folder, "actor_target.pt")
critic_target_file = os.path.join(save_folder, "critic_target.pt")
replay_file = os.path.join(save_folder, "replay_buffer.pkl")
log_file = os.path.join(save_folder, "log.txt")
log = open(log_file, 'a')

if trained_from_scratch:
    print("train from scratch!")
    actor = Actor()
    critic = Critic()
    actor_target = Actor()
    critic_target = Critic()
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    init_episode = 0
    init_score = 0.
    
else:
    actor = torch.load(actor_file)
    actor_target = torch.load(actor_target_file)
    critic = torch.load(critic_file)
    critic_target = torch.load(critic_target_file)
    with open(info_file, 'r') as f:
        info = json.load(f)
    init_score = info["score"]
    log_read = open(log_file, 'r')
    log_lines = log_read.readlines()
    init_episode = int(log_lines[-1][0])
    log_read.close()
    with open(replay_file, 'rb') as f:
        replay_buffer = pickle.load(f)
    
    
optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
loss_critic = nn.SmoothL1Loss()


def act(state):
    """
    use the actor net to compute the actor 
    given the state
    args:
        state: the current state, a torch tensor with 
               the shape [n_batch=1, n_s]
    return: 
        a: the corresponding action, range from -1 to 1
        0: the log probability
    """
    a = actor(state.float()).cpu().item()
    return a, 0


def compute_target_flow(density, previous_target_flow):
    """
    map the action to corresponding target flow
    """
    previous_target_flow = previous_target_flow/1800.
    #normalize the value to [0,1]
    density_bin = binary(density)
    density_bin.append(previous_target_flow)
    state = torch.tensor(density_bin)
    action, log_prob = act(state.unsqueeze(0))
    action_p = action+1 #range from 0 to 2
    target_flow = 870*action_p+60
    target_flow = min(target_flow, 1800)
    target_flow = max(target_flow, 60)
    return target_flow, action, state

    


