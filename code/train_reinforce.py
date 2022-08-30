"""Implementation of 
   reinforce algorithm (Monte-Carlo Policy Gradient)
   for Traffic Signal Optimization
   """


import os
import json

import time
import datetime

from collections import deque

import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import helpers as fu
import control_helpers as ch

from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.99,
                   help="discount factor.")
parser.add_argument("--lr", type=float, default=1e-3,
                   help="learning rate.")
parser.add_argument("--save-folder", type=str, default="logs/reinforce",
                   help="Where to save the trained model.")
parser.add_argument("--load-folder", type=str, default='',
                    help="load trained model")
parser.add_argument("--verbose-freq", type=int, default=100, 
                    help="show the training process.")
parser.add_argument("--clear-output", type=int, default=20,
                   help="clear console output.")
parser.add_argument("--max-episode", type=int, default=30000,
                   help="Maximal trained episodes.")


args = parser.parse_args()

gamma = args.gamma #discount factor
lr = args.lr #learning rate
max_episode = args.max_episode


if args.save_folder:
    exp_count = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    info_file = os.path.join(save_folder, "info.json")
    policy_file = os.path.join(save_folder, "policy.pt")
    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, 'w')
    

if args.load_folder != '':
    #load trained model and info
    model_path = os.path.join(args.load_folder, "policy.pt")
    info_path = os.path.join(args.load_folder, "info.json")
    policy = torch.load(model_path)
    with open(info_path, 'r') as f:
        info = json.load(f)
    init_episode = info["episode"]
    init_score = info["score"]
else:
    #train from scratch
    print("train from scratch!")
    policy = DMLP() #initialize the policy network
    init_episode = 0
    init_score = 0.

optimizer = optim.Adam(policy.parameters(), lr=lr) #use Adam optimizer




def act(state):
    """
    use the policy network to compute
    the prob distribution of the actions.
    Sample the actions based on their probs
    args:
        policy: a policy network
        state: current state, a torch tensor with shape [n_batch, n_s]
    """
    probs = policy(state.float()).cpu()
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)




def compute_target_flow(density, previous_target_flow):
    """
    compute target flow given the density on the main road 
    and previous target flow on the ramp
    """
    #normalize the density and previous target flow
    #density = density/100. #The maximal density is set to 500
    #previous_target_flow = previous_target_flow/1800.
    #state = torch.tensor([density, previous_target_flow]).float()
    density_bin = binary(density)
    state = torch.tensor(density_bin)
    action, log_prob = act(state.unsqueeze(0))
    target_flow = action*100
    return target_flow, log_prob, state



def simulate():
    print("Initial episode: {:2d}".format(init_episode),
          "Initial score: {:.4f}".format(init_score))
    episode = init_episode
    max_score = init_score
    scores_deque = deque(maxlen=100)
    print("episode,score,score_100_ave", file=log)
    log.flush()
    while True:
        episode = episode+1
        densities, saved_log_probs, _ = ch.run(compute_target_flow, verbose=False)

        flows_episode = [estimate_flow(d) for d in densities]
        score = np.mean(flows_episode)
        scores_deque.append(score)
        average_score = np.mean(scores_deque)

        print("{},{},{}".format(episode, score, average_score), file=log)
        log.flush()

        rewards = [flow_reward(fl) for fl in flows_episode]

        #ignore the reward before any action and the last action prob
        rewards = rewards[1:]
        saved_log_probs = saved_log_probs[:-1]
        discounts = [gamma**i for i in range(len(rewards))]

        R = sum([a*b for a,b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob*R)
        
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if score > max_score:
            max_score = score
            info = {"episode":episode, "score":score}
            with open(info_file, 'w') as f:
                json.dump(info, f)
            torch.save(policy, policy_file)
            

        if episode % args.clear_output == 0:
            os.system("clear")
        
        if episode % args.verbose_freq == 0:
             print("Episode {:2d} Score: {:.4f}".format(episode, score))

        if episode >= max_episode:
            break

        



simulate()
log.close()




        













