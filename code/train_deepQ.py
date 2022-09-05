"""
Implementation of Deep Q training
"""

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
from torch.distributions import Categorical

import helpers as fu
import control_helpers as ch

from models import *


parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.99,
                   help="discount factor.")
parser.add_argument("--replay-size", type=int, default=20000,
                   help="maximal replay buffer size.")
parser.add_argument("--batch-size", type=int, default=64,
                   help="batch size of replay.")
parser.add_argument("--lr", type=float, default=0.0005,
                   help="learning rate.")
parser.add_argument("--save-folder", type=str, default="logs/deepQ",
                   help="Where to save the trained model.")
parser.add_argument("--load-folder", type=str, default='',
                    help="load trained model")
parser.add_argument("--verbose-freq", type=int, default=100, 
                    help="show the training process.")
parser.add_argument("--clear-output", type=int, default=20,
                   help="clear console output.")
parser.add_argument("--max-episode", type=int, default=30000,
                   help="Maximal trained episodes.")
parser.add_argument("--eps-max", type=float, default=1.0, 
                    help="Max epsilon of epsilon greedy.")
parser.add_argument("--eps-min", type=float, default=0.1,
                    help="Min epsilon of epsilon greedy.")
parser.add_argument("--eps-steps", type=float, default=3000.0,
                    help="Epsilon greedy steps.")
parser.add_argument("--train_from", type=int, default=20,
                    help="train from episode n.")
parser.add_argument("--tupdate-freq", type=int, default=50,
                    help="target update frequence.")

args = parser.parse_args()

gamma = args.gamma #discount factor
lr = args.lr #learning rate
max_episode = args.max_episode

replay_buffer = ReplayBuffer(max_size=args.replay_size, batch_size=args.batch_size)

if args.save_folder:
    exp_count = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    info_file = os.path.join(save_folder, "info.json")
    policy_file = os.path.join(save_folder, "policy.pt")
    target_file = os.path.join(save_folder, "target.pt")
    replay_file = os.path.join(save_folder, "replay_buffer.pkl")
    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, 'w')

if args.load_folder != '':
    #load trained model and info
    model_path = os.path.join(args.load_folder, "policy.pt")
    target_path = os.path.join(args.load_folder, "target.pt")
    replay_path = os.path.join(args.load_folder, "replay_buffer.pkl")
    info_path = os.path.join(args.load_folder, "info.json")
    policy = torch.load(model_path)
    target = torch.load(target_path)
    with open(info_path, 'r') as f:
        info = json.load(f)
    init_episode = info["episode"]
    init_score = info["score"]
    init_eps = info["eps"]
    with open(replay_path, 'rb') as f:
        replay_buffer = pickle.load(f)

else:
    #train from scratch
    print("train from scratch!")
    policy = DMLPQ() #initialize the policy network
    target = DMLPQ() #initialize the target network
    target.load_state_dict(policy.state_dict())
    init_episode = 0
    init_score = 0.
    init_eps = args.eps_max

eps_interval = (args.eps_max-args.eps_min)


optimizer = optim.Adam(policy.parameters(), lr=lr) #use Adam optimizer
loss_criterion = nn.SmoothL1Loss()


def act(state):
    """
    use the policy net to compute the Q-values of actions
    return the action corresponding to the max Q value
    args:
       policy: a policy network
       state: current state, a torch tensor with shape [n_batch=1, n_s]
    """
    Q = policy(state.float()).cpu() #shape of Q: [n_batch=1, n_actions]
    action = torch.argmax(Q) #shape:[n_batch=1]
    return action.item(), 0


def compute_target_flow(density, previous_target_flow):
    """
    compute target flow given the 
     density on the main road and 
     the previous target flow on the ramp
    """
    #density = density/100. #The maximal density is set to 100
    previous_target_flow = previous_target_flow/1800.
    density_bin = binary(density)
    density_bin.append(previous_target_flow)
    state = torch.tensor(density_bin)
    action, log_prob = act(state.unsqueeze(0))
    target_flow = action*100
    # Check max and min
    target_flow = min(target_flow,1800)
    target_flow= max(target_flow, 60)
    return target_flow, action, state #return the current state

def compute_random_target_flow(density, previous_target_flow):
    """
    compute target flow randomly
    """
    #density = density/100.
    previous_target_flow = previous_target_flow/1800.
    density_bin = binary(density)
    density_bin.append(previous_target_flow)
    state = torch.tensor(density_bin)
    action = random.randint(0,18) #get random action
    target_flow = action*100
    # Check max and min
    target_flow = min(target_flow,1800)
    target_flow= max(target_flow, 60)
    return target_flow, action, state #return the current state
    



def update_target():
    target.load_state_dict(policy.state_dict())
    


def replay():
    """
    sample transitions from the replay buffer and train the policy network
    """
    batch = replay_buffer.get_minibatch()
    batch_trans = list(map(list, zip(*batch)))
    states = torch.stack(batch_trans[0], dim=0)
    actions = batch_trans[1]
    batch_size = len(actions)
    batch_indices = range(batch_size)
    next_states = torch.stack(batch_trans[2],dim=0)
    rewards = torch.tensor(batch_trans[3])
    is_dones = torch.tensor(batch_trans[4])

    #predict the Q values
    Q_predicted = policy(states.float())
    #get predicted Q(s,a)
    Q_predicted = Q_predicted[batch_indices, actions]

    #get target Q values
    with torch.no_grad():
        target.eval()
        Q_next = target(next_states.float())
        Q_next_max = torch.max(Q_next, -1).values
        Q_next_max[is_dones] = 0.
        Q_target = gamma*Q_next_max+rewards

    #train policy network
    loss = loss_criterion(Q_predicted, Q_target)
    optimizer.zero_grad()
    loss.backward()
    for param in policy.parameters():
            param.grad.data.clamp_(-1, 1)
    optimizer.step()
 


def simulate():
    print("Initial episode: {:2d}".format(init_episode),
          "Initial score: {:.4f}".format(init_score))
    episode = init_episode
    max_score = init_score
    eps = init_eps
    scores_deque = deque(maxlen=100)
    print("episode,score,score_100_ave", file=log)
    log.flush()
    guess = True #take random action
    while True:
        episode = episode+1
        if random.random()>eps:
            guess = False
            densities, actions, states = ch.run(compute_target_flow, verbose=False)
        else:
            guess = True
            densities, actions, states = ch.run(compute_random_target_flow, verbose=False)
        

        flows_episode = [estimate_flow(d) for d in densities] #estimated flow
        score = np.mean(flows_episode)
        scores_deque.append(score)
        average_score = np.mean(scores_deque)

        eps -= eps_interval/args.eps_steps
        eps = max(eps, args.eps_min)

        if score > max_score and (not guess):
            max_score = score
            info = {"episode":episode, "score":score, "eps":eps}
            with open(info_file, 'w') as f:
                json.dump(info, f)
            torch.save(policy, policy_file)
            torch.save(target, target_file)
            with open(replay_file, 'wb') as f:
                pickle.dump(replay_buffer, f)


        print("{},{},{}".format(episode, score, average_score), file=log)
        log.flush()

        #rewards = [speedMulCount_reward(r) for r in speed_mul_counts]
        rewards = [flow_reward(fl) for fl in flows_episode]

        #ignore the reward before any action and the last action prob
        rewards = rewards[1:]
        actions = actions[:-1]
        next_states = states[1:]
        states = states[:-1]
        is_dones = [False]*len(rewards)
        is_dones[-1] = True


        replay_buffer.add_experience(states, actions, next_states, rewards, is_dones)

        

        if episode > args.train_from:
            replay()

        if episode % args.tupdate_freq == 0:
            update_target()

        if episode % args.verbose_freq == 0:
             print("Episode {:2d} Score: {:.4f}".format(episode, score))

        if episode % args.clear_output == 0:
            os.system("clear")

        if episode >= max_episode:
            break

        

        
simulate()
log.close()








    

    





    






