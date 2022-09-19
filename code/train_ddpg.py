"""
Implementation of DDPG algorithm for
continuous traffic flow control.
"""

from ast import arg
import os
import json
import pickle
import re
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
parser.add_argument("--noise-std", type=float, default=0.2,
            help="Standard deviation of the exploration noise.")
parser.add_argument("--replay-size", type=int, default=20000,
            help="maximal replay buffer size.")
parser.add_argument("--batch-size", type=int, default=64,
            help="batch size of replay.")
parser.add_argument("--lr-actor", type=float, default=0.0007,
            help="learning rate of the actor network.")
parser.add_argument("--lr-critic", type=float, default=0.001,
            help="learning rate of the critic network.")
parser.add_argument("--save-folder", type=str, default="logs/ddpg",
            help="Where to save the trained model.")
parser.add_argument("--load-folder", type=str, default='',
            help="Where to load trained model.")
parser.add_argument("--verbose-freq", type=int, default=2, 
                    help="show the training process.")
parser.add_argument("--clear-output", type=int, default=20,
                   help="clear console output.")
parser.add_argument("--max-episode", type=int, default=10000,
                   help="Maximal trained episodes.")
parser.add_argument("--train-critic-from", type=int, default=10,
                    help="train critic from episode n.")
parser.add_argument("--train-actor-from", type=int, default=20,
                    help="train actor from episode n.")
parser.add_argument("--debug", action="store_true", default=False,
        help="Whether debug the training process.")

args = parser.parse_args()

gamma = args.gamma
tau = args.tau
lr_actor = args.lr_actor
lr_critic = args.lr_critic
max_episode = args.max_episode
train_critic_from = args.train_critic_from
train_actor_from = args.train_actor_from
if_debug = args.debug

replay_buffer = ReplayBuffer(max_size=args.replay_size, batch_size=args.batch_size)
ou_noise = OUActionNoise(0, args.noise_std)
# Exploration noise

trained_from_scratch = True

if args.save_folder and args.load_folder == '':
    exp_count = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = "{}/exp{}/".format(args.save_folder, timestamp)
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
if if_debug:
    debug_file = os.path.join(save_folder, "log_debug.txt")
    log_debug = open(debug_file, 'w')

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
    actor.eval()
    a = actor(state.float()).cpu().item()
    return a, 0


def compute_target_flow(density, previous_target_flow, noise=None):
    """
    map the action to corresponding target flow
    """
    previous_target_flow = previous_target_flow/1800.
    #normalize the value to [0,1]
    density_bin = binary(density)
    density_bin.append(previous_target_flow)
    state = torch.tensor(density_bin)
    action, log_prob = act(state.unsqueeze(0))
    if noise is not None:
        # add exploration noise
        action = action + noise
    action = np.clip(action, -1, 1) #clip the range to [-1,1]
    action_p = action+1 #change range from 0 to 2
    target_flow = 870*action_p+60
    #target_flow = 870*action+930
    target_flow = min(target_flow, 1800)
    target_flow = max(target_flow, 60)
    return target_flow, action, state


def update_target(episode):
    """
    update the actor and critic targets
    """
    if episode >= train_actor_from:
        for target_param, param in zip(actor_target.parameters(),
                                   actor.parameters()):
            target_param.data.copy_(target_param.data*(1-tau)+param.data*tau)
    if episode >= train_critic_from:
        for target_param, param in zip(critic_target.parameters(),
                                   critic.parameters()):
            target_param.data.copy_(target_param.data*(1-tau)+param.data*tau)



def update_params(episode):
    """
    update the parametres of actor and critic
    """
    # sample transitions from the replay buffer and train the critic network
    if episode >= train_critic_from:
        actor_target.eval()
        critic.train()
        critic_target.eval()
        batch = replay_buffer.get_minibatch()
        batch_trans = list(map(list, zip(*batch)))
        states = torch.stack(batch_trans[0], dim=0)
        #shape: [n_batch, n_s]
        actions = batch_trans[1]
        #shape: [n_batch]
        actions = torch.tensor(actions)
        actions = actions.unsqueeze(1)
        #shape: [n_batch, 1]
        #state_actions = torch.cat([states, actions], dim=-1)
        #shape: [n_batch, n_s+1]
        state_actions = [states.float(), actions.float()]
        batch_size = len(actions)
        batch_indices = range(batch_size)
        next_states = torch.stack(batch_trans[2],dim=0)
        rewards = torch.tensor(batch_trans[3])
        is_dones = torch.tensor(batch_trans[4])
        
        #compute next actions according to the actor network
        with torch.no_grad():
            next_actions = actor_target(next_states.float())
            #shape: [n_batch, 1]
        #next_state_actions = torch.cat([next_states, next_actions], dim=-1)
        next_state_actions = [next_states.float(), next_actions.float()]

        # Predict the Q values
        Q_predicted = critic(state_actions)
        #shape: [n_batch, 1]
        Q_predicted = Q_predicted.squeeze(1)
        #shape: [n_batch]

        # get the next Q values
        with torch.no_grad():
            Q_next = critic_target(next_state_actions)
            #shape: [batch_size, 1]
        Q_next = Q_next.squeeze(1)
        Q_next[is_dones] = 0.
        Q_target = gamma*Q_next+rewards

        # train critic network
        loss = loss_critic(Q_predicted, Q_target)
        optimizer_critic.zero_grad()
        loss.backward()
        if if_debug:
            print("Episode: {}".format(episode), file=log_debug)
            print("Gradient Norm of Critic: ", file=log_debug)
            log_debug.flush()
        for param in critic.parameters():
            param.grad.data.clamp_(-1,1)
            if if_debug:
                print(param.grad.norm(), file=log_debug)
                log_debug.flush()
        optimizer_critic.step()

    
    # Train the actor network to maximize the Q-values
    if episode >= train_actor_from:
        actor.train()
        critic.eval()
        actions = actor(states.float())
        state_actions = [states.float(), actions.float()]
        Q = critic(state_actions)
        #shape: [n_batch, 1]
        actor_loss = -Q.mean()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        if if_debug:
            print("Gradient Norm of actor: ", file=log_debug)
        for param in actor.parameters():
            param.grad.data.clamp_(-1,1)
            if if_debug:
                print(param.grad.norm(), file=log_debug)
                log_debug.flush()
        optimizer_actor.step()



def simulate():
    print("Initial episode: {:2d}".format(init_episode),
          "Initial score: {:.4f}".format(init_score))
    episode = init_episode
    max_score = init_score
    scores_deque = deque(maxlen=100)

    if trained_from_scratch:
        print("episode,score,score_100_ave", file=log)
        log.flush()

    while True:
        episode += 1
        noise = ou_noise() #get exploration noise
        # get simulation data
        densities, actions, states = ch.run(compute_target_flow, verbose=False, noise=noise)

        # estimate flow
        flows_episode = [estimate_flow(d) for d in densities]
        rewards = [flow_reward(fl) for fl in flows_episode]

        #ignore the reward before any action and the last action prob
        rewards = rewards[1:]
        actions = actions[:-1]
        next_states = states[1:]
        states = states[:-1]
        is_dones = [False]*len(rewards)
        is_dones[-1] = True

        replay_buffer.add_experience(states, actions, next_states, rewards, is_dones)

        # update parametres of the models.
        update_params(episode)
        # update parametres of the targets.
        update_target(episode)


        # Evaluate the performance of the deterministic policy
        densities, actions, states = ch.run(compute_target_flow, verbose=False, noise=None)
        # estimate flow
        flows_episode = [estimate_flow(d) for d in densities]
        score = np.mean(flows_episode)
        scores_deque.append(score)
        average_score = np.mean(scores_deque)

        print("{},{},{}".format(episode, score, average_score), file=log)
        log.flush()

        if score > max_score:
            max_score = score
            info = {"episode":episode,"score":score}
            with open(info_file, 'w') as f:
                json.dump(info, f)
            
            torch.save(actor, actor_file)
            torch.save(actor_target, actor_target_file)
            torch.save(critic, critic_file)
            torch.save(critic_target, critic_target_file)

        with open(replay_file, 'wb') as f:
            pickle.dump(replay_buffer, f)

        
        if episode % args.verbose_freq == 0:
             print("Episode {:2d} Score: {:.4f}".format(episode, score))

        if episode % args.clear_output == 0:
            os.system("clear")

        if episode >= max_episode:
            break



simulate()
log.close()
if if_debug:
    log_debug.close()

        


    


    







            
            
        

        

    

    


