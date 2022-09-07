from collections import deque

import random

import struct
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F



def estimate_flow(d):
    """
    estimate flow based on density according to Greenshields Theory
    """
    d_c = 35 #critical density
    v_max = 30 #maximal speed
    f_max = d_c*v_max #estimated maximal flow
    s = max(0, d*(2*d_c-d))
    return (f_max/d_c**2)*s

def flow_reward(fl, c=1.5):
    """
    reward function based on estimated flow
    """
    d_c = 35 #critical density
    v_max = 30 #maximal speed
    f_max = d_c*v_max #estimated maximal flow
    return 1-c*(abs(f_max-fl)/f_max)
    




def speedMulCount_reward(smc, c=0.001):
    """
    reward function based on speed X vehcle_number
    compute the reward obtained in one step
    """
    return c*smc


def binary(num):
    """convert a number to 32bit binary representation"""
    s = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
    return [int(a) for a in s]



class DMLP(nn.Module):
    """
    A MLP mapping the 
    state (density of the main road and previous target flow)
    to discrete action spaces (0 to 18)
    """
    def __init__(self, n_s=33, n_h=36, n_a=19):
        """
        args:
          n_s: dimension of states
          n_h: dimension of hidden 
          n_a: dimension of action spaces
        """
        super(DMLP, self).__init__()
        self.fc1 = nn.Linear(n_s, n_h)
        self.fc2 = nn.Linear(n_h, n_a)

    def forward(self, x):
        """
        args:
            x: state, shape: [n_batch=1, n_s]
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1) #shape [n_batch=1, n_a]


class DMLPQ(nn.Module):
    """
    A MLP mapping the 
    state (density of the main road and previous target flow)
    to Q values
    """
    def __init__(self, n_s=33, n_h=36, n_a=19):
        """
        args:
          n_s: dimension of states
          n_h: dimension of hidden 
          n_a: dimension of action spaces
        """
        super(DMLPQ, self).__init__()
        self.fc1 = nn.Linear(n_s, n_h)
        self.fc2 = nn.Linear(n_h, n_a)

    def forward(self, x):
        """
        args:
            x: state, shape: [n_batch=1, n_s]
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 


class CMLP(nn.Module):
    """
    A MLP mapping the state 
    or (state, action) to values, which 
    can be actions or Q-values
    """
    def __init__(self, n_s=33, n_h=32, n_o=1):
        super(CMLP, self).__init__()
        self.fc1 = nn.Linear(n_s, n_h)
        self.fc2 = nn.Linear(n_h, n_o)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayBuffer():
    """Replay Buffer stores the last N transitions
       (for deep Q-learning)
    """
    def __init__(self, max_size=30000, batch_size=64):
        """
        args:
            max_size: the maximal number of stored transitions
            batch_size: the number of transitions returned 
                in a minibatch
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self.states = deque([], maxlen=max_size)
        self.actions = deque([], maxlen=max_size)
        self.next_states = deque([], maxlen=max_size)
        self.rewards = deque([], maxlen=max_size)
        self.is_dones = deque([], maxlen=max_size)
        self.indices = [None]*batch_size
        
    def add_experience(self, states, actions, next_states, rewards, is_dones):
        self.states.extend(states)
        self.actions.extend(actions)
        self.next_states.extend(next_states)
        self.rewards.extend(rewards)
        self.is_dones.extend(is_dones)
        
    
    def get_valid_indices(self):
        experience_size = len(self.states)
        for i in range(self.batch_size):
            index = random.randint(0, experience_size-1)
            self.indices[i] = index
            
    def get_minibatch(self):
        """
        Return a minibatch
        """
        batch = []
        self.get_valid_indices()
        
        for idx in self.indices:
            state = self.states[idx]
            action = self.actions[idx]
            next_state = self.next_states[idx]
            reward = self.rewards[idx]
            is_done = self.is_dones[idx]
            
            batch.append((state, action, next_state, reward, is_done))
            
        return batch









