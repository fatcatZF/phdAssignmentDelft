from collections import deque

import random

import struct
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



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

class Actor(nn.Module):
    def __init__(self, n_in=33, n_h=32, n_o=1, init_w=3e-3):
        super(Actor, self).__init__()
        self.mlp = CMLP(n_s=n_in, n_h=n_h, n_o=n_o)
        self.init_weights(init_w)
        
    def init_weights(self, init_w):
        #nn.init.uniform_(self.mlp.fc1.weight, -init_w, init_w)
        #self.mlp.fc1.bias.data.fill_(0.001)
        nn.init.uniform_(self.mlp.fc2.weight, -init_w, init_w)
        self.mlp.fc2.bias.data.fill_(0.001)

    def forward(self, x):
        x = self.mlp(x)
        out = torch.tanh(x)
        return out

class Critic(nn.Module):
    def __init__(self, n_in=34, n_h=32, n_o=1, init_w=3e-3):
        super(Critic, self).__init__()
        self.mlp = CMLP(n_s=n_in, n_h=n_h, n_o=n_o)
        
    def forward(self, xs):
        """
        args:
          xs: a tensor concatenates states and actions
        """
        out = self.mlp(xs)
        return out 


class OUActionNoise:
    """
    Exploration Noise
    copied from:
        https://keras.io/examples/rl/ddpg_pendulum/
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal()
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)




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









