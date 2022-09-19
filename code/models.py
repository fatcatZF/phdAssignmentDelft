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

class DMLPQ2(nn.Module):
    """
    A residual MLP
    """
    def __init__(self, n_s=33, n_h=36, n_a=19):
        super(DMLPQ2, self).__init__()
        self.fc1 = nn.Linear(n_s, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, n_h)
        self.fc_skip = nn.Linear(n_s, n_h)
        self.fc_out = nn.Linear(n_h, n_a)
        self.bn1 = nn.BatchNorm1d(n_h)
        self.bn2 = nn.BatchNorm1d(n_h)

    def forward(self, x):
        """
        args:
          x: input states, shape: [n_batch, n_s]
        """
        x_skip = self.fc_skip(x)
        x1 = self.bn1(F.relu(self.fc1(x)))
        x2 = self.bn2(F.relu(self.fc2(x1)))
        x3 = x_skip + F.relu(self.fc3(x2))
        out = self.fc_out(x3)

        return out



class TCNQ(nn.Module):
    """
    A TCN mapping the state (history of densities and previous target flow)
    to Q values
    """
    def __init__(self, n_s=33, n_h=36, n_a=19, history=4):
        super(TCNQ, self).__init__()
        self.conv1 = nn.Conv1d(n_s, n_h, history)
        self.conv2 = nn.Conv1d(n_s, n_h, history)
        self.conv_skip = nn.Conv1d(n_s, n_a, history)
        self.lin = nn.Linear(n_h, n_a)

    def forward(self, x):
        """
        args:
          x: input state, shape: [n_batch, n_history, n_s]
        """
        x = x.permute(0,2,1) #change the shape to [n_batch, n_s, n_history]
        x1 = torch.sigmoid(self.conv1(x)).squeeze(-1)
        x2 = torch.tanh(self.conv2(x)).squeeze(-1)
        y = self.lin(x1*x2)
        x_skip = (self.conv_skip(x)).squeeze(-1)
        
        
        return y+x_skip

class TCNQ2(nn.Module):
    def __init__(self, n_s=33, n_h=36, n_a=19, history=4):
        super(TCNQ2, self).__init__()
        self.conv1_sin = nn.Conv1d(n_s, n_h, history, padding=history-1)
        self.conv1_tanh = nn.Conv1d(n_s, n_h, history, padding=history-1)
        self.conv2_sin = nn.Conv1d(n_h, n_h, history)
        self.conv2_tanh = nn.Conv1d(n_h, n_h, history)
        self.conv_skip = nn.Conv1d(n_s, n_h, history)
        self.bn1 = nn.BatchNorm1d(n_h)
        self.bn2 = nn.BatchNorm1d(n_h)
        self.fc_out = nn.Linear(n_h, n_a)
        self.padding = history-1
        
    def forward(self, x):
        """
        args:
          x: input states, shape: [n_batch, n_history, n_s]
        """
        x = x.permute(0,2,1)
        x_skip = self.conv_skip(x)
        x1_sin = torch.sigmoid(self.conv1_sin(x)[:,:,:-self.padding])
        x1_tanh = torch.tanh(self.conv1_tanh(x)[:,:,:-self.padding])
        x2 = self.bn1(x1_sin*x1_tanh)
        x2_sin = torch.sigmoid(self.conv2_sin(x2))
        x2_tanh = torch.tanh(self.conv2_tanh(x2))
        x3 = self.bn2(x2_sin*x2_tanh)
        x3 = (x3+x_skip).squeeze(-1)
        
        out = self.fc_out(x3)
        
        return out



class Actor(nn.Module):
    def __init__(self, n_in=33, n_h=36, n_o=1, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc_in = nn.Linear(n_in, n_h)
        self.fc_h1 = nn.Linear(n_h, n_h)
        self.fc_skip = nn.Linear(n_in, n_h)
        self.fc_out = nn.Linear(n_h, n_o)
        self.bn1 = nn.BatchNorm1d(n_h)
        self.bn2 = nn.BatchNorm1d(n_h)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        nn.init.uniform_(self.fc_in.weight, -init_w, init_w)
        nn.init.uniform_(self.fc_h1.weight, -init_w, init_w)
        nn.init.uniform_(self.fc_skip.weight, -init_w, init_w)
        nn.init.uniform_(self.fc_out.weight, -init_w, init_w)

    def forward(self, x):
        """
        args:
           x: input states, shape: [n_batch, n_in]
        """
        x_skip = self.fc_skip(x)
        x1 = self.bn1(F.leaky_relu(self.fc_in(x)))
        x2 = self.bn2(F.leaky_relu(self.fc_h1(x1)))
        x3 = x_skip + x2
        out = torch.tanh(self.fc_out(x3))
        return out

class Critic(nn.Module):
    def __init__(self, n_s=33, n_a=1, n_h=36,
                 n_o=1, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc_in = nn.Linear(n_s+n_a, n_h)
        self.fc_h1 = nn.Linear(n_h, n_h)
        self.fc_skip = nn.Linear(n_s+n_a, n_h)
        self.fc_o = nn.Linear(n_h, n_o)
        self.bn1 = nn.BatchNorm1d(n_h)
        self.bn2 = nn.BatchNorm1d(n_h)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        nn.init.uniform_(self.fc_in.weight, -init_w, init_w)
        nn.init.uniform_(self.fc_h1.weight, -init_w, init_w)
        nn.init.uniform_(self.fc_skip.weight, -init_w, init_w)
        nn.init.uniform_(self.fc_o.weight, -init_w, init_w)

    def forward(self, xa):
        """
        args:
          xa: a list of states and actions
        """
        x = torch.cat(xa, dim=-1)
        #shape: [n_batch, n_s+n_a]
        x_skip = self.fc_skip(x)
        x1 = self.bn1(F.leaky_relu(self.fc_in(x)))
        x2 = self.bn2(F.leaky_relu(self.fc_h1(x1)))
        x3 = x_skip+x2
        out = self.fc_o(x3)
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









