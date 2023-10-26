import torch
import torch.nn as nn
import gym
import d4rl
import numpy as np
import functools
import copy
import os
import torch.nn.functional as F
import tqdm
from scipy.special import softmax
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)

class D4RL_dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args=args
        data = d4rl.qlearning_dataset(gym.make(args.env))
        self.device = args.device
        self.states = torch.from_numpy(data['observations']).float().to(self.device)
        self.actions = torch.from_numpy(data['actions']).float().to(self.device)
        self.next_states = torch.from_numpy(data['next_observations']).float().to(self.device)
        reward = torch.from_numpy(data['rewards']).reshape(-1, 1).float().to(self.device)
        self.is_finished = torch.from_numpy(data['terminals']).reshape(-1, 1).float().to(self.device)

        reward_tune = "iql_antmaze" if "antmaze" in args.env else "iql_locomotion"
        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            min_ret, max_ret = return_range(data, 1000)
            reward /= (max_ret - min_ret)
            reward *= 1000
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.rewards = reward
        print("dql dataloard loaded")
        
        self.len = self.states.shape[0]
        print(self.len, "data loaded")
        self.current_idx = 0

    def __getitem__(self, index):
        use_index = index % self.len
        data = {'s': self.states[use_index],
                'a': self.actions[use_index],
                'r': self.rewards[use_index],
                's_':self.next_states[use_index],
                'd': self.is_finished[use_index],
            }
        return data

    def _shuffle_data(self):
        indices = torch.randperm(self.len).to("cuda")
        self.states = self.states[indices]
        self.next_states = self.next_states[indices]
        self.actions = self.actions[indices]
        self.rewards = self.rewards[indices]
        self.is_finished = self.is_finished[indices]

    def sample(self, batch_size):
        if self.current_idx+batch_size > self.len:
            self.current_idx = 0
        if self.current_idx == 0:
            self._shuffle_data()
        data = {'s': self.states[self.current_idx:self.current_idx+batch_size],
                'a': self.actions[self.current_idx:self.current_idx+batch_size],
                'r': self.rewards[self.current_idx:self.current_idx+batch_size],
                's_':self.next_states[self.current_idx:self.current_idx+batch_size],
                'd': self.is_finished[self.current_idx:self.current_idx+batch_size],
            }
        self.current_idx = self.current_idx + batch_size
        return data

    def __add__(self, other):
        pass

    def __len__(self):
        return self.len
