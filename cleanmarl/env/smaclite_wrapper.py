import torch
import numpy as np
from torch.distributions import Categorical

from .common_interface import CommonInterface

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import smaclite  

class SMACliteWrapper(CommonInterface):
    def __init__(self,map_name, seed=0, time_limit=150, agent_ids=False,**kwargs):
        self.env = gym.make(f"smaclite/{map_name}-v0", seed=seed, **kwargs)
        self.env = TimeLimit(self.env, max_episode_steps=time_limit)
        self.agent_ids = agent_ids
        self.n_agents = self.env.unwrapped.n_agents
        self.episode_limit = time_limit
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)
        

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(act) for act in actions]
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self.process_obs(obs)
        return obs, reward, terminated, truncated, info
    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.unwrapped.obs_size + self.agent_ids * self.n_agents
    def get_state_size(self):
        """Returns the size of the state (needed for QMIX)"""
        return self.env.unwrapped.state_size
    def get_state(self):
        """Returns the global state (needed for QMIX)"""
        return self.env.unwrapped.get_state()
    def get_action_size(self):
        """Returns the total number of actions an agent could ever take"""
        return flatdim(self.longest_action_space)
    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs,_ = self.env.reset(seed=seed, options=options)
        obs = self.process_obs(obs)
        return obs, {}
    def get_avail_actions(self):
        return np.array(self.env.unwrapped.get_avail_actions())
    def get_agents(self):
        return self.env.unwrapped.agents
    def sample(self):
        avail_actions = torch.tensor(self.get_avail_actions(),dtype=torch.float32)
        masked_probs = avail_actions / avail_actions.sum(dim=1, keepdim=True) # Normalize avail_actions to turn them into probabilities
        dist = Categorical(masked_probs)
        actions = dist.sample()  
        return actions
    def process_obs(self,obs):
        obs = np.array(obs)
        if self.agent_ids:
            obs = np.concatenate((obs,np.eye(self.n_agents)),axis=1)
        return obs
    def close(self):
        self.env.close()