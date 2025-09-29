import numpy as np
from .common_interface import CommonInterface

import lbforaging
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Tuple,flatdim

class LBFWrapper(CommonInterface):
    def __init__(self,map_name, reward_aggr='sum',seed=0, time_limit=150, agent_ids=False,**kwargs):
        super().__init__()
        self.env = gym.make(map_name,max_episode_steps=time_limit, **kwargs)
        self.env = TimeLimit(self.env, max_episode_steps=time_limit)
        self.agent_ids = agent_ids
        self.n_agents = self.env.unwrapped.n_agents
        self.agents = list(range(self.n_agents))
        self.episode_limit = time_limit
        self.reward_aggr = reward_aggr
        self.action_space = Tuple(
            tuple([self.env.action_space[agent] for agent in self.agents]))
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self.env.observation_space, key=lambda x: x.shape)
    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(act) for act in actions]
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.current_step += 1
        obs = self.process_obs(obs)
        if self.reward_aggr == "sum":
            reward = np.sum(reward)
        elif self.reward_aggr == "mean":
            reward = np.mean(reward)

        if terminated and self.current_step == self.env.unwrapped._max_episode_steps:
            truncated = True
        return obs, np.array(reward), terminated, truncated, info
    def reset(self, seed=None):
        """ 
        args will be used when the seed is specified 
        """
        self.current_step = 0
        obs, _ = self.env.reset(seed = seed)
        obs = self.process_obs(obs)
        return obs, {}
    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)  +  self.agent_ids * self.n_agents
    def get_state_size(self):
        """Returns the size of the state (needed for QMIX)"""
        return flatdim(self.longest_observation_space) * self.n_agents
    def get_state(self):
        """Returns the global state (needed for QMIX)"""
        return self.state
    def get_action_size(self):
        return self.longest_action_space.n
    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.array(avail_actions)

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid
    def sample(self):
        return list(self.env.action_space.sample())
    def process_obs(self,obs):
        obs = np.array(obs)
        self.state = obs.reshape(-1)
        if self.agent_ids:
            obs = np.concatenate((obs,np.eye(self.n_agents)),axis=1)
        return obs
    def close(self):
        return self.env.close()