from .common_interface import CommonInterface


from gymnasium.spaces import Tuple,flatdim
import importlib
#import gymnasium as gym
import numpy as np

class PettingZooWrapper(CommonInterface):
    # Inspired from :https://github.com/uoe-agents/epymarl/blob/main/src/envs/pz_wrapper.py
    def __init__(self,family, env_name,agent_ids=False,**kwargs):
        """ 
        PettingZoo has families of environments (Atari, Butterfly, Classic, MPE, SISL)
        if order to use the pursuit game (sisl family), you usually import it like this:
         >>>>from pettingzoo.sisl import pursuit_v4
         >>>>pursuit_env = pursuit_v4.parallel_env(render_mode="human") 
        """
        env = importlib.import_module(f"pettingzoo.{family}.{env_name}")
        self.env = env.parallel_env(**kwargs)
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.action_space = Tuple(
            tuple([self.env.action_space(agent) for agent in self.agents]))
        self.observation_space = Tuple(
            tuple([self.env.observation_space(agent) for agent in self.agents]))
        self.longest_action_space = max(self.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self.observation_space, key=lambda x: x.shape
        )
        self.agent_ids= agent_ids
    def reset(self, seed=None):
        """ 
        args will be used when the seed is specified 
        """
        obs, _ = self.env.reset(seed = seed)
        obs = self.process_obs(obs)
        self.last_obs = obs ## to avoid empty observations when done (look at step(actions))
        return obs, {}
    
    def render(self, mode="human"):
        return self.env.render(mode)
    
    def step(self, actions):
        
        dict_actions = {agent: actions[index].item() for index,agent in enumerate(self.agents)}
        observations, rewards, dones, truncated, infos = self.env.step(dict_actions)

        obs = self.process_obs(observations) 
        rewards = [rewards[agent] for agent in self.agents]
        done = all([dones[agent] for agent in self.agents])
        truncated = all([truncated[agent] for agent in self.agents])
        info = {
            f"{agent}_{key}": value
            for agent in self.agents
            for key, value in infos[agent].items()
        }
        if done:
            # empty obs and rewards for PZ environments on terminated episode
            assert len(obs) == 0
            assert len(rewards) == 0
            obs = self.last_obs
            rewards = [0] * len(obs)
        else:
            self.last_obs = obs
        return obs, rewards[0], done, truncated, info
    
    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)  +  self.agent_ids * self.n_agents
    def get_action_size(self):
        return self.action_space[0].n
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
        return  [ self.env.action_space(agent).sample() for agent in self.agents]
    def process_obs(self,obs):
        obs = np.array([obs[agent].flatten() for agent in self.agents])
        if self.agent_ids:
            obs = np.concatenate((obs,np.eye(self.n_agents)),axis=1)
        return obs
    def close(self):
        return self.env.close()

