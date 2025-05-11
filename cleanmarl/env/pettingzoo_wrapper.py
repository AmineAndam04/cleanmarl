from .common_interface import CommonInterface


from gymnasium.spaces import Tuple,flatdim
import importlib
#import gymnasium as gym
import numpy as np

class PettingZooWrapper(CommonInterface):
    # Inspired from :https://github.com/uoe-agents/epymarl/blob/main/src/envs/pz_wrapper.py
    def __init__(self,family, env_name,**kwargs):
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
        
        self.longest_observation_space = max(
            self.observation_space, key=lambda x: x.shape
        )
    def reset(self, *args):
        """ 
        args will be used when the seed is specified 
        """
        obs, info = self.env.reset(*args)
        obs = np.array([obs[agent].flatten() for agent in self.env.agents])
        self.last_obs = obs ## to avoid empty observations when done (look at step(actions))
        return obs, {}
    
    def render(self, mode="human"):
        return self.env.render(mode)
    
    def step(self, actions):
        
        dict_actions = {agent: actions[index].item() for index,agent in enumerate(self.agents)}
        observations, rewards, dones, truncated, infos = self.env.step(dict_actions)

        obs = np.array([observations[agent].flatten() for agent in self.agents])
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
        return flatdim(self.longest_observation_space)
    def get_action_size(self):
        return self.action_space[0].n
    def sample(self):
        return  [ self.env.action_space(agent).sample() for agent in self.agents]
    def close(self):
        return self._env.close()

