import importlib

import numpy as np
from gymnasium.spaces import Box, flatdim

from .common_interface import CommonInterface


class PettingZooWrapper(CommonInterface):
    # Inspired from :https://github.com/uoe-agents/epymarl/blob/main/src/envs/pz_wrapper.py
    def __init__(self, family, env_name, agent_ids=False, **kwargs):
        """
        PettingZoo has families of environments (Atari, Butterfly, Classic, MPE, SISL)
        in order to use the pursuit game (sisl family), you usually import it like this:
         >>>>from pettingzoo.sisl import pursuit_v4
         >>>>pursuit_env = pursuit_v4.parallel_env(render_mode="human")
        """
        env = importlib.import_module(f"pettingzoo.{family}.{env_name}")
        self.env = env.parallel_env(**kwargs)
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.act_dim = flatdim(self.env.action_space(self.agents[0]))
        if isinstance(self.env.action_space(self.agents[0]), Box):
            self.act_low = self.env.action_space(self.agents[0]).low
            self.act_high = self.env.action_space(self.agents[0]).high
        self.obs_dim = flatdim(self.env.observation_space(self.agents[0]))
        self.agent_ids = agent_ids

    def reset(self, seed=None):
        """
        args will be used when the seed is specified
        """
        obs, _ = self.env.reset(seed=seed)
        obs = self.process_obs(obs)
        self.last_obs = obs  ## to avoid empty observations when done (look at step(actions))
        return obs, {}

    def render(self, mode="human"):
        return self.env.render(mode)

    def step(self, actions):
        dict_actions = {agent: actions[index] for index, agent in enumerate(self.agents)}
        observations, rewards, dones, truncated, infos = self.env.step(dict_actions)

        obs = self.process_obs(observations)
        rewards = [rewards[agent] for agent in self.agents]
        done = all([dones[agent] for agent in self.agents])
        truncated = all([truncated[agent] for agent in self.agents])
        info = {
            f"{agent}_{key}": value for agent in self.agents for key, value in infos[agent].items()
        }
        if done:
            # empty obs and rewards for PZ environments on terminated episode
            if len(obs) == 0 and len(rewards) == 0:
                obs = self.last_obs
                rewards = [0] * len(obs)
        else:
            self.last_obs = obs
        return obs, rewards[0], done, truncated, info

    def get_obs_size(self):
        return self.obs_dim + self.agent_ids * self.n_agents

    def get_state_size(self):
        return self.obs_dim * self.n_agents

    def get_state(self):
        return self.state

    def get_action_size(self):
        return self.act_dim

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.act_dim))

    def sample(self):
        return [self.env.action_space(agent).sample() for agent in self.agents]

    def process_obs(self, obs):
        obs = np.array([obs[agent].flatten() for agent in self.agents])
        self.state = obs.reshape(-1)
        if self.agent_ids:
            obs = np.concatenate((obs, np.eye(self.n_agents)), axis=1)
        return obs

    def close(self):
        return self.env.close()
