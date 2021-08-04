"""Gym wrapper for training the DIAYN agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import gym


class DiaynGymEnvFixed(gym.Env):
    """Gym wrapper for training the DIAYN agent.

    Augment the observation by adding a randomly sampled z.
    """

    def __init__(self, environment, dim_z, categorical):
        """Initializes the environment.
        """
        self._env = environment

        if categorical:
            self._z_space = gym.spaces.Discrete(n=dim_z)
        else:
            self._z_space = gym.spaces.Box(low=-1, high=1., shape=[dim_z])
        self._z = 0
        self._categorical = categorical

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'observation': self._env.observation_space,
            'z': self._z_space,
        })

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def z_space(self):
        return self._z_space

    @property
    def reward_range(self):
        return self._env.reward_range

    @property
    def meta_data(self):
        return self._env.meta_data

    def reset(self):
        observation = self._env.reset()
        return self._augment_ob(observation)
    
    def set_z(self, z):
        self._z = z
        return

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        return self._augment_ob(observation), reward, done, info

    def _augment_ob(self, ob, seed=None):        
        return {
            'observation': ob,
            'z': self._z,
        }
    
    def render(self, mode='rgb_array'):
        return self._env.render(mode)
