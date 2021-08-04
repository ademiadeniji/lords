# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import gym
from gym import Wrapper
import pdb

class SkillWrapper(Wrapper):

  def __init__(
      self,
      env,
      # skill type and dimension
      num_latent_skills=None,
      skill_type='discrete_uniform',):

    super(SkillWrapper, self).__init__(env)
    self._skill_type = skill_type
    if num_latent_skills is None:
      self._num_skills = 0
    else:
      self._num_skills = num_latent_skills
    self.skill = np.zeros(self._num_skills)
    self.skill[0] = 1

    if isinstance(self.env.observation_space, gym.spaces.Dict):
      size = self.env.observation_space.spaces['observation'].shape[0] + self._num_skills
    else:
      size = self.env.observation_space.shape[0] + self._num_skills
    self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

  def _remake_time_step(self, cur_obs):
    if isinstance(self.env.observation_space, gym.spaces.Dict):
      cur_obs = cur_obs['observation']
    elif not isinstance(cur_obs, np.ndarray):
      cur_obs = cur_obs.observation
    if self._num_skills == 0:
      return cur_obs
    else:
      return np.concatenate([cur_obs, self.skill])

  def set_skill(self, skill_num):
    self.skill = np.zeros(self._num_skills)
    self.skill[skill_num] = 1

  def reset(self):
    cur_obs = self.env.reset()
    self._step_count = 0
    return self._remake_time_step(cur_obs)

  def step(self, action):
    cur_obs, reward, done, info = self.env.step(action)
    self._step_count += 1
    return self._remake_time_step(cur_obs), reward, done, info

  def close(self):
    return self.env.close()
