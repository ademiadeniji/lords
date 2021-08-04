# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Driver that takes N episodes in the environment using a tf.while_loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import pdb

from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.environments import bandit_tf_environment
from tf_agents.drivers import driver
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from . import skill_dynamic_episode_driver


def is_bandit_env(env):
  actual_env = env
  if isinstance(env, tf_py_environment.TFPyEnvironment):
    actual_env = env.pyenv
  is_bandit = (
      isinstance(actual_env, bandit_py_environment.BanditPyEnvironment) or
      isinstance(actual_env, bandit_tf_environment.BanditTFEnvironment))
  return is_bandit


@gin.configurable
class DynamicDemonstrationDriver(skill_dynamic_episode_driver.SkillDynamicEpisodeDriver):
  """A driver that takes N episodes in an environment using a tf.while_loop.

  The while loop will run num_episodes in the environment, counting transitions
  that result in ending an episode.

  As environments run batched time_episodes, the counters for all batch elements
  are summed, and execution stops when the total exceeds num_episodes.

  This termination condition can be overridden in subclasses by implementing the
  self._loop_condition_fn() method.
  """

  def __init__(self,
               env,
               policy,
               observers=None,
               transition_observers=None,
               num_episodes=1, 
               num_skills=5,
               skill_noise_epsilon=None):
    """Creates a DynamicDemonstrationDriver.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.Base policy.
      observers: A list of observers that are updated after every step in the
        environment. Each observer is a callable(Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)).
      num_episodes: The number of episodes to take in the environment.
      num_skills: The number of skills to collect demonstrations for.

    Raises:
      ValueError:
        If env is not a tf_environment.Base or policy is not an instance of
        tf_policy.Base.
    """
    super(DynamicDemonstrationDriver, self).__init__(env, policy, observers,
                                               transition_observers, num_episodes, 
                                               num_skills, skill_noise_epsilon)
    self._num_skills = num_skills
 
  def _demonstration_loop_condition_fn(self, num_skills):
      
      def loop_cond(skill_counter, *_):
 
          return tf.less(tf.reduce_sum(skill_counter), num_skills)

      return loop_cond 

  def _demonstration_loop_body_fn(self):

      def loop_body(skill_counter):

          self._run_fn(skill=skill_counter)
          skill_counter += tf.ones(1, dtype=tf.int32)
          return skill_counter

      return loop_body

  def run(self, maximum_iterations=None):
    """Takes episodes in the environment using the policy and update observers.

    If `time_step` and `policy_state` are not provided, `run` will reset the
    environment and request an initial state from the policy.

    Args:
      time_step: optional initial time_step. If None, it will be obtained by
        resetting the environment. Elements should be shape [batch_size, ...].
      policy_state: optional initial state for the policy. If None, it will be
        obtained from the policy.get_initial_state().
      num_episodes: Optional number of episodes to take in the environment. If
        None it would use initial num_episodes.
      maximum_iterations: Optional maximum number of iterations of the while
        loop to run. If provided, the cond output is AND-ed with an additional
        condition ensuring the number of iterations executed is no greater than
        maximum_iterations.

    Returns:
      time_step: TimeStep named tuple with final observation, reward, etc.
      policy_state: Tensor with final step policy state.
    """
    skill_counter = tf.zeros(1, dtype=tf.int32, name="skill_counter")
    skill_counter = tf.while_loop(
        cond=self._demonstration_loop_condition_fn(self._num_skills),
        body=self._demonstration_loop_body_fn(),
        loop_vars=[skill_counter],
        parallel_iterations=1, 
        maximum_iterations=maximum_iterations,
        name='skill_loop')
    return

