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
import numpy as np

from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.environments import bandit_tf_environment
from tf_agents.drivers import driver
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils


def is_bandit_env(env):
  actual_env = env
  if isinstance(env, tf_py_environment.TFPyEnvironment):
    actual_env = env.pyenv
  is_bandit = (
      isinstance(actual_env, bandit_py_environment.BanditPyEnvironment) or
      isinstance(actual_env, bandit_tf_environment.BanditTFEnvironment))
  return is_bandit


@gin.configurable
class SkillDynamicEpisodeDriver(driver.Driver):
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
               skill_noise_epsilon=False):
    """Creates a SkillDynamicEpisodeDriver.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.Base policy.
      observers: A list of observers that are updated after every step in the
        environment. Each observer is a callable(Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)).
      num_episodes: The number of episodes to take in the environment.

    Raises:
      ValueError:
        If env is not a tf_environment.Base or policy is not an instance of
        tf_policy.Base.
    """
    super(SkillDynamicEpisodeDriver, self).__init__(env, policy, observers,
                                               transition_observers)
    self._num_episodes = num_episodes
    self._run_fn = common.function_in_tf1()(self._run)
    self._is_bandit_env = is_bandit_env(env)
    self._num_skills = num_skills
    self._skill_noise_epsilon = skill_noise_epsilon
    self._sampled_skill = None
    self._mask = None

  def _loop_condition_fn(self, num_episodes):
    """Returns a function with the condition needed for tf.while_loop."""

    def loop_cond(counter, *_):
      """Determines when to stop the loop, based on episode counter.

      Args:
        counter: Episode counters per batch index. Shape [batch_size] when
          batch_size > 1, else shape [].

      Returns:
        tf.bool tensor, shape (), indicating whether while loop should continue.
      """
      return tf.less(tf.reduce_sum(input_tensor=counter), num_episodes)

    return loop_cond

  def _sample_skill(self, skill):
    return tf.cond(tf.random.uniform([1]) < tf.constant(self._skill_noise_epsilon), \
      lambda: tf.squeeze(tf.random.categorical(logits=tf.math.log([[1.0]*self._num_skills]), \
      num_samples=1, dtype=tf.int32), 1), lambda: skill)

  def _loop_body_fn(self):
    """Returns a function with the driver's loop body ops."""

    def loop_body(counter, step_counter, time_step, policy_state, skill):
      """Runs a step in environment.

      While loop will call multiple times.

      Args:
        counter: Episode counters per batch index. Shape [batch_size].
        time_step: TimeStep tuple with elements shape [batch_size, ...].
        policy_state: Poicy state tensor shape [batch_size, policy_state_dim].
          Pass empty tuple for non-recurrent policies.

      Returns:
        loop_vars for next iteration of tf.while_loop.
      """
      # potentially take action with random skill if using skill noise
      if self._skill_noise_epsilon is not None:
        self._sampled_skill = tf.cond((step_counter % 20) == tf.constant(0), \
        lambda: self._sample_skill(skill), lambda: self._sampled_skill)
        time_step.observation["z"] = tf.cast(self._sampled_skill, tf.int64)
      else:
        time_step.observation["z"] = tf.cast(skill, tf.int64)
      
      action_step = self.policy.action(time_step, policy_state)

      # TODO(b/134487572): TF2 while_loop seems to either ignore
      # parallel_iterations or doesn't properly propagate control dependencies
      # from one step to the next. Without this dep, self.env.step() is called
      # in parallel.
      with tf.control_dependencies(tf.nest.flatten([time_step])):
        next_time_step = self.env.step(action_step.action)
      
      # relabel to original skill for entry into buffer
      time_step.observation["z"] = tf.cast(skill, tf.int64)
      next_time_step.observation["z"] = tf.cast(skill, tf.int64)

      policy_state = action_step.state

      if self._is_bandit_env:
        # For Bandits we create episodes of length 1.
        # Since the `next_time_step` is always of type LAST we need to replace
        # the step type of the current `time_step` to FIRST.
        batch_size = tf.shape(input=time_step.discount)
        time_step = time_step._replace(
            step_type=tf.fill(batch_size, ts.StepType.FIRST))

      if self._mask is not None:
        obs_size = time_step.observation["observation"].shape[1]
        if self._mask == "velocity":
          mask = tf.one_hot([15], obs_size, on_value=0, off_value=1) * \
            tf.one_hot([16], obs_size, on_value=0, off_value=1) * \
            tf.one_hot([17], obs_size, on_value=0, off_value=1)
        if self._mask == "all":
          mask = tf.one_hot([0], obs_size, on_value=1, off_value=0) + \
            tf.one_hot([1], obs_size, on_value=1, off_value=0)   
        mask = tf.cast(mask, dtype=tf.float64)
        time_step_masked = tf.nest.map_structure(tf.identity, time_step)
        next_time_step_masked = tf.nest.map_structure(tf.identity, next_time_step)
        time_step_masked.observation["observation"] *= mask
        next_time_step_masked.observation["observation"] *= mask
        traj = trajectory.from_transition(time_step_masked, action_step, next_time_step_masked)
      else:
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

      observer_ops = [observer(traj) for observer in self._observers]
      transition_observer_ops = [
          observer((time_step, action_step, next_time_step))
          for observer in self._transition_observers
      ]
      with tf.control_dependencies(
          [tf.group(observer_ops + transition_observer_ops)]):
        time_step, next_time_step, policy_state = tf.nest.map_structure(
            tf.identity, (time_step, next_time_step, policy_state))

      # While loop counter is only incremented for episode reset episodes.
      # For Bandits, this is every trajectory, for MDPs, this is at boundaries.
      if self._is_bandit_env:
        counter += tf.ones(batch_size, dtype=tf.int32)
      else:
        counter += tf.cast(traj.is_boundary(), dtype=tf.int32)
      step_counter += tf.cast(~traj.is_boundary(), dtype=tf.int32)

      return [counter, step_counter, next_time_step, policy_state, skill]

    return loop_body

  def run(self,
          time_step=None,
          policy_state=None,
          num_episodes=None,
          maximum_iterations=None):
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
    return self._run_fn(
        time_step=time_step,
        policy_state=policy_state,
        num_episodes=num_episodes,
        maximum_iterations=maximum_iterations)

  def _run(self,
           time_step=None,
           policy_state=None,
           num_episodes=None,
           maximum_iterations=None,
           skill=None):
    """See `run()` docstring for details."""
    if time_step is None:
      time_step = self.env.reset()
    if skill is not None:
      time_step.observation["z"] = tf.cast(skill, tf.int64)
      self._sampled_skill = tf.identity(skill)
    if policy_state is None:
      policy_state = self.policy.get_initial_state(self.env.batch_size)

    # Batch dim should be first index of tensors during data
    # collection.
    batch_dims = nest_utils.get_outer_shape(time_step,
                                            self.env.time_step_spec())
    counter = tf.zeros(batch_dims, tf.int32)
    step_counter = tf.zeros(batch_dims, tf.int32)

    num_episodes = num_episodes or self._num_episodes
    [_, _, time_step, policy_state, skill] = tf.while_loop(
        cond=self._loop_condition_fn(num_episodes),
        body=self._loop_body_fn(),
        loop_vars=[counter, step_counter, time_step, policy_state, skill],
        back_prop=False,
        parallel_iterations=1,
        maximum_iterations=maximum_iterations,
        name='driver_loop')
    return time_step, policy_state
