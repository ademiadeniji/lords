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

"""A Driver that takes N steps in the environment using a tf.while_loop."""

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
from tf_agents.trajectories import policy_step


def is_bandit_env(env):
  actual_env = env
  if isinstance(env, tf_py_environment.TFPyEnvironment):
    actual_env = env.pyenv
  is_bandit = (
      isinstance(actual_env, bandit_py_environment.BanditPyEnvironment) or
      isinstance(actual_env, bandit_tf_environment.BanditTFEnvironment))
  return is_bandit


@gin.configurable
class HrlDynamicStepDriver(object):
  """A driver that takes N steps in an environment using a tf.while_loop.

  The while loop will run num_steps in the environment, only counting steps that
  result in an environment transition, i.e. (time_step, action, next_time_step).
  If a step results in environment resetting, i.e. time_step.is_last() and
  next_time_step.is_first() (traj.is_boundary()), this is not counted toward the
  num_steps.

  As environments run batched time_steps, the counters for all batch elements
  are summed, and execution stops when the total exceeds num_steps. When
  batch_size > 1, there is no guarantee that exactly num_steps are taken -- it
  may be more but never less.

  This termination condition can be overridden in subclasses by implementing the
  self._loop_condition_fn() method.
  """

  def __init__(
      self,
      env,
      high_level_policy,
      low_level_policy,
      observers=None,
      transition_observers=None,
      low_level_observers=None,
      num_steps=1,
  ):
    """Creates a HrlDynamicStepDriver.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.Base policy.
      observers: A list of observers that are updated after every step in the
        environment. Each observer is a callable(time_step.Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)).
      num_steps: The number of steps to take in the environment.

    Raises:
      ValueError:
        If env is not a tf_environment.Base or policy is not an instance of
        tf_policy.Base.
    """

    self._num_steps = num_steps
    self._run_fn = common.function_in_tf1()(self._run)
    self._is_bandit_env = is_bandit_env(env)
    self.env = env
    self.high_level_policy = high_level_policy
    self.low_level_policy = low_level_policy
    self._observers = observers or []
    self._transition_observers = transition_observers or []
    self._low_level_observers = low_level_observers or []

  def _loop_condition_fn(self):
    """Returns a function with the condition needed for tf.while_loop."""

    def loop_cond(counter, *_):
      """Determines when to stop the loop, based on step counter.

      Args:
        counter: Step counters per batch index. Shape [batch_size] when
          batch_size > 1, else shape [].

      Returns:
        tf.bool tensor, shape (), indicating whether while loop should continue.
      """
      return tf.less(tf.reduce_sum(input_tensor=counter), self._num_steps)

    return loop_cond

  def _compute_observers(self, z_step, time_step, high_level_time_step, next_time_step, high_level_policy_state, 
    low_level_policy_state, transition_observer_ops, low_level_observer_ops):
      traj = trajectory.from_transition(high_level_time_step, z_step, next_time_step)
      observer_ops = [observer(traj) for observer in self._observers]
      with tf.control_dependencies(
          [tf.group(observer_ops + transition_observer_ops)]):
        time_step, next_time_step, high_level_policy_state, low_level_policy_state = tf.nest.map_structure(
            tf.identity, (time_step, next_time_step, high_level_policy_state, low_level_policy_state))
      return (time_step, next_time_step, high_level_policy_state, low_level_policy_state)

  def _loop_body_fn(self):
    """Returns a function with the driver's loop body ops."""

    def loop_body(counter, skill_counter, z, time_step, high_level_time_step, high_level_policy_state, low_level_policy_state):
      """Runs a step in environment.

      While loop will call multiple times.

      Args:
        counter: Step counters per batch index. Shape [batch_size].
        time_step: TimeStep tuple with elements shape [batch_size, ...].
        policy_state: Policy state tensor shape [batch_size, policy_state_dim].
          Pass empty tuple for non-recurrent policies.

      Returns:
        loop_vars for next iteration of tf.while_loop.
      """
      # reset counter and thus z and high level time step on first time step of horizon
      skill_counter = tf.cond(time_step.is_first(),
        lambda: skill_counter*0, lambda: skill_counter)
        
      # replan z and reset high level step every steps_per_option steps
      new_z = self.high_level_policy.action(time_step, high_level_policy_state).action
      z = tf.cond(skill_counter % self.low_level_policy.steps_per_option == 0, 
        lambda: new_z, lambda: z)
      high_level_time_step = tf.cond(skill_counter % self.low_level_policy.steps_per_option == 0,
        lambda: tf.nest.map_structure(tf.identity, (time_step)), lambda: high_level_time_step)

      z_step = policy_step.PolicyStep(action=z, state=high_level_policy_state, info=())
      
      action_step = self.low_level_policy.action(time_step, z, low_level_policy_state)
      high_level_policy_state = self.high_level_policy.action(time_step, high_level_policy_state).state
      low_level_policy_state = action_step.state
      next_time_step = self.env.step(action_step.action)

      traj = trajectory.from_transition(time_step, z_step, next_time_step)

      if self._is_bandit_env:
        # For Bandits we create episodes of length 1.
        # Since the `next_time_step` is always of type LAST we need to replace
        # the step type of the current `time_step` to FIRST.
        batch_size = tf.shape(input=time_step.discount)
        time_step = time_step._replace(
            step_type=tf.fill(batch_size, ts.StepType.FIRST))

      transition_observer_ops = [
        observer((time_step, z_step, next_time_step))
        for observer in self._transition_observers
      ]
      low_level_observer_ops = [
        observer(traj)
        for observer in self._low_level_observers
      ]

      # every steps_per_option store (high_level_time_step, latent action z, next_time_step) in buffer
      steps_and_states = tf.cond((skill_counter+1) % self.low_level_policy.steps_per_option == 0, 
        lambda: self._compute_observers(z_step, time_step, high_level_time_step, next_time_step, 
        high_level_policy_state, low_level_policy_state, transition_observer_ops, low_level_observer_ops), 
        lambda: (time_step, next_time_step, high_level_policy_state, low_level_policy_state))
      time_step, next_time_step, high_level_policy_state, low_level_policy_state = steps_and_states

      # While loop counter should not be incremented for episode reset steps.
      skill_counter += tf.cast(~traj.is_boundary(), dtype=tf.int32)
      counter += tf.cast(~traj.is_boundary(), dtype=tf.int32)

      return [counter, skill_counter, z, next_time_step, high_level_time_step, high_level_policy_state, low_level_policy_state]

    return loop_body

  def run(self, skill_counter=None, z=None, time_step=None, high_level_time_step=None, high_level_policy_state=None, low_level_policy_state=None, maximum_iterations=None):
    """Takes steps in the environment using the policy while updating observers.

    Args:
      time_step: optional initial time_step. If None, it will use the
        current_time_step of the environment. Elements should be shape
        [batch_size, ...].
      policy_state: optional initial state for the policy.
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
        high_level_policy_state=high_level_policy_state,
        low_level_policy_state=low_level_policy_state,
        maximum_iterations=maximum_iterations)

  # TODO(b/113529538): Add tests for policy_state.
  def _run(self, skill_counter=None, z=None, time_step=None, high_level_time_step=None, high_level_policy_state=None, low_level_policy_state=None, maximum_iterations=None):
    """See `run()` docstring for details."""
    if time_step is None:
      time_step = self.env.current_time_step()
    if high_level_policy_state is None:
      high_level_policy_state = self.high_level_policy.get_initial_state(self.env.batch_size)
    if low_level_policy_state is None:
      low_level_policy_state = self.low_level_policy.get_initial_state(self.env.batch_size)

    # Batch dim should be first index of tensors during data collection.
    batch_dims = nest_utils.get_outer_shape(time_step,
                                            self.env.time_step_spec())
    counter = tf.zeros(batch_dims, tf.int32)
    if skill_counter is None:
      skill_counter = tf.zeros(batch_dims, tf.int32)
    if z is None:
      z = tf.zeros([1], tf.int64)
    if high_level_time_step is None:
      high_level_time_step = tf.nest.map_structure(tf.identity, (time_step))

    [_, skill_counter, z, time_step, high_level_time_step, high_level_policy_state, low_level_policy_state] = tf.while_loop(
        cond=self._loop_condition_fn(),
        body=self._loop_body_fn(),
        loop_vars=[counter, skill_counter, z, time_step, high_level_time_step, high_level_policy_state, low_level_policy_state],
        back_prop=False,
        parallel_iterations=1,
        maximum_iterations=maximum_iterations,
        name='driver_loop')
    return skill_counter, z, time_step, high_level_time_step, high_level_policy_state, low_level_policy_state
