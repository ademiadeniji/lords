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

"""Actor Policy based on an actor network.

This is used in e.g. actor-critic algorithms like DDPG.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
import pdb

@gin.configurable
class LatentPolicy(tf_policy.Base):
  """Class to build Actor Policies."""

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               num_skills,
               info_spec=(),
               embedding_network=None,
               observation_mask=None,
               handcraft_obs_policy=False,
               observation_normalizer=None,
               clip=True,
               training=False,
               name=None):
    """Builds an Actor Policy given a actor network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_network: An instance of a `tf_agents.networks.network.Network` to be
        used by the policy. The network will be called with call(observation,
        step_type, policy_state) and should return (actions_or_distributions,
        new_state).
      info_spec: A nest of TensorSpec representing the policy info.
      observation_normalizer: An object to use for observation normalization.
      clip: Whether to clip actions to spec before returning them.  Default
        True. Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.
      training: Whether the network should be called in training mode.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      ValueError: if actor_network is not of type network.Network.
    """
    if not isinstance(actor_network, network.Network):
      raise ValueError('actor_network must be a network.Network. Found '
                       '{}.'.format(type(actor_network)))
    actor_network.create_variables()
    self._actor_network = actor_network
    self._embedding_network = embedding_network
    self._observation_mask = observation_mask
    self._num_skills = num_skills
    self._handcraft_obs_policy = handcraft_obs_policy
    self._observation_normalizer = observation_normalizer
    self._training = training

    super(LatentPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=actor_network.state_spec,
        info_spec=info_spec,
        clip=clip,
        name=name)

  def _apply_actor_network(self, observation, policy_state, step_type):
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)
    return self._actor_network(
        observation, step_type, policy_state, training=self._training)

  @property
  def observation_normalizer(self):
    return self._observation_normalizer

  def _variables(self):
    return self._actor_network.variables

  def _distribution(self, time_step, policy_state):
    # Actor network outputs nested structure of distributions or actions.
    if self._embedding_network is not None:
        step_type = ts.StepType.MID
        skill = time_step.observation[:, -self._num_skills:]
        observation = time_step.observation[:, :-self._num_skills]
        embedded_obs_distribution, _ = self._embedding_network(
            observation, step_type, ())
        embedded_obs = tf.nest.map_structure(lambda d: d.sample(), 
            embedded_obs_distribution)
        embedded_input = tf.concat((embedded_obs, skill), axis=-1)
        actions_or_distributions, policy_state = self._apply_actor_network(
            embedded_input, policy_state, time_step.step_type)
    elif self._observation_mask is not None:
        def log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        skill = time_step.observation[:, -self._num_skills:]
        observation = time_step.observation[:, :-self._num_skills]
        sigmask = tf.math.sigmoid(self._observation_mask) 
        max_val = tf.math.reduce_max(sigmask)
        cutoff = tf.math.floor(log10(max_val))
        cutoff = 10 ** cutoff
        binary_mask = sigmask > cutoff
        masked_obs = observation * tf.cast(binary_mask, tf.float32)
        masked_input = tf.concat((masked_obs, skill), axis=-1)
        actions_or_distributions, policy_state = self._apply_actor_network(
            masked_input, policy_state, time_step.step_type)
    elif self._handcraft_obs_policy:
        skill = time_step.observation[:, -self._num_skills:]
        observation = time_step.observation[:, :-self._num_skills]
        handcrafted_obs = observation[:, 3:5]
        handcrafted_input = tf.concat((handcrafted_obs, skill), axis=-1)
        actions_or_distributions, policy_state = self._apply_actor_network(
            handcrafted_input, policy_state, time_step.step_type)
    else:
        raise NotImplementedError


    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution,
                                          actions_or_distributions)
    return policy_step.PolicyStep(distributions, policy_state)
