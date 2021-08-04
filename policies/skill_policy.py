"""Skill Policy based on an actor network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import reparameterized_sampling
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.specs import tensor_spec
from collections import OrderedDict
from tf_agents.utils import common
import pdb

@gin.configurable
class SkillPolicy(tf.Module):
    """Class to build Actor Policies."""

    def __init__(self,
                 time_step_spec,
                 z_spec,
                 action_spec,
                 generator_network,
                 steps_per_option,
                 dim_z,
                 concat_obs=False,
                 info_spec=(),
                 observation_normalizer=None,
                 clip=True,
                 training=False,
                 name=None):
        """Builds an Actor Policy given a actor network.

        Args:
            time_step_spec: A `TimeStep` spec of the expected time_steps.
            action_spec: A nest of BoundedTensorSpec representing the actions.
            generator_network: An instance of a
                `tf_agents.networks.network.Network` to be used by the policy.
                The network will be called with call(observation, step_type,
                policy_state) and should return (actions_or_distributions,
                new_state).
            info_spec: A nest of TensorSpec representing the policy info.
            observation_normalizer: An object to use for observation
                normalization.
            clip: Whether to clip actions to spec before returning them.
                Default True. Most policy-based algorithms (PCL, PPO,
                REINFORCE) use unclipped continuous actions for training.
            training: Whether the network should be called in training mode.
            name: The name of this policy. All variables in this module will
                fall under that name. Defaults to the class name.

        Raises:
            ValueError: if generator_network is not of type network.Network.
        """
        if not isinstance(generator_network, network.Network):
            raise ValueError('generator_network must be a network.Network. Found '
                             '{}.'.format(type(generator_network)))
        generator_network.create_variables()
        self._generator_network = generator_network
        self._observation_normalizer = observation_normalizer
        self._training = training

        self.time_step_spec=time_step_spec
        self.z_spec=z_spec
        self.action_spec=action_spec
        self.policy_state_spec=generator_network.state_spec
        self.info_spec=info_spec
        self.clip=clip
        self.steps_per_option = steps_per_option
        self.concat_obs = concat_obs
        self.dim_z = dim_z

    def _variables(self):
        return self._generator_network.variables

    def distribution(self, time_step, z, policy_state, seed=1):
        seed_stream = tfp.util.SeedStream(seed=seed, salt='ppo_policy')
        if self.concat_obs:
            one_hot = tf.one_hot(z, depth=self.dim_z)
            list_inputs = [tf.cast(time_step.observation, tf.float32), one_hot]
            obs_and_skill = tf.concat(list_inputs, axis=-1)
            action_distribution, _ = self._generator_network(obs_and_skill, time_step.step_type, policy_state)
        else:
            action_distribution, _ = self._generator_network(OrderedDict({"observation":
                time_step.observation, "z":z}), time_step.step_type, policy_state)
            
        def _to_distribution(action_or_distribution):
            if isinstance(action_or_distribution, tf.Tensor):
                # This is an action tensor, so wrap it in a deterministic
                # distribution.
                return tfp.distributions.Deterministic(
                    loc=action_or_distribution)
            return action_or_distribution

        distributions = tf.nest.map_structure(_to_distribution,
                                              action_distribution)
        return policy_step.PolicyStep(distributions, policy_state)

    def action(self, time_step, z, policy_state=(), seed=1):
        """Implementation of `action`.

        Args:
            time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
            policy_state: A Tensor, or a nested dict, list or tuple of Tensors
                representing the previous policy_state.
            seed: Seed to use if action performs sampling (optional).

        Returns:
            A `PolicyStep` named tuple containing:
                `action`: An action Tensor matching the `action_spec()`.
                `state`: A policy state tensor to be fed into the next call.
                `info`: Optional information such as action log probabilities.
        """
        seed_stream = tfp.util.SeedStream(seed=seed, salt='ppo_policy')
        distribution_step = self.distribution(time_step, z, policy_state)
        info = distribution_step.info
        actions = tf.nest.map_structure(
            lambda d: d.sample(), distribution_step.action)
        step = distribution_step._replace(action=actions, info=info, state=policy_state)
        def clip_action(action, action_spec):
            if isinstance(action_spec, tensor_spec.BoundedTensorSpec):
                return common.clip_to_spec(action, action_spec)
            return action
        if self.clip:
            clipped_actions = tf.nest.map_structure(clip_action,
                                                    step.action,
                                                    self.action_spec)
            step = step._replace(action=clipped_actions)
        return step
    
    def get_initial_state(self, batch_size):
        """Returns the initial state of the policy network.
        Args:
        batch_size: A constant or Tensor holding the batch size. Can be None, in
            which case the state will not have a batch dimension added.
        Returns:
        A nest of zero tensors matching the spec of the policy network state.
        """
        return tensor_spec.zero_spec_nest(
            self.policy_state_spec,
            outer_dims=None if batch_size is None else [batch_size])