"""Sample Keras actor network that generates distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tensorflow.python.framework import tensor_spec as tspec

from networks import utils as network_utils  # NOQA
from networks import multiplex_projection_network
import pdb


def scaled_tanh_squash_to_spec(inputs, spec, scale=5.0):
    """Maps inputs with arbitrary range to range defined by spec with tanh."""
    # Used to prevent the means become too large in TransformedDistribution.
    # This is a temporal solution.
    means = (spec.maximum + spec.minimum) / 2.0
    magnitudes = (spec.maximum - spec.minimum) / 2.0
    return means + magnitudes * tf.tanh(inputs) * scale


def std_clip_transform(stddevs):
    # stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -20, 2),
    #                                 stddevs)
    stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -5, 2),
                                    stddevs)
    return tf.exp(stddevs)


def multiplex_projection_net(action_spec,
                             z_spec,
                             init_action_stddev=0.35,
                             init_means_output_factor=0.1):
    del init_action_stddev
    return multiplex_projection_network.MultiplexProjectionNetwork(
        action_spec,
        z_spec=z_spec,
        mean_transform=scaled_tanh_squash_to_spec,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=std_clip_transform,
        scale_distribution=True)


@gin.configurable
class MultiplexActorDistributionNetwork(network.DistributionNetwork):
    """Creates an actor producing either Normal or Categorical distribution.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 state_spec,
                 obs_encoder_ctor=None,
                 fc_layer_params=None,
                 parameter_encoder_ctor=None,
                 dtype=tf.float32,
                 name='ActorDistributionNetwork'):
        """Creates an instance of `ActorDistributionNetwork`.
        """
        del parameter_encoder_ctor
        
        self._num_skills = state_spec.maximum - state_spec.minimum + 1
        raw_obs_size = input_tensor_spec.shape[0] - self._num_skills
        observation_spec = tspec.BoundedTensorSpec(
                    shape=[256, raw_obs_size],
                    dtype=tf.float32,
                    minimum=-3.4028234663852886e+38,
                    maximum=3.4028234663852886e+38,
                    name='observation')

        def map_proj(spec):
            assert not tensor_spec.is_discrete(spec)
            return multiplex_projection_net(
                spec,
                z_spec=state_spec)

        projection_networks = tf.nest.map_structure(
            map_proj, output_tensor_spec)
        output_spec = tf.nest.map_structure(
            lambda proj_net: proj_net.output_spec, projection_networks)

        super(MultiplexActorDistributionNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=state_spec,
            output_spec=output_spec,
            name=name)

        self._obs_encoder = obs_encoder_ctor(
            input_tensor_spec=observation_spec,
            fc_layer_params=fc_layer_params)

        self._projection_networks = projection_networks
        self._output_tensor_spec = output_tensor_spec

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec

    def call(self, inputs, step_type, network_state, training=False):
        if len(inputs.shape) > 2:
            inputs = tf.squeeze(inputs, axis=1)
        obs_and_skill = inputs
        obs = obs_and_skill[:, :-self._num_skills]
        z = obs_and_skill[:, -self._num_skills:]
        z = tf.math.argmax(z, axis=-1)

        # Observation encoding.
        state, _ = self._obs_encoder(obs, training=training)
        outer_rank = nest_utils.get_outer_rank(
            inputs, self.input_tensor_spec)
        output_actions = tf.nest.map_structure(
            lambda proj_net: proj_net(state, z, outer_rank),
            self._projection_networks)

        return output_actions, network_state


@gin.configurable
class MultiplexCriticNetwork(network.Network):
    """Creates a critic network."""

    def __init__(self,
                 input_tensor_spec,
                 state_spec,
                 obs_encoder_ctor=None,
                 fc_layer_params=None,
                 parameter_encoder_ctor=None,
                 name='CriticNetwork'):
        """Creates an instance of `CriticNetwork`.
        """
        del parameter_encoder_ctor

        super(MultiplexCriticNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=state_spec,
            name=name)

        observation_spec, action_spec = input_tensor_spec
        num_skills = state_spec.maximum - state_spec.minimum + 1
        raw_obs_size = observation_spec.shape[0] - num_skills
        observation_spec = tspec.BoundedTensorSpec(
                    shape=[256, raw_obs_size],
                    dtype=tf.float32,
                    minimum=-3.4028234663852886e+38,
                    maximum=3.4028234663852886e+38,
                    name='observation')

        assert state_spec.dtype.is_integer
        self._num_skills = state_spec.maximum - state_spec.minimum + 1

        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')
        self._single_action_spec = flat_action_spec[0]
        assert not self._single_action_spec.dtype.is_integer

        self._obs_encoder = obs_encoder_ctor(
            input_tensor_spec=observation_spec,
            fc_layer_params=fc_layer_params)

        self._action_encoding_layers = [
            tf.keras.layers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_regularizer=None),
        ]

        self._joint_layers = [
            tf.keras.layers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_regularizer=None),
        ]

        self._value_layer = tf.keras.layers.Dense(
            self._num_skills,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='value')

    def call(self, inputs, step_type=(), network_state=(), training=False):
        obs_and_skill, action = inputs
        if len(obs_and_skill.shape) == 3 and len(action.shape) == 2:
            obs = obs_and_skill[:, :, :-self._num_skills]
            z = obs_and_skill[:, :, -self._num_skills:]
            action = tf.expand_dims(action, axis=1)
        elif len(obs_and_skill.shape) == 3:
            obs = obs_and_skill[:, :, :-self._num_skills]
            z = obs_and_skill[:, :, -self._num_skills:]
        else:
            obs = obs_and_skill[:, :-self._num_skills]
            z = obs_and_skill[:, -self._num_skills:]
        z = tf.math.argmax(z, axis=-1)

        # Observation encoding.
        obs, _ = self._obs_encoder(obs, training=training)

        # Action encoding.
        for layer in self._action_encoding_layers:
            action = layer(action, training=training)

        state = tf.concat([obs, action], axis=-1)
        for layer in self._joint_layers:
            state = layer(state, training=training)

        # Values.
        value = self._value_layer(state, training=training)
        value = network_utils.task_multiplex(value, z, self._num_skills)
        return tf.reshape(value, [-1]), network_state
