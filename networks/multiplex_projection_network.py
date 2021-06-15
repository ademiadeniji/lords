"""Project inputs to a normal distribution object."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import bias_layer
from tf_agents.networks import network
# from tf_agents.networks import utils as network_utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec

from utils import distribution_utils
from networks.utils import task_multiplex


def tanh_squash_to_spec(inputs, spec):
    """Maps inputs with arbitrary range to range defined by spec with tanh."""
    means = (spec.maximum + spec.minimum) / 2.0
    magnitudes = (spec.maximum - spec.minimum) / 2.0

    return means + magnitudes * tf.tanh(inputs)


@gin.configurable
class MultiplexProjectionNetwork(network.DistributionNetwork):
    """Generates a tfp.distribution.Normal by predicting a mean and std.
    """

    def __init__(self,
                 sample_spec,
                 z_spec,
                 dim_fc=64,
                 activation_fn=None,
                 init_means_output_factor=0.1,
                 std_bias_initializer_value=0.0,
                 mean_transform=tanh_squash_to_spec,
                 std_transform=tf.nn.softplus,
                 state_dependent_std=False,
                 scale_distribution=False,
                 name='ProjectionNetwork'):
        """Creates an instance of NormalProjectionNetwork.
        """
        if len(tf.nest.flatten(sample_spec)) != 1:
            raise ValueError('Normal Projection network only supports '
                             'single spec samples.')

        self._scale_distribution = scale_distribution

        output_spec = self._output_distribution_spec(sample_spec, name)

        super(MultiplexProjectionNetwork, self).__init__(
            # We don't need these, but base class requires them.
            input_tensor_spec=None,
            state_spec=(),
            output_spec=output_spec,
            name=name)

        self._sample_spec = sample_spec
        self._z_spec = z_spec
        self._mean_transform = mean_transform
        self._std_transform = std_transform
        self._state_dependent_std = state_dependent_std

        assert z_spec.dtype.is_integer
        self._num_skills = z_spec.maximum - z_spec.minimum + 1

        self._means_projection_layers = [
            tf.keras.layers.Dense(
                dim_fc,
                activation=tf.nn.relu,
                kernel_regularizer=None),
            tf.keras.layers.Dense(
                self._num_skills * sample_spec.shape.num_elements(),
                activation=activation_fn,
                kernel_initializer=(
                    tf.compat.v1.keras.initializers.VarianceScaling(
                        scale=init_means_output_factor)),
                bias_initializer=tf.keras.initializers.Zeros(),
                name='means_projection_layer')
        ]

        self._stddev_projection_layers = None
        if self._state_dependent_std:
            self._stddev_projection_layers = [
                tf.keras.layers.Dense(
                    dim_fc,
                    activation=tf.nn.relu,
                    kernel_regularizer=None),
                tf.keras.layers.Dense(
                    self._num_skills * sample_spec.shape.num_elements(),
                    activation=activation_fn,
                    kernel_initializer=(
                        tf.compat.v1.keras.initializers.VarianceScaling(
                            scale=init_means_output_factor)),
                    bias_initializer=tf.keras.initializers.Constant(
                            value=std_bias_initializer_value),
                    name='stddev_projection_layer'),
            ]
        else:
            self._bias = bias_layer.BiasLayer(
                    bias_initializer=tf.keras.initializers.Constant(
                            value=std_bias_initializer_value))

    def _output_distribution_spec(self,
                                  sample_spec,
                                  network_name):
        input_param_shapes = tfp.distributions.Normal.param_static_shapes(
            sample_spec.shape)

        input_param_spec = {
            name: tensor_spec.TensorSpec(
                shape=shape,
                dtype=sample_spec.dtype,
                name=network_name + '_' + name)
            for name, shape in input_param_shapes.items()
        }

        def distribution_builder(*args, **kwargs):
            distribution = tfp.distributions.Normal(*args, **kwargs)
            if self._scale_distribution:
                return distribution_utils.scale_distribution_to_spec(
                    distribution, sample_spec)
            return distribution

        return distribution_spec.DistributionSpec(
            distribution_builder,
            input_param_spec,
            sample_spec=sample_spec)

    def call(self, inputs, z, outer_rank):
        if inputs.dtype != self._sample_spec.dtype:
            raise ValueError(
                'Inputs to NormalProjectionNetwork must match the '
                'sample_spec.dtype.')

        means = inputs
        for layer in self._means_projection_layers:
            means = layer(means)

        # # If scaling the distribution later, use a normalized mean.
        # if not self._scale_distribution and self._mean_transform is not None:
        #     means = self._mean_transform(means, self._sample_spec)
        # means = tf.cast(means, self._sample_spec.dtype)

        # We scale the mean even with scaling the distribution to further
        # avoid numerical errors.
        if self._mean_transform is not None:
            means = self._mean_transform(means, self._sample_spec)
        means = tf.cast(means, self._sample_spec.dtype)

        if self._state_dependent_std:
            stds = inputs
            for layer in self._stddev_projection_layers:
                stds = layer(stds)
        else:
            stds = self._bias(tf.zeros_like(means))

        if self._std_transform is not None:
            stds = self._std_transform(stds)
        stds = tf.cast(stds, self._sample_spec.dtype)

        means = task_multiplex(means, z, self._num_skills)
        stds = task_multiplex(stds, z, self._num_skills)

        return self.output_spec.build_distribution(loc=means, scale=stds)