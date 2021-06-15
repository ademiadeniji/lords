"""Network utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.utils import nest_utils
from tf_agents.trajectories import time_step as ts


def expand_and_tile(inputs, outer_shape, batch_dims=0):
    inputs_shape = list(inputs.shape)

    for _ in range(len(outer_shape)):
        inputs = tf.expand_dims(inputs, axis=batch_dims)

    tile_shape = tf.concat([
        tf.cast([1] * batch_dims, dtype=tf.int32),
        tf.cast(outer_shape, dtype=tf.int32),
        tf.cast([1] * (len(inputs_shape) - batch_dims), dtype=tf.int32),
    ],
                           axis=-1)
    return tf.multiply(inputs, tf.ones(tile_shape, dtype=inputs.dtype))


def tanh_squash_to_spec(inputs, spec, scale=1.0):
    inputs_mean = 0.5 * (spec.maximum + spec.minimum)
    inputs_range = 0.5 * (spec.maximum - spec.minimum) * scale
    return tf.tanh(inputs) * inputs_range + inputs_mean


def one_hot(input_tensor, spec):
    assert spec.dtype.is_integer
    return tf.one_hot(input_tensor - spec.minimum,
                      depth=spec.maximum - spec.minimum + 1,
                      dtype=tf.float32)


def convert_agent_pos(input_tensor, spec):
    if spec.maximum.ndim == 2:
        width = int(spec.maximum[0])
        height = int(spec.maximum[1])
    else:
        width = int(spec.maximum)
        height = int(spec.maximum)

    # Conver (x, y) coordinates to spatial one-hot array.
    outer_rank = nest_utils.get_outer_rank(input_tensor, spec)
    x_grid = tf.one_hot(input_tensor[..., 0], depth=width, dtype=tf.float32)
    x_grid = tf.tile(tf.expand_dims(x_grid, axis=-1),
                     [1] * outer_rank + [1, height])
    y_grid = tf.one_hot(input_tensor[..., 1], depth=height, dtype=tf.float32)
    y_grid = tf.tile(tf.expand_dims(y_grid, axis=-2),
                     [1] * outer_rank + [width, 1])
    pos_grid = x_grid * y_grid

    # Remove the agent if it is on the border walls.
    pos_grid = pos_grid[..., 1:-1, 1:-1]
    paddings = tf.constant([[0, 0]] * outer_rank + [[1, 1], [1, 1]])
    pos_grid = tf.pad(pos_grid, paddings)

    pos_grid = tf.expand_dims(pos_grid, axis=-1)
    return pos_grid


def get_displacement_mask(body_pos, next_body_pos, thresh=1e-3):
    delta_body_pos = next_body_pos - body_pos
    displacement_mask = tf.greater(tf.norm(delta_body_pos, axis=-1), thresh)
    displacement_mask = tf.expand_dims(displacement_mask, axis=-1)
    displacement_mask = tf.cast(displacement_mask, tf.float32)
    return displacement_mask


def episode_mask(step_type):
    is_first_step = tf.cast(step_type == ts.StepType.FIRST, tf.float32)
    is_first_episode = tf.math.cumsum(is_first_step, axis=-1)
    mask = tf.cast(is_first_episode == 1, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def task_multiplex(inputs, z, num_tasks):
    """The multiplex module for multitask data.

    Args:
        inputs: Tensor of shape [batch_size, ..., num_tasks * dim_outputs].
        z: The integer task index of shape [batch_size].
        num_tasks: The number of tasks.

    Returns:
        A tensor of shape [batch_size, ..., dim_outputs].
    """
    # dim_inputs = num_tasks * dim_outputs
    dim_inputs = int(inputs.shape[-1])
    assert dim_inputs % num_tasks == 0
    dim_outputs = int(dim_inputs / num_tasks)
    new_shape = tf.concat(
        [tf.shape(inputs)[:-1], [num_tasks, dim_outputs]],
        axis=-1)

    state = tf.reshape(inputs, new_shape)
    # [batch_size, ..., num_tasks, dim_outputs]

    state = tf.stack(tf.unstack(state, axis=-2), axis=1)
    # [batch_size, num_tasks, ..., dim_outputs]

    indices = tf.expand_dims(z, axis=-1)
    # [batch_size, 1]

    state = tf.gather_nd(
        state,
        indices,
        batch_dims=1)
    # [batch_size, ..., dim_outputs]

    return state


class WeightedMerge(tf.keras.Model):

    def __init__(self,
                 dim_output,
                 axis,
                 activation=tf.nn.relu,
                 name='WeightedMerge'):
        super(WeightedMerge, self).__init__(name=name)

        self._axis = axis

        self._value_layer = tf.keras.layers.Dense(
            dim_output,
            activation=activation,
            kernel_regularizer=None)

        self._weight_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_regularizer=None)

    def call(self, input_tensor, mask=None, training=False):
        value = self._value_layer(input_tensor, training=training)

        weight = self._weight_layer(input_tensor, training=training)
        weight -= (1 - mask) * 1e9
        weight = tf.nn.softmax(weight, axis=self._axis)

        return tf.reduce_sum(weight * value, axis=self._axis)


class AttentionModule(tf.keras.Model):

    def __init__(self,
                 dim_key,
                 dim_value,
                 activation=None,
                 # activation=tf.nn.relu,
                 name='AttentionModule'):
        super(AttentionModule, self).__init__(name=name)

        self._q_layer = tf.keras.layers.Dense(
            dim_key,
            use_bias=False,
            activation=None,
            kernel_regularizer=None)

        self._k_layer = tf.keras.layers.Dense(
            dim_key,
            use_bias=False,
            activation=None,
            kernel_regularizer=None)

        self._v_layer = tf.keras.layers.Dense(
            dim_value,
            use_bias=False,
            activation=activation,
            kernel_regularizer=None)

    def call(self, query, memory, bias=None, training=False):
        q = self._q_layer(query, training=training)
        k = self._k_layer(memory, training=training)
        v = self._v_layer(memory, training=training)

        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias

        weight = tf.nn.softmax(logits, axis=-1)
        state = tf.matmul(weight, v)

        return state