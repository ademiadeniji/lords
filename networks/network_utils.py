"""Network utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

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