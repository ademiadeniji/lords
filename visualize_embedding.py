r"""Train and Eval DIAYN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
import time
import imageio
import IPython
import base64
import pdb
import copy
import d4rl

from absl import app
from absl import flags
from absl import logging
from sklearn.cluster import KMeans

import gin
import tensorflow as tf
from tensorflow.python.framework import tensor_spec as tspec
from tf_agents.trajectories import time_step as ts
from collections import OrderedDict
import matplotlib.pyplot as plt

import gym
from gym.wrappers import FlattenObservation


from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import normal_projection_network
from tf_agents.policies import actor_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from agents.diayn import diayn_agent
from networks import critic_network
from utils import diayn_gym_env_fixed
from networks import actor_distribution_network

from agents.lacma import lacma_agent
from policies import latent_actor_policy
import numpy as np

flags.DEFINE_integer(
    'dim_z', 4, 'Embedding size.'
)
flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'AntSpace-v1', 'Name of an environment')
flags.DEFINE_boolean('object', False, 'If object-centric embedding')

@gin.configurable
def std_clip_transform(stddevs):
    stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -20, 2),
                                    stddevs)
    return tf.exp(stddevs)

@gin.configurable
def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.001):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        init_means_output_factor=init_means_output_factor,
        std_transform=std_clip_transform,
        scale_distribution=True,
        state_dependent_std=True)

class DictConcatenateLayer(tf.keras.layers.Layer):

    def __init__(self, axis=-1):
        super(DictConcatenateLayer, self).__init__()
        self._concat_layer = tf.keras.layers.Concatenate(axis=axis)
        self._axis = -1

    def call(self, inputs):
        list_inputs = list(inputs.values())
        return self._concat_layer(list_inputs)

    def get_config(self):
        return {'axis': self._axis}

class OneHotConcatenateLayer(DictConcatenateLayer):

    def __init__(self, depth, axis=-1):
        super(OneHotConcatenateLayer, self).__init__(axis=axis)
        self._depth = depth

    def call(self, inputs):
        one_hot = tf.one_hot(inputs['z'], depth=self._depth)
        list_inputs = [inputs['observation'], one_hot]
        return self._concat_layer(list_inputs)

    def get_config(self):
        return {'depth': self._depth, 'axis': self._axis}

def evaluate_embedding():
    env_load_fn=suite_mujoco.load
    categorical = True
    embedding_size = 2
    embedding_layer_size = 256
    FLAGS = flags.FLAGS
    dim_z = FLAGS.dim_z
    eval_env_name = FLAGS.env_name
    has_object = FLAGS.object
    num_clusters = 4
    embedding_path = FLAGS.root_dir + "/embeddings"
    if not path.exists(embedding_path):
        os.mkdir(embedding_path)

    def _env_load_fn(env_name):
        diayn_wrapper = (
            lambda x: diayn_gym_env_fixed.DiaynGymEnvFixed(x, dim_z, categorical)
        )
        return env_load_fn(
            env_name,
            gym_env_wrappers=[FlattenObservation, diayn_wrapper],
        )

    root_dir = FLAGS.root_dir
    embedding_fc_layers=(embedding_layer_size, embedding_layer_size)
    _preprocessing_combiner = DictConcatenateLayer()
    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_env = tf_py_environment.TFPyEnvironment(_env_load_fn(eval_env_name))
    eval_py_env = _env_load_fn(eval_env_name)
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation["observation"]
    action_spec = tf_env.action_spec()
    augmented_time_step_spec = tf_env.time_step_spec()
    augmented_observation_spec = augmented_time_step_spec.observation
    z_spec = augmented_observation_spec["z"]

    if tensor_spec.is_discrete(z_spec):
        _preprocessing_combiner = OneHotConcatenateLayer(dim_z)
    else:
        _preprocessing_combiner = DictConcatenateLayer()

    embedding_spec = tspec.BoundedTensorSpec(
                    shape=[embedding_size],
                    dtype=tf.float32,
                    minimum=-100,
                    maximum=100,
                    name='embedding')
    embedding_net = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=embedding_spec,
        fc_layer_params=embedding_fc_layers,
        continuous_projection_net=normal_projection_net)

    embedding_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'diayn_embedding'),
        embedding_net=embedding_net,
        global_step=global_step
    )
    embedding_checkpointer.initialize_or_restore()
    
    if has_object:
        obs = np.copy(tf_env.reset().observation["observation"].numpy())
    else:
        xy = np.expand_dims(np.array([0, 0]), axis=0)
        _obs = tf_env.reset().observation["observation"].numpy()[:, 2:]    
        obs = np.concatenate((xy, _obs), axis=1)
    embedding_grid_filename = root_dir + '/embeddings/embedding_grid.png'
    embedding_grid_filename_1 = root_dir + '/embeddings/embedding_grid_1.png'
    embedding_grid_filename_2 = root_dir + '/embeddings/embedding_grid_2.png'
    embedding_line_filename_1 = root_dir + '/embeddings/embedding_line_1.png'
    embedding_line_filename_2 = root_dir + '/embeddings/embedding_line_2.png'

    points = []
    for i in range(0, 21, 1):
        for j in range (0, 21, 1):
            points.append([i/10, j/10])

    embeddings = []
    embeddings_dim_1 = []
    embeddings_dim_2 = []
    for pair in points:
        x = pair[0]
        y = pair[1]
        curr_obs = copy.deepcopy(obs)
        if has_object:
            curr_obs[:, 3] = x
            curr_obs[:, 4] = y
        else:
            curr_obs[:, 0] = x
            curr_obs[:, 1] = y
        obs_tensor = tf.convert_to_tensor(curr_obs)
        embedding_dist, _ = embedding_net(obs_tensor, ts.StepType.MID, ())
        embedding = embedding_dist.sample()
        embeddings.append(np.squeeze(embedding))
        embeddings_dim_1.append([np.squeeze(embedding)[0]])
        embeddings_dim_2.append([np.squeeze(embedding)[1]])
        print("point: ({}, {}), embedding: {}".format(x, y, embedding))

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    kmeans_1 = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_dim_1)
    kmeans_2 = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_dim_2)
    print(kmeans.cluster_centers_)

    labels_to_points = {}
    for pair in zip(kmeans.labels_, points):
        if pair[0] in labels_to_points:
            labels_to_points[pair[0]].append(pair[1])
        else:
            labels_to_points[pair[0]] = [pair[1]]
    for i in range(num_clusters):
        plt.scatter(*zip(*labels_to_points[i]))
    plt.savefig(embedding_grid_filename)
    plt.clf()

    labels_to_points = {}
    for pair in zip(kmeans_1.labels_, points):
        if pair[0] in labels_to_points:
            labels_to_points[pair[0]].append(pair[1])
        else:
            labels_to_points[pair[0]] = [pair[1]]
    for i in range(num_clusters):
        plt.scatter(*zip(*labels_to_points[i]))
    plt.savefig(embedding_grid_filename_1)
    plt.clf()

    labels_to_points = {}
    for pair in zip(kmeans_2.labels_, points):
        if pair[0] in labels_to_points:
            labels_to_points[pair[0]].append(pair[1])
        else:
            labels_to_points[pair[0]] = [pair[1]]
    for i in range(num_clusters):
        plt.scatter(*zip(*labels_to_points[i]))
    plt.savefig(embedding_grid_filename_2)
    plt.clf()

    points_to_embeddings = {}
    for pair in zip(points, embeddings):
        points_to_embeddings[tuple(pair[0])] = pair[1]

    x_embeddings = []
    y_embeddings = []
    x_axis = np.arange(0, 21, 1)
    for y in range(0, 21, 1):
        x_embeddings.append(points_to_embeddings[tuple([0.5, y/10])][0])
        y_embeddings.append(points_to_embeddings[tuple([0.5, y/10])][1])
    plt.title("values along vertical line")
    plt.plot(x_axis, x_embeddings, label='embedding_dim_1')
    plt.plot(x_axis, y_embeddings, label='embedding_dim_2')
    plt.legend(loc="upper left")
    plt.savefig(embedding_line_filename_1)

    plt.clf()
    x_embeddings = []
    y_embeddings = []
    for x in range(0, 21, 1):
        x_embeddings.append(points_to_embeddings[tuple([x/10, 0])][0])
        y_embeddings.append(points_to_embeddings[tuple([x/10, 0])][1])
    plt.title("values along horizontal line")
    plt.plot(x_axis, x_embeddings, label='embedding_dim_1')
    plt.plot(x_axis, y_embeddings, label='embedding_dim_2')
    plt.legend(loc="upper left")
    plt.savefig(embedding_line_filename_2)

def main(_):
    tf.compat.v1.enable_v2_behavior()
    evaluate_embedding()


if __name__ == '__main__':
    app.run(main)
