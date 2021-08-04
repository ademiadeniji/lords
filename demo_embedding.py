r"""Train and Eval demo embedding.
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
import numpy as np

from collections import namedtuple
from collections import OrderedDict 
import matplotlib.pyplot as plt

import gym
from gym.wrappers import FlattenObservation

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from utils import dynamic_demonstration_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_gym
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import actor_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tensorflow.python.framework import tensor_spec as tspec

from networks import actor_distribution_network
from utils import diayn_gym_env
from utils import train_utils
from agents.embedding import embedding_agent


flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('teacher_dir', None, 'Directory for teacher policy')
flags.DEFINE_string('env_name', "FetchReach-v1", 'Environment to run algorithm in.')
flags.DEFINE_multi_string(
    'gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string(
    'gin_param', None, 'Gin binding to pass through.')
flags.DEFINE_boolean(
    'debug', True, 'If printing debug summaries')
flags.DEFINE_integer(
    'dim_z', 4, 'Embedding size.'
)
flags.DEFINE_boolean(
    'categorical', True, 'If using categorical embeddings.'
)
flags.DEFINE_integer(
    'option_length', 2, 'Length of sub-episodic option.'
)
flags.DEFINE_integer(
    'num_demonstrations', 10, 'Number of demonstrations to collect per skill'
)
flags.DEFINE_integer(
    'num_demo_skills', None, 'How many skills to collect demos for if different from dim_z'
)
flags.DEFINE_float(
    'skill_noise_epsilon', None, 'Probability to use random skill'
)
flags.DEFINE_integer(
    'embedding_size', 2, 'Size of state embedding for inference'
)
flags.DEFINE_string(
    'inference_type', 'dynamics_displacement', 'Input to inference network.'
)
flags.DEFINE_boolean(
    'xy_prediction', False, 'Whether to predict xy'
)
flags.DEFINE_float(
    'linear_loss', 0.0, 'Regularization term for linearity of embeddings'
)
flags.DEFINE_float(
    'dynamics_loss', 0.0, 'Regularization term for predictability of embeddings'
)
flags.DEFINE_float(
    'l1_reg', 0.0, 'Sparsity Regularization'
)
flags.DEFINE_boolean(
    'only_collect', False, 'Collect and plot demos then exit'
)
flags.DEFINE_boolean('object', False, 'Object-Centric demos')
flags.DEFINE_boolean(
    'learn_mask', False, 'Whether to learn observation mask'
)
FLAGS = flags.FLAGS

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
        mean_transform=None, # don't allow tanh squash to spec -1 to 1
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


@gin.configurable  # NOQA
def train_eval(
        root_dir,
        teacher_dir,
        env_name=None,
        eval_env_name=None,
        env_load_fn=suite_mujoco.load,
        num_iterations=1000000,
        dim_z=None,
        categorical=None,
        actor_fc_layers=(256, 256),
        inference_fc_layers=(2,),
        embedding_fc_layers=(256, 256),
        dynamics_fc_layers=(256, 256),
        num_parallel_environments=1,
        # Params for collect
        num_demonstrations=None,
        replay_buffer_capacity=1000000, 
        option_length=None,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=256,
        learning_rate=3e-4,
        use_tf_functions=True,
        embedding_size=None,
        num_demo_skills=None,
        skill_noise_epsilon=None,
        inference_type=None,
        xy_prediction=None,
        linear_loss=None,
        dynamics_loss=None,
        l1_reg=None,
        learn_mask=None,
        # Params for eval
        num_eval_episodes=30,
        # Params for summaries and logging
        checkpoint_interval=25000,
        log_interval=1000,
        summary_interval=1000,
        summaries_flush_secs=10,
        debug_summaries=True,
        summarize_grads_and_vars=True,
        ):
    """A simple train and eval for DIAYN on Mujoco.
    
    All hyperparameters come from the original DIAYN paper
    (https://arxiv.org/pdf/1801.01290.pdf).
    """
    if num_demo_skills is None:
        num_demo_skills = dim_z

    episode_length = option_length
    
    root_dir = os.path.expanduser(root_dir)
    if not path.exists(root_dir):
        os.mkdir(root_dir)

    summary_writer = tf.compat.v2.summary.create_file_writer(
        root_dir, flush_millis=summaries_flush_secs * 1000)
    summary_writer.set_as_default()

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

        def diayn_env_load_fn(env_name):
            diayn_wrapper = (
                lambda x: diayn_gym_env.DiaynGymEnv(x, dim_z, categorical)
            )
            return env_load_fn(
                env_name,
                gym_env_wrappers=[FlattenObservation, diayn_wrapper],
            )

        _env_load_fn = diayn_env_load_fn

        # create training environment
        if num_parallel_environments == 1:
            py_env = _env_load_fn(env_name)
        else:
            py_env = parallel_py_environment.ParallelPyEnvironment(
                [lambda: _env_load_fn(env_name)] * num_parallel_environments)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
       
        eval_env_name = eval_env_name or env_name
        eval_py_env = _env_load_fn(eval_env_name)

        augmented_time_step_spec = tf_env.time_step_spec()
        augmented_observation_spec = augmented_time_step_spec.observation
        observation_spec = augmented_observation_spec['observation']
        if inference_type == "transition":
            observation_spec = tspec.BoundedTensorSpec(
                    shape=[observation_spec.shape[0]*2],
                    dtype=tf.float64,
                    minimum=-1.79769313e+308,
                    maximum=1.79769313e+308,
                    name='observation')
        z_spec = augmented_observation_spec['z']
        reward_spec = augmented_time_step_spec.reward
        action_spec = tf_env.action_spec()
        time_step_spec = ts.time_step_spec(observation_spec)

        if learn_mask:
            embedding_size = observation_spec.shape[0]

        if tensor_spec.is_discrete(z_spec):
            _preprocessing_combiner = OneHotConcatenateLayer(dim_z)
        else:
            _preprocessing_combiner = DictConcatenateLayer()
        
        if teacher_dir is not None:
            teacher_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=augmented_observation_spec,
                output_tensor_spec=action_spec,
                preprocessing_combiner=_preprocessing_combiner,
                fc_layer_params=actor_fc_layers,
                continuous_projection_net=normal_projection_net,
                mask_xy=True,
                name="teacher_net")
        else:
            raise NotImplementedError

        if embedding_size:
            embedding_spec = tspec.BoundedTensorSpec(
                    shape=[embedding_size],
                    dtype=tf.float32,
                    minimum=-100,
                    maximum=100,
                    name='embedding')
            if not learn_mask:
                embedding_net = actor_distribution_network.ActorDistributionNetwork(
                    input_tensor_spec=observation_spec,
                    output_tensor_spec=embedding_spec,
                    fc_layer_params=embedding_fc_layers,
                    continuous_projection_net=normal_projection_net)
                if inference_type == "dynamics_transition":
                    dynamics_embedding_spec = tspec.BoundedTensorSpec(
                        shape=[embedding_size*2],
                        dtype=tf.float32,
                        minimum=-100,
                        maximum=100,
                        name='dynamics')
            else:
                embedding_net = None
        else:
            raise NotImplementedError
        
        if dynamics_loss > 0:
            latent_dynamics_embedding_spec = tspec.BoundedTensorSpec(
                shape=[embedding_size+num_demo_skills],
                dtype=tf.float32,
                minimum=-100,
                maximum=100,
                name='latent_dynamics')

        if inference_type == "dynamics_transition":
            inference_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=dynamics_embedding_spec,
                output_tensor_spec=z_spec,
                fc_layer_params=inference_fc_layers,
                continuous_projection_net=normal_projection_net,
                name="pred_net")
        else:
            inference_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=embedding_spec,
                output_tensor_spec=z_spec,
                fc_layer_params=inference_fc_layers,
                continuous_projection_net=normal_projection_net,
                dropout_layer_params=(0.7,),
                name="pred_net")
        if dynamics_loss > 0:
            dynamics_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=latent_dynamics_embedding_spec,
                output_tensor_spec=embedding_spec,
                fc_layer_params=dynamics_fc_layers,
                continuous_projection_net=normal_projection_net,
                name="dyna_net")
        else:
            dynamics_net = None

        if teacher_net:
            teacher_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(teacher_dir, 'diayn_actor'),
                actor_net=teacher_net,
                global_step=global_step
            )
            teacher_checkpointer.initialize_or_restore()
            teacher_policy = actor_policy.ActorPolicy(
                time_step_spec=augmented_time_step_spec,
                action_spec=action_spec,
                actor_network=teacher_net,
                training=False)
            teacher_policy_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(root_dir, 'diayn_policy'),
                policy=teacher_policy,
                global_step=global_step)
            teacher_policy_checkpointer.initialize_or_restore()

        if embedding_net:
            embedding_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(root_dir, 'diayn_embedding'),
                embedding_net=embedding_net,
                global_step=global_step
            )
            embedding_checkpointer.initialize_or_restore()

        inference_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'inference'),
            inference_net=inference_net,
            global_step=global_step
        )
        inference_checkpointer.initialize_or_restore()
        
        if dynamics_loss:
            dynamics_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(root_dir, 'dynamics'),
                dynamics_net=dynamics_net,
                global_step=global_step
            )
            dynamics_checkpointer.initialize_or_restore()
        
        if learn_mask:
            observation_size = time_step_spec.observation.shape[0]
            observation_mask = tf.Variable(tf.zeros(shape=(observation_size,), dtype=tf.float64), trainable=True, \
                shape=(observation_size,), name='observation_mask', dtype=tf.float64)
            observation_mask_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(root_dir, 'observation_mask'),
                observation_mask=observation_mask,
                global_step=global_step
            )
            observation_mask_checkpointer.initialize_or_restore()
        else:
            observation_mask = None
        print(observation_mask)
        pdb.set_trace()
        tf_agent = embedding_agent.EmbeddingAgent(
                time_step_spec,
                action_spec,
                z_spec,
                embedding_spec,
                num_demo_skills,
                inference_net,
                embedding_net,
                dynamics_net,
                teacher_policy,
                optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate),
                option_length=option_length,
                inference_type=inference_type,
                xy_prediction=xy_prediction,
                linear_loss=linear_loss,
                dynamics_loss=dynamics_loss,
                l1_reg=l1_reg,
                learn_mask=learn_mask,
                observation_mask=observation_mask,
                batch_size=batch_size,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars,
                train_step_counter=global_step)
        tf_agent.initialize()
      
        # Make the replay buffer.
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)
        replay_buffer_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_replay_buffer'),
            replay_buffer=replay_buffer,
            global_step=global_step
        )
        replay_buffer_checkpointer.initialize_or_restore()
        replay_observer = [replay_buffer.add_batch]

        if num_demonstrations is not None:
            teacher_demonstration_driver = dynamic_demonstration_driver.DynamicDemonstrationDriver(
                tf_env, 
                teacher_policy,
                observers=replay_observer,
                num_episodes=num_demonstrations,
                num_skills=num_demo_skills,
                skill_noise_epsilon=skill_noise_epsilon)
        else:
            raise NotImplementedError

        if use_tf_functions:
            teacher_demonstration_driver.run = common.function(
                teacher_demonstration_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        # Collect initial replay data.
        if replay_buffer.num_frames().numpy() == 0:
            logging.info(
                'Initializing replay buffer by collecting experience for '
                '%d demonstration(s) per skill with teacher policy.', num_demonstrations)
            teacher_demonstration_driver.run()
        
        time_acc = 0
       
        # Build dataset from replay buffer
        # Dataset generates trajectories with shape [Bx2x...]
        def _filter_invalid_transition(trajectories, unused_arg1):
            return tf.reduce_all(~trajectories.is_boundary()[:-1]) and \
            tf.reduce_all(~trajectories.is_first()[1:])

        dataset = replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=option_length).unbatch().filter(
                _filter_invalid_transition).batch(batch_size).prefetch(5)
        iterator = iter(dataset)

        logging.info('Plotting collected skill demonstrations')
        color_wheel = ['b', 'r', 'g', 'c', 'm']
        collected_observations = replay_buffer.gather_all().observation["observation"].numpy()
        if FLAGS.object:
            collected_xy = np.squeeze(collected_observations[:, :, 3:5], 0)
        else:
            collected_xy = np.squeeze(collected_observations[:, :, 0:2], 0)
        collected_x = collected_xy[:, 0]
        collected_y = collected_xy[:, 1]
        if "Fetch" in env_name: 
            plt.xlim(1.1, 1.5)
            plt.ylim(0.6, 1.0)
        else:
            plt.xlim(-15, 15)
            plt.ylim(-15, 15)
        for i in range(num_demo_skills*num_demonstrations):
            plt.plot(collected_x[(episode_length+1)*i:(episode_length+1)*i+episode_length], 
                collected_y[(episode_length+1)*i:(episode_length+1)*i+episode_length], 
                color_wheel[(i//num_demonstrations)%(num_demo_skills*num_demonstrations)])
        plt.savefig(root_dir + "/collected_skills.png")
     
        if FLAGS.only_collect:
            replay_buffer_checkpointer.save(global_step=global_step.numpy())
            raise NotImplementedError
        
        logging.info('Training embedding on %d demo(s) for each of %d skills',
         num_demonstrations, num_demo_skills)

        # Train evaluation loop
        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience)

        if use_tf_functions:
            train_step = common.function(train_step)
       
        for _ in range(num_iterations):
            start_time = time.time()
            for _ in range(train_steps_per_iteration):
                train_loss = train_step() 
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
                logging.info('step = %d, total_loss = %f, inference_loss = %f, linear_loss = %f, dynamics_loss = %f', \
                     global_step.numpy(), train_loss.extra.total_loss, train_loss.extra.inference_loss, 
                     train_loss.extra.linear_loss, train_loss.extra.dynamics_loss)

            if global_step.numpy() % checkpoint_interval == 0:
                if embedding_net: embedding_checkpointer.save(global_step=global_step.numpy())
                inference_checkpointer.save(global_step=global_step.numpy())
                if dynamics_loss: dynamics_checkpointer.save(global_step=global_step.numpy())
                replay_buffer_checkpointer.save(global_step=global_step.numpy())
                if learn_mask: observation_mask_checkpointer.save(global_step=global_step.numpy())

def main(_):
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    train_eval(FLAGS.root_dir,
               teacher_dir = FLAGS.teacher_dir,
               env_name = FLAGS.env_name,
               dim_z = FLAGS.dim_z,
               categorical = FLAGS.categorical,
               num_demonstrations = FLAGS.num_demonstrations,
               option_length = FLAGS.option_length,
               embedding_size = FLAGS.embedding_size,
               num_demo_skills = FLAGS.num_demo_skills,
               skill_noise_epsilon = FLAGS.skill_noise_epsilon,
               inference_type = FLAGS.inference_type,
               xy_prediction = FLAGS.xy_prediction,
               linear_loss = FLAGS.linear_loss,
               dynamics_loss = FLAGS.dynamics_loss,
               l1_reg = FLAGS.l1_reg,
               learn_mask = FLAGS.learn_mask,
               debug_summaries=FLAGS.debug)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
