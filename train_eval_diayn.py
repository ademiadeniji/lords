r"""Train and Eval DIAYN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.wrappers import FlattenObservation

import os
from os import path
import copy
import time
import imageio
import IPython
import base64
import pdb
from collections import namedtuple
from collections import OrderedDict 
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents.sac import sac_agent
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
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tensorflow.python.framework import tensor_spec as tspec

from agents.diayn import diayn_agent
from networks import critic_network
from networks import actor_distribution_network
from utils import diayn_gym_env
from utils import train_utils


flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('teacher_dir', None, 'Directory for teacher policy')
flags.DEFINE_string('env', None, 'Environment to run algorithm in (ant or plane with preset params).')
flags.DEFINE_string('env_name', None, 'Environment to run algorithm in.')
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
flags.DEFINE_string(
    'inference_type', 'diayn', 'Input to inference network.'
)
flags.DEFINE_boolean(
    'xy_prior', True, 'If inference is from COM transform.'
)
flags.DEFINE_float(
    'skill_regularization', None, 'Amount to regularize l1-norm of skill embedding for sparsity.'
)
flags.DEFINE_integer(
    'clip_q_value', None, 'Value to clip state action value function output'
)
flags.DEFINE_boolean(
    'clip_q_to_range', False, 'Clip state action value function to experience range'
)
flags.DEFINE_integer(
    'gradient_clipping', None, 'Value to clip gradients to'
)
flags.DEFINE_boolean(
    'twin_critic', False, 'If using twin network from TD3'
)
flags.DEFINE_boolean(
    'mock_inference', False, 'If using mocked inference network'
)
flags.DEFINE_boolean(
    'mock_rewards', False, 'If mocking axis-aligned reward'
)
flags.DEFINE_boolean(
    'partition_rewards', False, 'If using k-means to generate subgoals'
)
flags.DEFINE_boolean(
    'displacement_rewards', False, 'If using displacement reward skills'
)
flags.DEFINE_boolean(
    'push_rewards', False, 'If using handcrafted pushing policy'
)
flags.DEFINE_integer(
    'rl_steps', None, 'Number of steps to train actor'
)
flags.DEFINE_integer(
    'inference_steps', None, 'Number of steps to train inference'
)
flags.DEFINE_integer(
    'temperature', None, 'Sensitivity of posterior distribution'
)
flags.DEFINE_float(
    'reward_scaling', 10.0, 'Amount to scale intrinsic reward by in agent.'
)
flags.DEFINE_float(
    'target_entropy', None, 'Target Entropy for SAC'
)
flags.DEFINE_float(
    'from_origin', None, 'How much to weight auxilliary distance loss'
)
flags.DEFINE_boolean(
    'manual_inference_noise', False, 'Whether to train inference with inputted noise'
)
flags.DEFINE_boolean(
    'extrinsic_rewards', False, 'Whether to train on environment rewards'
)
flags.DEFINE_boolean(
    'sac_baseline', False, 'Whether to train sac baseline'
)
flags.DEFINE_boolean(
    'mask_xy', True, 'Whether to mask xy position information from policy'
)
flags.DEFINE_boolean(
    'reinforce_all_actions', False, 'Whether to reinforce all actions in trajectory'
)
flags.DEFINE_boolean(
    'reinforce_average_action', False, 'Whether to reinforce average of actions in trajectory'
)
flags.DEFINE_boolean(
    'freeze_rl', False, 'Whether to only train inference and not rl'
)
flags.DEFINE_boolean(
    'freeze_inference', False, 'Whether to only train rl and not inference'
)
flags.DEFINE_integer(
    'num_demonstrations', None, 'Number of demonstrations to collect per skill'
)
flags.DEFINE_boolean(
    'cloning_loss', False, 'Whether to include cloning to demonstrations'
)
flags.DEFINE_boolean(
    'irl_loss', False, 'Whether to include irl to demonstrations'
)
flags.DEFINE_boolean(
    'cloning_only', False, 'Whether to perform behavior cloning'
)
flags.DEFINE_boolean(
    'cloning_only_sac', False, 'Whether to perform behavior cloning with SAC'
)
flags.DEFINE_float(
    'cloning_lambda', 1.0, 'How much to weight cloning loss'
)
flags.DEFINE_float(
    'irl_lambda', 1.0, 'How much to weight irl loss'
)
flags.DEFINE_integer(
    'num_demo_skills', None, 'How many skills to collect demos for'
)
flags.DEFINE_integer(
    'embedding_size', None, 'Size of state embedding for inference'
)
flags.DEFINE_boolean(
    'no_collect', False, 'Whether or not to collect new data'
)
flags.DEFINE_boolean(
    'on_policy', False, 'Whether data collection is on or off policy'
)
flags.DEFINE_integer(
    'collect_steps_per_iteration', 1, 'Steps to collect per iteration'
)
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
        

def merge_experience(off_policy_experience, demonstration_experience):
    step_type = tf.concat([off_policy_experience.step_type, demonstration_experience.step_type], axis=0)
    obs = tf.concat([off_policy_experience.observation["observation"], 
        demonstration_experience.observation["observation"]], axis=0)
    z = tf.concat([off_policy_experience.observation["z"], 
        demonstration_experience.observation["z"]], axis=0)
    observation = OrderedDict()
    observation["observation"] = obs
    observation["z"] = z
    action = tf.concat([off_policy_experience.action, demonstration_experience.action], axis=0)
    if off_policy_experience.policy_info is () and demonstration_experience.policy_info is ():
        policy_info = ()
    else:
        policy_info = tf.concat([off_policy_experience.policy_info, demonstration_experience.policy_info], axis=0)
    next_step_type = tf.concat([off_policy_experience.next_step_type, demonstration_experience.next_step_type], axis=0)
    reward = tf.concat([off_policy_experience.reward, demonstration_experience.reward], axis=0)
    discount = tf.concat([off_policy_experience.discount, demonstration_experience.discount], axis=0)
    return trajectory.Trajectory(step_type, observation, action, policy_info, next_step_type, reward, discount)

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


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
        env_load_fn=None,
        num_iterations=None,
        mask_xy=None,
        dim_z=None,
        categorical=None,
        recurrent=False,
        inference_fc_layers=(512, 512),
        actor_fc_layers=(512, 512),
        input_fc_layers=(256, 100),
        output_fc_layers=(256, 100),
        lstm_size=(100,),
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=(256, 256),
        num_parallel_environments=1,
        # Params for collect
        initial_collect_steps=None, 
        num_demonstrations=None,
        collect_steps_per_iteration=None,
        replay_buffer_capacity=None, 
        option_length=None,
        # Params for target update
        target_update_tau=0.005,
        target_update_period=1,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        inference_learning_rate=3e-4,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=0.99,
        reward_scale_factor=None,
        gradient_clipping=None,
        use_tf_functions=True,
        rl_steps=None,
        inference_steps=None,
        reinforce_all_actions=None,
        reinforce_average_action=None,
        # Params inference training
        inference_type=None,
        xy_prior=None,
        embedding_size=None,
        freeze_rl=None,
        freeze_inference=None,
        no_collect=None,
        on_policy=None,
        skill_regularization=None,
        cloning_loss=None,
        irl_loss=None,
        cloning_only=None,
        cloning_only_sac=None,
        cloning_lambda=None,
        irl_lambda=None,
        num_demo_skills=None,
        mock_inference=None,
        mock_rewards=None,
        partition_rewards=None,
        displacement_rewards=None,
        push_rewards=None,
        temperature=None,
        from_origin=None,
        manual_inference_noise=None,
        extrinsic_rewards=None,
        sac_baseline=None,
        learn_mask=None,
        # Params critic training
        clip_q_value=None,
        clip_q_to_range=None,
        twin_critic=None,
        # Params for eval
        num_eval_episodes=30,
        # Params for summaries and logging
        checkpoint_interval=10000,
        log_interval=1000,
        summary_interval=1000,
        video_interval = 10000000,
        render_interval = 1000,
        summaries_flush_secs=10,
        debug_summaries=True,
        summarize_grads_and_vars=True,
        ):
    """A simple train and eval for DIAYN on Mujoco.
    
    All hyperparameters come from the original DIAYN paper
    (https://arxiv.org/pdf/1801.01290.pdf).
    """
    # if env_name == "Plane-v1":
    #     reward_scale_factor = 1.0
    
    if num_demo_skills is None:
        num_demo_skills = dim_z

    plot_states = False

    if cloning_only or cloning_only_sac: cloning_loss = True

    inference = True
    if sac_baseline or mock_inference or mock_rewards or extrinsic_rewards \
        or partition_rewards or displacement_rewards or push_rewards:
       inference = False
    
    if reinforce_all_actions:
        train_steps_per_iteration = option_length

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

        if sac_baseline:
            _env_load_fn = env_load_fn
        else:
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
     
        if sac_baseline:
            time_step_spec = tf_env.time_step_spec()
            observation_spec = time_step_spec.observation
            action_spec = tf_env.action_spec()
        else:
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

            if tensor_spec.is_discrete(z_spec):
                _preprocessing_combiner = OneHotConcatenateLayer(dim_z)
            else:
                _preprocessing_combiner = DictConcatenateLayer()

            if inference_type == "reward":
                input_inference_spec = reward_spec
            else:
                if inference_type == "diayn":
                    option_length = 2
                if xy_prior:
                    input_inference_spec = tspec.BoundedTensorSpec(
                        shape=[2],
                        dtype=tf.float64,
                        minimum=-1.79769313e+308,
                        maximum=1.79769313e+308,
                        name='body_com')
                else:
                    input_inference_spec = observation_spec
        
        if sac_baseline:
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                observation_spec,
                action_spec,
                fc_layer_params=actor_fc_layers,
                continuous_projection_net=normal_projection_net)
            critic_net = critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=critic_obs_fc_layers,
                action_fc_layer_params=critic_action_fc_layers,
                joint_fc_layer_params=critic_joint_fc_layers)
            teacher_net = None
        else:
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=augmented_observation_spec,
                output_tensor_spec=action_spec,
                preprocessing_combiner=_preprocessing_combiner,
                fc_layer_params=actor_fc_layers,
                continuous_projection_net=normal_projection_net,
                mask_xy=mask_xy,)
            critic_net = critic_network.CriticNetwork(
                (augmented_observation_spec, action_spec),
                preprocessing_combiner=_preprocessing_combiner,
                observation_fc_layer_params=critic_obs_fc_layers,
                action_fc_layer_params=critic_action_fc_layers,
                joint_fc_layer_params=critic_joint_fc_layers,
                mask_xy=mask_xy,)
            if twin_critic:
                critic_net_2 = critic_network.CriticNetwork(
                    (augmented_observation_spec, action_spec),
                    preprocessing_combiner=_preprocessing_combiner,
                    observation_fc_layer_params=critic_obs_fc_layers,
                    action_fc_layer_params=critic_action_fc_layers,
                    joint_fc_layer_params=critic_joint_fc_layers)
            else:
                critic_net_2 = None
            if teacher_dir is not None:
                teacher_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=augmented_observation_spec,
                output_tensor_spec=action_spec,
                preprocessing_combiner=_preprocessing_combiner,
                fc_layer_params=actor_fc_layers,
                continuous_projection_net=normal_projection_net,
                mask_xy=False,)
            else:
                teacher_net = None

        network_state = ()
        if inference:
            if embedding_size:
                embedding_spec = tspec.BoundedTensorSpec(
                        shape=[embedding_size],
                        dtype=tf.float32,
                        minimum=-100,
                        maximum=100,
                        name='embedding')
                embedding_net = actor_distribution_network.ActorDistributionNetwork(
                    input_tensor_spec=input_inference_spec,
                    output_tensor_spec=embedding_spec,
                    fc_layer_params=inference_fc_layers,
                    continuous_projection_net=normal_projection_net)
                inference_net = actor_distribution_network.ActorDistributionNetwork(
                    input_tensor_spec=embedding_spec,
                    output_tensor_spec=z_spec,
                    fc_layer_params=inference_fc_layers,
                    continuous_projection_net=normal_projection_net)
            elif recurrent:
                inference_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                    input_tensor_spec=input_inference_spec,
                    output_tensor_spec=z_spec,
                    input_fc_layer_params=input_fc_layers,
                    output_fc_layer_params=output_fc_layers,
                    lstm_size=lstm_size,
                    continuous_projection_net=normal_projection_net)
                
                network_state = actor_policy.ActorPolicy(time_step_spec, z_spec,
                                                inference_net).get_initial_state(batch_size=batch_size)
                embedding_net = None
            else:
                inference_net = actor_distribution_network.ActorDistributionNetwork(
                    input_tensor_spec=input_inference_spec,
                    output_tensor_spec=z_spec,
                    fc_layer_params=inference_fc_layers,
                    continuous_projection_net=normal_projection_net)
                embedding_net = None
        else:
            inference_net = None
            embedding_net = None

        actor_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_actor'),
            actor_net=actor_net,
            global_step=global_step
        )
        actor_checkpointer.initialize_or_restore()

        critic_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_critic'),
            critic_net=critic_net,
            global_step=global_step
        )
        critic_checkpointer.initialize_or_restore()

        if twin_critic:
            critic_2_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(root_dir, 'diayn_critic_2'),
                critic_net=critic_net_2,
                global_step=global_step
            )
            critic_2_checkpointer.initialize_or_restore()
 
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

        if inference:
            inference_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(root_dir, 'diayn_inference'),
                inference_net=inference_net,
                global_step=global_step
            )
            inference_checkpointer.initialize_or_restore()
            if embedding_size:
                embedding_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(root_dir, 'diayn_embedding'),
                embedding_net=embedding_net,
                global_step=global_step
                )
                embedding_checkpointer.initialize_or_restore()

        env_steps = tf_metrics.EnvironmentSteps(prefix='Train')
        average_return = tf_metrics.AverageReturnMetric(
            prefix='Train',
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size)
        train_metrics = [
            tf_metrics.NumberOfEpisodes(prefix='Train'),
            env_steps,
            average_return,
            tf_metrics.AverageEpisodeLengthMetric(
                prefix='Train',
                buffer_size=num_eval_episodes,
                batch_size=tf_env.batch_size),
        ]

        if sac_baseline:
            tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=alpha_learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=td_errors_loss_fn,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                gradient_clipping=gradient_clipping,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars,
                train_step_counter=global_step)
        else:
            tf_agent = diayn_agent.DiaynAgent(
                augmented_time_step_spec,
                action_spec,
                z_spec=z_spec,
                actor_policy_ctor=actor_policy.ActorPolicy,
                actor_network=actor_net,
                inference_network=inference_net,
                embedding_network=embedding_net,
                critic_network=critic_net,
                critic_network_2=critic_net_2,
                recurrent=recurrent,
                network_state=network_state,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=alpha_learning_rate),
                inference_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=inference_learning_rate),
                option_length=option_length,
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=td_errors_loss_fn,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                target_entropy=FLAGS.target_entropy,
                gradient_clipping=gradient_clipping,
                batch_size=batch_size,
                rl_steps=rl_steps,
                inference_steps=inference_steps,
                reinforce_all_actions=reinforce_all_actions,
                reinforce_average_action=reinforce_average_action,
                inference_type=inference_type,
                xy_prior=xy_prior,
                freeze_rl=freeze_rl,
                freeze_inference=freeze_inference,
                cloning_loss=cloning_loss,
                cloning_only=cloning_only,
                cloning_only_sac=cloning_only_sac,
                irl_loss=irl_loss,
                cloning_lambda=cloning_lambda,
                irl_lambda=irl_lambda,
                skill_regularization=skill_regularization,
                mock_inference = mock_inference,
                mock_rewards = mock_rewards,
                partition_rewards = partition_rewards,
                displacement_rewards = displacement_rewards,
                push_rewards = push_rewards,
                from_origin = from_origin,
                manual_inference_noise = manual_inference_noise,
                extrinsic_rewards = extrinsic_rewards,
                learn_mask = learn_mask,
                clip_q_value=clip_q_value,
                clip_q_to_range=clip_q_to_range,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars,
                train_step_counter=global_step)
        tf_agent.initialize()
      
        # Make the replay buffer.
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)
        replay_observer = [replay_buffer.add_batch]

        # Initialize collection and evaluation policies
        eval_policy = tf_agent.policy
        if teacher_net:
            initial_collect_policy = teacher_policy
        else:
            if sac_baseline:
                initial_collect_policy = random_tf_policy.RandomTFPolicy(
                    time_step_spec, tf_env.action_spec())
            else:
                initial_collect_policy = random_tf_policy.RandomTFPolicy(
                    augmented_time_step_spec, tf_env.action_spec())
        collect_policy = tf_agent.collect_policy

        # Initialize checkpointers
        train_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_train'),
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics,
                                                'train_metrics')
        )
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_policy'),
            policy=eval_policy,
            global_step=global_step)
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'diayn_replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)

        train_checkpointer.initialize_or_restore()
        policy_checkpointer.initialize_or_restore()
        rb_checkpointer.initialize_or_restore()

        if num_demonstrations is not None:
            initial_demonstration_driver = dynamic_demonstration_driver.DynamicDemonstrationDriver(
                tf_env, 
                initial_collect_policy,
                observers=replay_observer + train_metrics,
                num_episodes=num_demonstrations,
                num_skills=num_demo_skills)
        else:
            initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                tf_env,
                initial_collect_policy,
                observers=replay_observer + train_metrics,
                num_steps=initial_collect_steps)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=collect_steps_per_iteration)

        if use_tf_functions:
            if num_demonstrations is None:
                initial_collect_driver.run = common.function(
                    initial_collect_driver.run)
            else: 
                initial_demonstration_driver.run = common.function(
                    initial_demonstration_driver.run)
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)
        
        # Collect initial replay data.
        if replay_buffer.num_frames().numpy() == 0 and not on_policy:
            if num_demonstrations is not None:   
                logging.info(
                    'Initializing replay buffer by collecting experience for '
                    '%d demonstration(s) per skill with teacher policy.', num_demonstrations)
                initial_demonstration_driver.run()
            else:
                logging.info(
                    'Initializing replay buffer by collecting experience for '
                    '%d steps with a random policy.', initial_collect_steps)
                initial_collect_driver.run()
        
        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        time_acc = 0
        env_steps_before = env_steps.result().numpy()

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
        if cloning_loss or irl_loss:
            demonstration_replay_buffer = copy.deepcopy(replay_buffer)
            demonstration_dataset = replay_buffer.as_dataset(
                sample_batch_size=batch_size,
                num_steps=2).unbatch().filter(
                    _filter_invalid_transition).batch(batch_size).prefetch(5)
            demonstration_iterator = iter(demonstration_dataset)
        logging.info(
                'Training with reward_scaling = %d, option_length = %d, '
                'dim_z = %d, inference_type = %s', 
                reward_scale_factor, option_length, dim_z,
                inference_type)

        if plot_states:
            hist_x = plt.hist(replay_buffer.gather_all().observation["observation"][0, :, 0].numpy(), bins=10)
            plt.savefig(FLAGS.root_dir + "hist_x.png")
            plt.close()
            hist_y = plt.hist(replay_buffer.gather_all().observation["observation"][0, :, 1].numpy(), bins=10)
            plt.savefig(FLAGS.root_dir + "hist_y.png")
            raise NotImplementedError

        # Train evaluation loop
        def train_step():
            off_policy_experience, _ = next(iterator)
            if cloning_loss or irl_loss: 
                demonstration_experience, _ = next(demonstration_iterator)
                experience = merge_experience(off_policy_experience, demonstration_experience)
            else:
                experience = off_policy_experience
            return tf_agent.train(experience)

        if use_tf_functions:
            train_step = common.function(train_step)
       
        for _ in range(num_iterations):
            start_time = time.time()
            if on_policy:
                replay_buffer.clear()
            if not no_collect:
                time_step, policy_state = collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state,
                )
            for _ in range(train_steps_per_iteration):
                train_loss = train_step() 
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
                if sac_baseline:
                    logging.info('step = %d, actor_loss = %f, critic_loss=%f, alpha_loss=%f', 
                    global_step.numpy(), train_loss.extra.actor_loss, 
                    train_loss.extra.critic_loss, train_loss.extra.alpha_loss)
                else:
                    logging.info('step = %d, actor_loss = %f, critic_loss=%f, alpha_loss=%f, \
                    inference_loss=%f', global_step.numpy(), train_loss.extra.actor_loss, 
                    train_loss.extra.critic_loss, train_loss.extra.alpha_loss, train_loss.extra.inference_loss)
                logging.info('env steps = %d, average return = %f',
                             env_steps.result(),
                             average_return.result())
                env_steps_per_sec = (
                    env_steps.result().numpy() - env_steps_before) / time_acc
                logging.info('%.3f env steps/sec', env_steps_per_sec)
                tf.compat.v2.summary.scalar(
                    name='env_steps_per_sec',
                    data=env_steps_per_sec,
                    step=env_steps.result())
                time_acc = 0
                env_steps_before = env_steps.result().numpy()
            
            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step=env_steps.result())
            
            global_step_val = global_step.numpy()
            if global_step_val % checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0:
                policy_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0:
                rb_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0:
                actor_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0:
                critic_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0 and twin_critic:
                critic_2_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0 and inference:
                inference_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0 and inference and embedding_size:
                embedding_checkpointer.save(global_step=global_step_val)

            if env_name == 'Plane-v1':
                if global_step_val % render_interval == 0:
                    path_len = 10
                    num_eps = 1
                    for _ in range(num_eps):
                        _time_step = eval_py_env.reset()
                        eval_py_env.render()
                        steps = 0
                        while steps < path_len:
                            action_step = eval_policy.action(_time_step)
                            _time_step = eval_py_env.step(action_step.action.numpy())
                            eval_py_env.render()
                            steps += 1
            else:
                if global_step_val % video_interval == 0:
                    print('Creating video at {}k iterations...'.format(int(global_step_val/1000)))
                    video_path = root_dir + "/videos"
                    if not path.exists(video_path):
                        os.mkdir(video_path)
                    video_filename = FLAGS.root_dir + '/videos/lacma_' + env_name[:-3] + str(
                        int(global_step_val/1000)) + 'k.mp4'
                    path_len = 200
                    num_eps = 1
                    with imageio.get_writer(video_filename, fps=60) as video:
                        for _ in range(num_eps):
                            _time_step = eval_py_env.reset()
                            video.append_data(eval_py_env.render())
                            steps = 0
                            while steps < path_len:
                                action_step = eval_policy.action(_time_step)
                                _time_step = eval_py_env.step(action_step.action.numpy())
                                video.append_data(eval_py_env.render())
                                steps += 1
                        embed_mp4(video_filename)

def main(_):
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    if FLAGS.env == "ant":
        env_load_fn = suite_mujoco.load
        num_iterations = 3000000
        initial_collect_steps = 10000
        replay_buffer_capacity = 1000000
        env_name = 'AntSpace-v1'
    elif FLAGS.env == "plane":
        env_load_fn = suite_gym.load
        num_iterations = 100000
        initial_collect_steps = 100
        replay_buffer_capacity = 1000
        env_name = 'Plane-v1'
    else: # customize additional params
        env_load_fn = suite_mujoco.load
        num_iterations = 1000000
        initial_collect_steps = 10000
        replay_buffer_capacity = 1000000
        env_name = FLAGS.env_name

    train_eval(FLAGS.root_dir,
               teacher_dir = FLAGS.teacher_dir,
               env_name = env_name,
               env_load_fn = env_load_fn,
               dim_z = FLAGS.dim_z,
               mask_xy = FLAGS.mask_xy,
               categorical = FLAGS.categorical,
               num_iterations = num_iterations,
               initial_collect_steps = initial_collect_steps,
               num_demonstrations = FLAGS.num_demonstrations,
               replay_buffer_capacity = replay_buffer_capacity,
               option_length = FLAGS.option_length,
               reinforce_all_actions = FLAGS.reinforce_all_actions,
               reinforce_average_action = FLAGS.reinforce_average_action,
               rl_steps = FLAGS.rl_steps,
               inference_steps = FLAGS.inference_steps,
               inference_type = FLAGS.inference_type,
               xy_prior = FLAGS.xy_prior,
               embedding_size = FLAGS.embedding_size,
               freeze_rl = FLAGS.freeze_rl,
               freeze_inference = FLAGS.freeze_inference,
               no_collect = FLAGS.no_collect,
               on_policy = FLAGS.on_policy,
               collect_steps_per_iteration = FLAGS.collect_steps_per_iteration,
               skill_regularization = FLAGS.skill_regularization,
               cloning_loss = FLAGS.cloning_loss,
               cloning_only = FLAGS.cloning_only,
               cloning_only_sac = FLAGS.cloning_only_sac,
               irl_loss = FLAGS.irl_loss,
               cloning_lambda = FLAGS.cloning_lambda,
               irl_lambda = FLAGS.irl_lambda,
               num_demo_skills = FLAGS.num_demo_skills,
               mock_inference = FLAGS.mock_inference,
               mock_rewards = FLAGS.mock_rewards,
               partition_rewards = FLAGS.partition_rewards,
               displacement_rewards = FLAGS.displacement_rewards,
               push_rewards = FLAGS.push_rewards,
               temperature = FLAGS.temperature,
               reward_scale_factor = FLAGS.reward_scaling,
               gradient_clipping = FLAGS.gradient_clipping,
               from_origin = FLAGS.from_origin,
               manual_inference_noise = FLAGS.manual_inference_noise,
               extrinsic_rewards = FLAGS.extrinsic_rewards,
               sac_baseline = FLAGS.sac_baseline,
               learn_mask = FLAGS.learn_mask,
               clip_q_value = FLAGS.clip_q_value,
               clip_q_to_range = FLAGS.clip_q_to_range,
               twin_critic = FLAGS.twin_critic,
               debug_summaries=FLAGS.debug)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
