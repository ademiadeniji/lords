r"""Finetune Pretrained Skills.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
import time
import pdb
import gym
import numpy as np

from absl import app
from absl import flags
from absl import logging

import base64
import IPython
import imageio

import gin
import tensorflow as tf
from tensorflow.python.framework import tensor_spec as ts
from collections import OrderedDict
import matplotlib.pyplot as plt

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import normal_projection_network
from tf_agents.policies import random_tf_policy
from tf_agents.policies import greedy_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.trajectories import trajectory

from tf_agents.agents.sac import sac_agent
from tf_agents.policies import actor_policy
from networks import actor_distribution_network
from networks import critic_network
from envs import skill_wrapper

from gym.wrappers import FlattenObservation
from gym.envs.robotics.fetch import reach
from gym.envs.robotics.fetch import jenga
from gym.envs.robotics.fetch import push



flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string(
    'gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string(
    'gin_param', None, 'Gin binding to pass through.')
flags.DEFINE_boolean(
    'debug', True, 'If printing debug summaries')
flags.DEFINE_boolean(
    'finetune_skills', True, 'If finetuning generator'
)
flags.DEFINE_float(
    'reward_scaling', 10.0, 'Amount to scale intrinsic reward by in agent.'
)
flags.DEFINE_integer(
    'dim_z', 32, 'number of discrete skills'
)
flags.DEFINE_string(
    'env_name', 'FetchReach-v1', 'Environment to use for HRL')
flags.DEFINE_boolean(
    'mask_xy', False, 'Whether to mask xy from policy')
FLAGS = flags.FLAGS

def _normal_projection_net(action_spec, init_means_output_factor=0.1):
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)

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

@gin.configurable  # NOQA
def train_eval(
        root_dir,
        env_name=None,
        eval_env_name=None,
        env_load_fn=suite_mujoco.load,
        num_iterations=100000,
        dim_z=None,
        categorical=True,
        mask_xy=None,
        finetune_skills=None,
        generator_fc_layers=(512, 512),
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=(512, 512),
        num_parallel_environments=1,
        # Params for collect
        initial_collect_steps=10000,
        collect_steps_per_iteration=1,
        replay_buffer_capacity=1000000,
        # Params for target update
        target_update_tau=0.005,
        target_update_period=1,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=0.99,
        reward_scale_factor=None,
        gradient_clipping=None,
        use_tf_functions=True,
        # Params for eval
        num_eval_episodes=30,
        eval_interval=10000,
        evaluate=True,
        # Params for summaries and logging
        video_interval=10000,
        checkpoint_interval=10000,
        log_interval=1000,
        summary_interval=1000,
        summaries_flush_secs=10,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):
    """A simple train and eval for LACMA on Mujoco.

    All hyperparameters come from the original LACMA paper
    (https://arxiv.org/pdf/1801.01290.pdf).
    """
    root_dir = os.path.expanduser(root_dir)
    if not path.exists(root_dir):
        os.mkdir(root_dir)

    video_path = root_dir + "/videos"
    if not path.exists(video_path):
        os.mkdir(video_path) 

    summary_writer = tf.compat.v2.summary.create_file_writer(
        root_dir, flush_millis=summaries_flush_secs * 1000)
    summary_writer.set_as_default()

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()

    def get_environment(env_name='FetchReach-v1'):
        global observation_omit_size
        if env_name == 'Ant-v1':
            env = ant.AntEnv(
                expose_all_qpos=True,
                task='motion')
            observation_omit_size = 0
        elif env_name == 'AntSpace-v1':
            env = ant_space.AntEnv(
                expose_all_qpos=True,
                task='motion')
            observation_omit_size = 0
        elif env_name == 'FetchPickAndPlace-v1':
            env = FlattenObservation(pick_and_place.FetchPickAndPlaceEnv())
            observation_omit_size = 0
        elif env_name == 'FetchPush-v1':
            env = FlattenObservation(push.FetchPushEnv())
            observation_omit_size = 0
        elif env_name == 'FetchReach-v1':
            env = FlattenObservation(reach.FetchReachEnv())
            observation_omit_size = 0
        elif env_name == 'FetchJenga-v1':
            env = FlattenObservation(jenga.FetchJengaEnv())
            observation_omit_size = 0
        elif env_name == 'FetchPickAndPlacePosServo-v1':
            env = FlattenObservation(pick_and_place_pos_servo.FetchPickAndPlacePosServoEnv())
            observation_omit_size = 0
        elif env_name == 'FetchPushPrimitives-v1':
            env = FlattenObservation(push_primitives.FetchPushPrimitivesEnv())
            observation_omit_size = 0
        elif env_name == "RobotPrimitives-v1":
            env = robot_primitives_env.RobotPrimitivesEnv()
            observation_omit_size = 0
        elif env_name == "Plane-v1":
            env = plane.PlaneEnv()
            observation_omit_size = 0
        else:
            # note this is already wrapped, no need to wrap again
            env = suite_mujoco.load(env_name)
        return env

    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        py_env = get_environment(env_name=FLAGS.env_name)
        py_env = wrap_env(
            skill_wrapper.SkillWrapper(
                py_env,
                num_latent_skills=FLAGS.dim_z,
                skill_type='discrete_uniform'),
                max_episode_steps=50)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

        time_step_spec = tf_env.time_step_spec()
        observation_spec = time_step_spec.observation
        action_spec = tf_env.action_spec()

        generator_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec, action_spec, fc_layer_params=generator_fc_layers,
            continuous_projection_net=_normal_projection_net)
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers)

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

        generator_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'actor_net'),
            actor_net=generator_net,
        )
        
        if generator_checkpointer.checkpoint_exists:
            generator_checkpointer.initialize_or_restore()
            logging.info("Low-level actor checkpoint exists...continuing paused finetuning")
        else:
            generator_checkpointer.initialize_or_restore()
            logging.info("No low-level actor checkpoint exists...training skills from scratch")
       
        critic_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'critic'),
            critic_net=critic_net,
            global_step=global_step
        )
        if critic_checkpointer.checkpoint_exists:
            critic_checkpointer.initialize_or_restore()
        else:
            critic_checkpointer.initialize_or_restore()
            logging.info("No critic checkpoint exists...training from scratch")
        
        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=generator_net,
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
        tf_agent.initialize()
        
        eval_policy = tf_agent.policy

        best_reward = float('-inf')
        best_skill = None
        logging.info("Determining best of pretrained skills to initialize with...")
        for skill in range(FLAGS.dim_z):
            print('Evaluating Skill #', skill)
            py_env.set_skill(skill)
            results = metric_utils.eager_compute(
                eval_metrics,
                tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
            )
            if eval_metrics_callback is not None:
                eval_metrics_callback(results, env_steps.result())
            metric_utils.log_metrics(eval_metrics)
            average_reward = results['AverageReturn'].numpy()
            if average_reward > best_reward:
                best_reward = average_reward
                best_skill = skill

            video_filename = FLAGS.root_dir + '/videos/finetuneskills' + str(skill) + env_name[:-3] + '.mp4'
            path_len = 50
            num_eps = 1
            with imageio.get_writer(video_filename, fps=60) as video:
                for _ in range(num_eps):
                    _time_step = py_env.reset()
                    video.append_data(py_env.render())
                    steps = 0
                    while steps < path_len:
                        action_step = eval_policy.action(_time_step)
                        _time_step = py_env.step(action_step.action.numpy())
                        # print(np.linalg.norm(_time_step.observation[5:8]))
                        video.append_data(py_env.render())
                        steps += 1
                embed_mp4(video_filename)
        print('Best Skill:', best_skill)
        print('Highest Reward:', best_reward)
        py_env.set_skill(best_skill)
        raise NotImplementedError
        
        # Make the replay buffer.
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)
        replay_observer = [replay_buffer.add_batch]

        collect_policy = tf_agent.collect_policy
        
        train_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'train'),
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics,
                                                'train_metrics')
        )
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)
        
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(root_dir, 'policy'),
            policy=tf_agent.policy,
            global_step=global_step)

        train_checkpointer.initialize_or_restore()
        rb_checkpointer.initialize_or_restore()
        policy_checkpointer.initialize_or_restore()

        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=initial_collect_steps)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=collect_steps_per_iteration)

        if use_tf_functions:
            initial_collect_driver.run = common.function(
                initial_collect_driver.run)
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        # Collect initial replay data.
        if replay_buffer.num_frames() == 0:
            logging.info(
                'Initializing replay buffer by collecting experience for '
                '%d steps with policy.', initial_collect_steps)
            initial_collect_driver.run()
        
        results = metric_utils.eager_compute(
            eval_metrics,
            tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=env_steps.result(),
            summary_writer=summary_writer,
            summary_prefix='Eval',
        )
     
        if eval_metrics_callback is not None:
            eval_metrics_callback(results, env_steps.result())
        metric_utils.log_metrics(eval_metrics)
        
        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        time_acc = 0
        env_steps_before = env_steps.result().numpy()

        # Dataset generates trajectories with shape [Bx2x...]
        def _filter_invalid_transition(trajectories, unused_arg1):
            return tf.reduce_all(~trajectories.is_boundary()[:-1]) and \
            tf.reduce_all(~trajectories.is_first()[1:])

        dataset = replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=2).unbatch().filter(
                _filter_invalid_transition).batch(batch_size).prefetch(5)
        iterator = iter(dataset)

        logging.info("Training...")
        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience)

        if use_tf_functions:
            train_step = common.function(train_step)

        for _ in range(num_iterations):
            start_time = time.time()
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                train_step()
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
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

            if global_step.numpy() % eval_interval == 0:
                results = metric_utils.eager_compute(
                    eval_metrics,
                    tf_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=env_steps.result(),
                    summary_writer=summary_writer,
                    summary_prefix='Eval',
                )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, env_steps.result())
                metric_utils.log_metrics(eval_metrics)

            global_step_val = global_step.numpy()
            if global_step_val % checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0:
                policy_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0:
                rb_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0:
                generator_checkpointer.save(global_step=global_step_val)
            if global_step_val % checkpoint_interval == 0:
                critic_checkpointer.save(global_step=global_step_val)              

            if global_step_val % video_interval == 0:
                logging.info('Creating video at {}k iterations...'.format(int(global_step_val/1000)))

                video_filename = FLAGS.root_dir + '/videos/' + env_name[:-3] + str(
                    int(global_step_val/1000)) + 'k.mp4'
            
                path_len = 50
                num_eps = 1
                with imageio.get_writer(video_filename, fps=60) as video:
                    for _ in range(num_eps):
                        _time_step = py_env.reset()
                        video.append_data(py_env.render())
                        steps = 0
                        while steps < path_len:
                            action_step = eval_policy.action(_time_step)
                            _time_step = py_env.step(action_step.action.numpy())
                            video.append_data(py_env.render())
                            steps += 1
                    embed_mp4(video_filename)

def main(_):
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_eval(FLAGS.root_dir,
               debug_summaries=FLAGS.debug,
               env_name=FLAGS.env_name,
               eval_env_name=FLAGS.env_name,
               dim_z=FLAGS.dim_z,
               mask_xy=FLAGS.mask_xy,
               finetune_skills=FLAGS.finetune_skills,
               reward_scale_factor=FLAGS.reward_scaling,)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
