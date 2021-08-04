import numpy as np
import random
from .metaenv import MetaEnv
# from meta_policy_search.utils import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
import pdb

class AntRandGoalEvalEnv(MetaEnv, gym.utils.EzPickle, MujocoEnv):
    def __init__(self):
        self.n_tasks = 1
        self.tasks = self.sample_tasks(self.n_tasks)
        self.tasks[0] = np.array([0, -10])
        self.set_task(self.tasks[0])
        MujocoEnv.__init__(self, 'ant.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        a = np.random.random(n_tasks) * 2 * np.pi
        r = 3 * np.random.random(n_tasks) ** 0.5
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_pos = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_pos

    def step(self, a):
        xposbefore = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")
        goal_reward = (np.linalg.norm(xposbefore[:2] - self.goal_pos)**2 - np.linalg.norm(xposafter[:2] - self.goal_pos)**2) 
        goal_reward = goal_reward + np.exp(-np.linalg.norm(xposbefore[:2] - self.goal_pos)) * 10
        ctrl_cost = .001 * np.square(a).sum()
        contact_cost = 0.005 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

if __name__ == "__main__":
    env = AntRandGoalEvalEnv()
    while True:
        env.reset()
        for _ in range(100):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action