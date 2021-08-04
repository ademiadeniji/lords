import gym
from gym import spaces
import numpy as np

class PlaneEnv(gym.Env):
  """2D continuous plane Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(PlaneEnv, self).__init__()
    self.reward_range = (0, 10)
    self.action_space = spaces.Box(low=np.array([-10, -10]), high=np.array([10, 10]))
    self.observation_space = spaces.Box(low=np.array([-10000, -10000]), high=np.array([100, 100]))
    self.goal_pos = np.array([-50, 50])

  def step(self, action):
    posbefore = np.copy(self._obs)
    self._obs = self._obs + action
    posafter = self._obs
    obs = self._obs
    # reward = self._obs[0] - abs(self._obs[1])
    if np.maximum(np.abs(posafter - self.goal_pos)) < 5:
      reward = 1000
    else:
      reward = -1
    done = False
    return obs, reward, done, {}

  def reset(self):
    self._obs = np.array([0, 0])
    return self._obs
  
  def render(self, mode='human', close=False):
    print("{} {}".format(self._obs[0], self._obs[1]))