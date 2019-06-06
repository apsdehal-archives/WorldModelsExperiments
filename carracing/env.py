import numpy as np
import os
import gym

from scipy.misc import imresize as resize
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

SCREEN_X = 64
SCREEN_Y = 64

def _process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

class CarRacingWrapper(CarRacing):
  def __init__(self, full_episode=False, start_episode=0):
    super(CarRacingWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3)) # , dtype=np.uint8
    self.observations = np.zeros((1001, 64, 64, 3))
    self.actions = np.zeros((1000, 3))
    self.rewards = np.zeros((1000,))
    self.terminals = np.zeros((1000,), dtype=bool)
    self.iteration = 0
    self.episode = start_episode
    self.init = True

  def step(self, action):
    obs, reward, done, _ = super(CarRacingWrapper, self).step(action)

    if action is not None:
     self.rewards[self.iteration] = reward
     self.terminals[self.iteration] = done
     self.actions[self.iteration] = action

    if self.full_episode:
      obs = _process_frame(obs)
      if action is not None:
        self.observations[self.iteration + 1] = obs
        self.terminals[self.iteration] = False
        self.iteration += 1
      return obs, reward, False, {}

    obs =  _process_frame(obs)

    if action is not None:
      self.observations[self.iteration + 1] = obs
      self.iteration += 1

    if self.iteration >= 1000:
        done = True

    return obs, reward, done, {}

  def reset(self):
    if not self.init:
      print("Completed %d episode with %d iteration" % (self.episode, self.iteration))
      self.terminals[-1] = True
      os.makedirs('expert_rollouts', exist_ok=True)

      np.savez_compressed('expert_rollouts/rollout_%d.npz' % self.episode,
                          observations=self.observations,
                          actions=self.actions,
                          rewards=self.rewards,
                          terminals=self.terminals)
      self.episode += 1
      self.observations.fill(0)
      self.actions.fill(0)
      self.rewards.fill(0)
      self.terminals.fill(False)
      self.iteration = 0

    self.init = False
    obs = super(CarRacingWrapper, self).reset()
    self.observations[self.iteration] = obs

    return obs


def make_env(env_name, seed=-1, render_mode=False, full_episode=False, start_episode=0):
  env = CarRacingWrapper(full_episode=full_episode, start_episode=start_episode)
  if (seed >= 0):
    env.seed(seed)
  '''
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  assert False
  '''
  return env

# from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
if __name__=="__main__":
  from pyglet.window import key
  a = np.array( [0.0, 0.0, 0.0] )
  def key_press(k, mod):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  a[0] = -1.0
    if k==key.RIGHT: a[0] = +1.0
    if k==key.UP:    a[1] = +1.0
    if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
  def key_release(k, mod):
    if k==key.LEFT  and a[0]==-1.0: a[0] = 0
    if k==key.RIGHT and a[0]==+1.0: a[0] = 0
    if k==key.UP:    a[1] = 0
    if k==key.DOWN:  a[2] = 0
  env = CarRacing()
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release
  while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
      s, r, done, info = env.step(a)
      total_reward += r
      if steps % 200 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
      steps += 1
      env.render()
      if done or restart: break
  env.monitor.close()
