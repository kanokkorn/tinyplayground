import os
import time
import random
from collections import deque
from typing import Tuple

from tinygrad import Tensor, nn, TinyJit
from tinygrad import Device
from tinygrad.helpers import trange
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

import gymnasium as gym

# hyperparams
ENVIRONMENT_NAME = 'CartPole-v1'

BATCH_SIZE = 256
ENTROPY_SCALE = 0.0005
REPLAY_BUFFER_SIZE = 10000
PPO_EPSILON = 0.2
HIDDEN_UNITS = 32
LEARNING_RATE = 1e-2
TRAIN_STEPS = 5
EPISODES = 1000
DISCOUNT_FACTOR = 0.99


class DQN:
  def __init__(self, state_dim, action_dim, hidden_dim=256):
    self.l1 = nn.Linear(state_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, action_dim)

  def __call__(self, x):
    x = self.l1(x).relu()
    return self.l2(x)

class ExperienceReplay:
  def __init__(self, maxlen, seed=None):
    self.memory = deque([], maxlen=maxlen)
    if seed is not None:
      random.seed(seed)

  def append(self, transition):
    self.memory.append(transition)

  def sample(self, sample_size):
    return rando.sample(self.memory, sample_size)

  def __len__(self):
    return len(self.memory)

class Agent:
  def run(self, is_training=True, render=False):
    env = gym.make(ENVIRONMENT_NAME)
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n
    policy_dqn = DQN(num_state, num_action)
    state_dict = get_state_dict(model)
    reward_per_episode = []
    try:
      state_dict = safe_load('checkpoint.safetensors')
      load_state_dict(model, state_dict)
      print("checkpoint loaded..")
    except FileNotFoundError:
      print("checkpoint not found, creating new one..")
      safe_save(state_dict, 'checkpoint.safetensors')
    if is_training:
      memory = ExperienceReplay(REPLAY_BUFFER_SIZE)
      policy_dqn = DQN(num_state, num_action)
      state_dict = get_state_dict(policy_dqn)
      try:
        state_dict = safe_load('checkpoint.safetensors')
        load_state_dict(model, state_dict)
        print("checkpoint loaded..")
      except FileNotFoundError:
        print("checkpoint not found, creating new one..")
        safe_save(state_dict, 'checkpoint.safetensors')
    for episode in range(EPISODES):
      obs, _ = env.reset()
      terminated = False
      episode_reward = 0.0
      while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        episode_reward
        if is_training:
          memory.append((state, action, reward, terminated))
        state = obs
      reward_per_episode.append(episode_reward)
    env.close()

@TinyJit
def jit(x):
  return net(x).realize()

def intialize_env():
  env = gym.make(ENVIRONMENT_NAME)
  env.action_space
  env.observation_space
  print(f"{ENVIRONMENT_NAME} have {env.action_space.n} action(s) and {env.observation_space} observe(s)")
  print(f"Running on device {Device.DEFAULT}")


if __name__ == "__main__":
  intialize_env()
