import argparse
import collections
import os
import random
import gymnasium as gym
import numpy as np
from tinygrad import TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor

# ----------------------------------------------------
# 1. Neural Network Architecture
# ----------------------------------------------------
class QNetwork:

  def __init__(self, state_dim: int = 2, action_dim: int = 3):
    self.l1 = Tensor.glorot_uniform(state_dim, 64)
    self.b1 = Tensor.zeros(64)
    self.l2 = Tensor.glorot_uniform(64, 64)
    self.b2 = Tensor.zeros(64)
    self.l3 = Tensor.glorot_uniform(64, action_dim)
    self.b3 = Tensor.zeros(action_dim)

  def __call__(self, x: Tensor) -> Tensor:
    x = (x @ self.l1 + self.b1).relu()
    x = (x @ self.l2 + self.b2).relu()
    return x @ self.l3 + self.b3


# ----------------------------------------------------
# 2. Experience Replay Buffer
# ----------------------------------------------------
class ReplayBuffer:

  def __init__(self, capacity: int = 50_000):
    self.buffer = collections.deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size: int):
    batch = random.sample(self.buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int32),
        np.array(rewards, dtype=np.float32).reshape(-1, 1),
        np.array(next_states, dtype=np.float32),
        np.array(dones, dtype=np.float32).reshape(-1, 1),
    )

  def __len__(self):
    return len(self.buffer)


# ----------------------------------------------------
# 3. Checkpoint Helper Functions
# ----------------------------------------------------
def save_checkpoint(model: QNetwork, path: str = "mountaincar_dqn.safetensors"):
  state_dict = get_state_dict(model)
  safe_save(state_dict, path)
  print(f"[+] Saved model checkpoint to '{path}'")


def load_checkpoint(model: QNetwork, path: str = "mountaincar_dqn.safetensors"):
  if os.path.exists(path):
    state_dict = safe_load(path)
    load_state_dict(model, state_dict)
    print(f"[+] Loaded model checkpoint from '{path}'")
    return True
  print(f"[-] Checkpoint '{path}' not found.")
  return False


# ----------------------------------------------------
# 4. Evaluation / Test Loop
# ----------------------------------------------------
def evaluate_policy(
    model: QNetwork, episodes: int = 3, render: bool = True, slow_down: bool = True
):
  import time
  render_mode = "human" if render else None
  eval_env = gym.make("MountainCar-v0", render_mode=render_mode)

  for ep in range(1, episodes + 1):
    state, _ = eval_env.reset()
    total_reward = 0
    done = False

    while not done:
      q_vals = model(Tensor(state).reshape(1, -1)).numpy()
      action = int(np.argmax(q_vals))
      state, reward, terminated, truncated, _ = eval_env.step(action)
      done = terminated or truncated
      total_reward += reward
      if render and slow_down:
        time.sleep(0.01)

    print(f"Eval Episode {ep} - Total Reward: {total_reward}")

  eval_env.close()


# ----------------------------------------------------
# 5. Main Execution
# ----------------------------------------------------
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--test",
      action="store_true",
      help="Evaluate the saved model without training",
  )
  parser.add_argument(
      "--episodes", type=int, default=300, help="Number of training episodes"
  )
  parser.add_argument(
      "--checkpoint",
      type=str,
      default="mountaincar_dqn.safetensors",
      help="Checkpoint file path",
  )
  args = parser.parse_args()

  # Ensure types are pure Python integers
  env = gym.make("MountainCar-v0")
  state_dim = int(env.observation_space.shape[0])
  action_dim = int(env.action_space.n)

  q_net = QNetwork(state_dim, action_dim)
  target_net = QNetwork(state_dim, action_dim)

  # Try to resume from checkpoint if available
  load_checkpoint(q_net, args.checkpoint)

  # Sync initial target weights
  for p_tgt, p_src in zip(get_parameters(target_net), get_parameters(q_net)):
    p_tgt.assign(p_src.detach()).realize()

  if args.test:
    evaluate_policy(q_net, episodes=5, render=True)
    exit(0)

  optimizer = Adam(get_parameters(q_net), lr=1e-3)

  BATCH_SIZE = 64
  GAMMA = 0.99
  EPS_START = 0.5 if os.path.exists(args.checkpoint) else 1.0
  EPS_END = 0.02
  EPS_DECAY = 0.995
  TARGET_UPDATE_FREQ = 10

  Tensor.training = True
  @TinyJit
  def train_step(
      states: Tensor,
      actions: Tensor,
      rewards: Tensor,
      next_states: Tensor,
      dones: Tensor,
  ) -> Tensor:
    q_values = q_net(states)
    action_one_hot = actions.one_hot(action_dim)
    pred_q = (q_values * action_one_hot).sum(axis=1, keepdim=True)

    next_q = target_net(next_states).max(axis=1, keepdim=True).detach()
    target_q = rewards + (Tensor(1.0) - dones) * GAMMA * next_q

    loss = (pred_q - target_q).square().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.realize()

  replay_buffer = ReplayBuffer()
  eps = EPS_START
  best_reward = -float("inf")

  for episode in range(1, args.episodes + 1):
    state, _ = env.reset()
    total_reward = 0
    max_pos = -1.2

    while True:
      if random.random() < eps:
        action = env.action_space.sample()
      else:
        q_vals = q_net(Tensor(state).reshape(1, -1)).numpy()
        action = int(np.argmax(q_vals))

      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      # Potential-based reward shaping
      pos, vel = next_state
      height_reward = abs(pos + 0.5) ** 2
      shaped_reward = ( reward + height_reward * 5.0 + (100.0 if pos >= 0.5 else 0.0))
      max_pos = max(max_pos, pos)

      replay_buffer.push(state, action, shaped_reward, next_state, done)
      state = next_state
      total_reward += reward

      if len(replay_buffer) >= BATCH_SIZE:
        s_b, a_b, r_b, ns_b, d_b = replay_buffer.sample(BATCH_SIZE)
        train_step(
            Tensor(s_b), Tensor(a_b), Tensor(r_b), Tensor(ns_b), Tensor(d_b)
        )

      if done:
        break

    eps = max(EPS_END, eps * EPS_DECAY)

    if episode % TARGET_UPDATE_FREQ == 0:
      TAU = 0.01
      for p_tgt, p_src in zip(
          get_parameters(target_net), get_parameters(q_net)
      ):
          p_tgt.assign(TAU * p_src.detach() + (1.0 - TAU) * p_tgt).realize()

    print(
        f"    Episode {episode:03d} | Total Reward: {total_reward:4.0f} | Max Pos:"
        f" {max_pos:+.2f} | Eps: {eps:.2f}"
    )

    # Save checkpoint when reaching the goal consistently
    if total_reward > best_reward or episode % 50 == 0:
      best_reward = total_reward
      save_checkpoint(q_net, args.checkpoint)

  env.close()
