import argparse
import os
import time
import gymnasium as gym
import numpy as np
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.tensor import Tensor


# ----------------------------------------------------
# 1. Neural Network Architecture (Matches Checkpoint)
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
# 2. Main Inference Routine
# ----------------------------------------------------
def run_inference(
    checkpoint_path: str, episodes: int = 5, fps: int = 60, render: bool = True
):
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")

  # Initialize Gym Environment
  render_mode = "human" if render else None
  env = gym.make("LunarLander-v3", render_mode=render_mode)

  state_dim = int(env.observation_space.shape[0])
  action_dim = int(env.action_space.n)

  # Initialize model and load safetensors weights
  model = QNetwork(state_dim, action_dim)
  state_dict = safe_load(checkpoint_path)
  load_state_dict(model, state_dict)
  print(f"[+] Loaded weights from '{checkpoint_path}'")

  delay = 1.0 / fps if fps > 0 else 0.0

  for episode in range(1, episodes + 1):
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    max_pos = -1.2

    while not done:
      # Inference forward pass
      state_tensor = Tensor(state).reshape(1, -1)
      q_values = model(state_tensor).numpy()
      action = int(np.argmax(q_values))

      # Step environment
      state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      total_reward += reward
      steps += 1
      max_pos = max(max_pos, state[0])

      if render and delay > 0:
        time.sleep(delay)

    reached_flag = (
        "SUCCESS (Reached Goal!)" if state[0] >= 0.5 else "FAILED (Timed Out)"
    )
    print(
        f"Episode {episode:02d} | Result: {reached_flag} | Steps: {steps:3d} |"
        f" Reward: {total_reward:5.1f} | Max Pos: {max_pos:+.2f}"
    )

  env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Run inference on LunarLander-v3 with Tinygrad safetensors"
  )
  parser.add_argument(
      "--checkpoint",
      type=str,
      default="lunarlander_dqn.safetensors",
      help="Path to safetensors checkpoint",
  )
  parser.add_argument(
      "--episodes",
      type=int,
      default=5,
      help="Number of evaluation episodes to run",
  )
  parser.add_argument(
      "--fps",
      type=int,
      default=60,
      help="Target frames per second for rendering playback",
  )
  parser.add_argument(
      "--no-render",
      action="store_true",
      help="Disable graphical window rendering",
  )

  args = parser.parse_args()
  run_inference(
      checkpoint_path=args.checkpoint,
      episodes=args.episodes,
      fps=args.fps,
      render=not args.no_render,
  )
