# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful methods shared by all scripts."""

import os
import pickle
import typing
from typing import Any, Dict, Optional

from absl import logging
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
from sac import replay_buffer
from sac import wrappers
import torch
from torchkit import CheckpointManager
from torchkit.experiment import git_revision_hash
from xirl import common
import mani_skill
import mani_skill.envs.tasks as mani_envs
import mani_skill
import mani_skill.envs.tasks as mani_envs
import importlib


import yaml

# Global variable to store reference flattened size
_FLATTEN_REF_SIZE = None

def flatten_observation(obs):
    """Robust flattening of nested ManiSkill observations into a fixed-length float32 vector."""
    def extract_arrays(item):
        arrays = []

        if isinstance(item, np.ndarray):
            if item.dtype == np.object_:
                for sub_item in item.flat:
                    arrays.extend(extract_arrays(sub_item))
            elif np.issubdtype(item.dtype, (np.floating, np.integer)):
                arr = item.astype(np.float32).ravel()
                # Replace NaNs or infs
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                arrays.append(arr)

        elif isinstance(item, torch.Tensor):
            # Safely handle GPU tensors
            arr = item.detach().cpu().numpy().astype(np.float32).ravel()
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arrays.append(arr)

        elif isinstance(item, (list, tuple)):
            for sub_item in item:
                arrays.extend(extract_arrays(sub_item))

        elif isinstance(item, dict):
            for v in item.values():
                arrays.extend(extract_arrays(v))

        elif isinstance(item, (int, float, bool, np.number)):
            arrays.append(np.array([float(item)], dtype=np.float32))

        return arrays

    float32_arrays = extract_arrays(obs)

    if not float32_arrays:
        flat = np.zeros((1,), dtype=np.float32)
    else:
        flat = np.concatenate(float32_arrays, dtype=np.float32)

    # Enforce consistent size across timesteps
    global _FLATTEN_REF_SIZE
    if _FLATTEN_REF_SIZE is None:
        _FLATTEN_REF_SIZE = flat.shape[0]
        # print(f"[flatten_observation] reference size set to {_FLATTEN_REF_SIZE}")
    else:
        if flat.shape[0] != _FLATTEN_REF_SIZE:
            diff = _FLATTEN_REF_SIZE - flat.shape[0]
            if diff > 0:
                # Pad missing elements with zeros
                flat = np.pad(flat, (0, diff))
                # print(f"[flatten_observation] padded +{diff} to maintain size {_FLATTEN_REF_SIZE}")
            elif diff < 0:
                # Truncate extras (shouldn't happen)
                flat = flat[:_FLATTEN_REF_SIZE]
                # print(f"[flatten_observation] truncated {abs(diff)} to maintain size {_FLATTEN_REF_SIZE}")

    return flat
  
# pylint: disable=logging-fstring-interpolation

ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict

# ========================================= #
# Experiment utils.
# ========================================= #


def setup_experiment(exp_dir, config, resume = False):
  """Initializes a pretraining or RL experiment."""
  #  If the experiment directory doesn't exist yet, creates it and dumps the
  # config dict as a yaml file and git hash as a text file.
  # If it exists already, raises a ValueError to prevent overwriting
  # unless resume is set to True.
  if os.path.exists(exp_dir):
    if not resume:
      raise ValueError(
          "Experiment already exists. Run with --resume to continue.")
    load_config_from_dir(exp_dir, config)
  else:
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
      yaml.dump(ConfigDict.to_dict(config), fp)
    with open(os.path.join(exp_dir, "git_hash.txt"), "w") as fp:
      fp.write(git_revision_hash())


def load_config_from_dir(
    exp_dir,
    config = None,
):
  """Load experiment config."""
  with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)
  # Inplace update the config if one is provided.
  if config is not None:
    config.update(cfg)
    return
  return ConfigDict(cfg)


def dump_config(exp_dir, config):
  """Dump config to disk."""
  # Note: No need to explicitly delete the previous config file as "w" will
  # overwrite the file if it already exists.
  with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
    yaml.dump(ConfigDict.to_dict(config), fp)


def copy_config_and_replace(
    config,
    update_dict = None,
    freeze = False,
):
  """Makes a copy of a config and optionally updates its values."""
  # Using the ConfigDict constructor leaves the `FieldReferences` untouched
  # unlike `ConfigDict.copy_and_resolve_references`.
  new_config = ConfigDict(config)
  if update_dict is not None:
    new_config.update(update_dict)
  if freeze:
    return FrozenConfigDict(new_config)
  return new_config


def load_model_checkpoint(pretrained_path, device):
  """Load a pretrained model and optionally a precomputed goal embedding."""
  config = load_config_from_dir(pretrained_path)
  model = common.get_model(config)
  model.to(device).eval()
  checkpoint_dir = os.path.join(pretrained_path, "checkpoints")
  checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
  global_step = checkpoint_manager.restore_or_initialize()
  logging.info("Restored model from checkpoint %d.", global_step)
  return config, model


def save_pickle(experiment_path, arr, name):
  """Save an array as a pickle file."""
  filename = os.path.join(experiment_path, name)
  with open(filename, "wb") as fp:
    pickle.dump(arr, fp)
  logging.info("Saved %s to %s", name, filename)


def load_pickle(pretrained_path, name):
  """Load a pickled array."""
  filename = os.path.join(pretrained_path, name)
  with open(filename, "rb") as fp:
    arr = pickle.load(fp)
  logging.info("Successfully loaded %s from %s", name, filename)
  return arr


# ========================================= #
# RL utils.
# ========================================= #


def make_env(
    env_name,
    seed,
    save_dir = None,
    add_episode_monitor = True,
    action_repeat = 1,
    frame_stack = 1,
):
  """Env factory with wrapping.

  Args:
    env_name: The name of the environment.
    seed: The RNG seed.
    save_dir: Specifiy a save directory to wrap with `VideoRecorder`.
    add_episode_monitor: Set to True to wrap with `EpisodeMonitor`.
    action_repeat: A value > 1 will wrap with `ActionRepeat`.
    frame_stack: A value > 1 will wrap with `FrameStack`.

  Returns:
    gym.Env object.
  """
  # Create ManiSkill environment 
    # This guarantees the environments (e.g. StackPyramid-v1) are registered with gym.
  try:
    importlib.import_module("mani_skill.envs.tasks.tabletop")
    print("Successfully imported mani_skill.envs.tasks.tabletop")
  except Exception:
    # If import fails, fall back to importing the top-level tasks package which
    # should still register most environments.
    print("Importing mani_skill.envs.tasks.tabletop failed, trying mani_skill.envs.tasks instead.")
    try:
      importlib.import_module("mani_skill.envs.tasks")
    except Exception:
      # If both imports fail, proceed and let gym.raise the appropriate error.
      pass
    
  env = gym.make(
    "StackPyramid-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pos", # there is also "pd_joint_delta_pos", ...
    render_mode="rgb_array"
    )

  if add_episode_monitor:
    env = wrappers.EpisodeMonitor(env)
  if action_repeat > 1:
    env = wrappers.ActionRepeat(env, action_repeat)
  # Temporarily disable RescaleAction to debug reset issue
  # env = RescaleAction(env, -1.0, 1.0)
  if save_dir is not None:
    env = wrappers.VideoRecorder(env, save_dir=save_dir)
  if frame_stack > 1:
    env = wrappers.FrameStack(env, frame_stack)

  # Seed.
#   env.seed(seed)
#   env.action_space.seed(seed)
#   env.observation_space.seed(seed)

  return env


def wrap_learned_reward(env, config, device):
  """Wrap the environment with a learned reward wrapper.

  Args:
    env: A `gym.Env` to wrap with a `LearnedVisualRewardWrapper` wrapper.
    config: RL config dict, must inherit from base config defined in
      `configs/rl_default.py`.

  Returns:
    gym.Env object.
  """
  print("Wrapping environment with learned reward wrapper...")
  pretrained_path = config.reward_wrapper.pretrained_path
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_config, model = load_model_checkpoint(pretrained_path, device)
  
  if config.reward_wrapper.type == "reds":
    print("Model loaded")
    model.load_state_dict(torch.load(
        os.path.join(pretrained_path, "reds_model.pth"),
        map_location=device,
    ))
    model.to(device).eval()

  kwargs = {
      "env": env,
      "model": model,
      "device": device,
      "res_hw": model_config.data_augmentation.image_size,
  }
  

  if config.reward_wrapper.type == "goal_classifier":
    env = wrappers.GoalClassifierLearnedVisualReward(**kwargs)

  elif config.reward_wrapper.type == "distance_to_goal":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.DistanceToGoalLearnedVisualReward(**kwargs)
    
  elif config.reward_wrapper.type == "inest":
    all_means = load_pickle(pretrained_path, "subtask_means.pkl")
    selected_means_2_4_6 = [all_means[i] for i in [1]]  # elements 2,4,6
    kwargs["subtask_means"] = selected_means_2_4_6
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.INESTIRLLearnedVisualReward(**kwargs)
    
  elif config.reward_wrapper.type == "inest_knn":
    all_means = load_pickle(pretrained_path, "subtask_means.pkl")
    selected_means_2_4_6 = [all_means[i] for i in [0,1]]  # elements 2,4,6
    kwargs["subtask_means"] = selected_means_2_4_6
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.KNNINESTIRLLearnedVisualReward(**kwargs)
    
  elif config.reward_wrapper.type == "state_intrinsic":
    all_means = load_pickle(pretrained_path, "subtask_means.pkl")
    selected_means_2_4_6 = [all_means[i] for i in [0,1]]  # elements 2,4,6
    kwargs["subtask_means"] = selected_means_2_4_6
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.STATEINTRINSICLearnedVisualReward(**kwargs)
    
  elif config.reward_wrapper.type == "reds":
    print("AAAAAAA")
    env = wrappers.REDSLearnedVisualReward(**kwargs)

  else:
    raise ValueError(
        f"{config.reward_wrapper.type} is not a valid reward wrapper.")

  return env




def make_buffer(
    env,
    device,
    config,
    flattened_obs_shape=None,
    flattened_next_obs_shape=None,
):
  """Replay buffer factory.

  Args:
    env: A `gym.Env`.
    device: A `torch.device` object.
    config: RL config dict, must inherit from base config defined in
      `configs/rl_default.py`.
    flattened_obs_shape: Optional tuple specifying the shape of flattened observations.
      If provided, this will be used instead of env.observation_space.shape.

  Returns:
    ReplayBuffer.
  """
  # Use flattened observation shape if provided, otherwise use environment's shape
  obs_shape = flattened_obs_shape if flattened_obs_shape is not None else env.observation_space.shape
  next_obs_shape = flattened_next_obs_shape if flattened_next_obs_shape is not None else env.observation_space.shape

  kwargs = {
      "obs_shape": obs_shape,
      "next_obs_shape": next_obs_shape,
      "action_shape": env.action_space.shape,
      "capacity": config.replay_buffer_capacity,
      "device": device,
  }

  buffer = replay_buffer.ReplayBuffer(**kwargs)
  return buffer


# ========================================= #
# Misc. utils.
# ========================================= #


def plot_reward(rews):
  """Plot raw and cumulative rewards over an episode."""
  _, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
  axes[0].plot(rews)
  axes[0].set_xlabel("Timestep")
  axes[0].set_ylabel("Reward")
  axes[1].plot(np.cumsum(rews))
  axes[1].set_xlabel("Timestep")
  axes[1].set_ylabel("Cumulative Reward")
  for ax in axes:
    ax.grid(visible=True, which="major", linestyle="-")
    ax.grid(visible=True, which="minor", linestyle="-", alpha=0.2)
  plt.minorticks_on()
  plt.show()


# ========================================= #
# Vector Environment utils for DDP.
# ========================================= #

def make_vector_env(
    env_name,
    num_envs,
    seed_start,
    save_dir = None,
    add_episode_monitor = True,
    action_repeat = 1,
    frame_stack = 1,
):
  """Create synchronized vector environment for DDP training.
  
  Args:
    env_name: The name of the environment.
    num_envs: Number of parallel environments to create.
    seed_start: Starting seed (will be incremented for each env).
    save_dir: Specify a save directory to wrap with `VideoRecorder`.
    add_episode_monitor: Set to True to wrap with `EpisodeMonitor`.
    action_repeat: A value > 1 will wrap with `ActionRepeat`.
    frame_stack: A value > 1 will wrap with `FrameStack`.
    
  Returns:
    gym.vector.SyncVectorEnv object.
  """
  from gym.vector import SyncVectorEnv
  
  def _make_env(rank):
    def _init():
      env_seed = seed_start + rank
      return make_env(
          env_name=env_name,
          seed=env_seed,
          save_dir=save_dir if rank == 0 else None,  # Only first env saves videos
          add_episode_monitor=add_episode_monitor,
          action_repeat=action_repeat,
          frame_stack=frame_stack,
      )
    return _init
  
  # Create vector environment
  env_fns = [_make_env(i) for i in range(num_envs)]
  venv = SyncVectorEnv(env_fns)
  
  return venv

def wrap_vector_learned_reward(venv, config, device):
  """Wrap vector environment with learned reward.
  
  Args:
    venv: A vector environment.
    config: RL config dict.
    device: Torch device.
    
  Returns:
    Wrapped vector environment.
  """
  # For vector environments, we need to wrap each individual environment
  print("Wrapping vector environment with learned reward...")
  
  pretrained_path = config.reward_wrapper.pretrained_path
  model_config, model = load_model_checkpoint(pretrained_path, device)
  
  if config.reward_wrapper.type == "reds":
    model.load_state_dict(torch.load(
        os.path.join(pretrained_path, "reds_model.pth"),
        map_location=device,
    ))
    model.to(device).eval()

  # Apply the wrapper to each individual environment
  for i, env in enumerate(venv.envs):
    venv.envs[i] = wrap_learned_reward_single(env, config, device, model, model_config)
  
  return venv

def wrap_learned_reward_single(env, config, device, model, model_config):
  """Wrap a single environment with learned reward."""
  kwargs = {
      "env": env,
      "model": model,
      "device": device,
      "res_hw": model_config.data_augmentation.image_size,
  }
  
  if config.reward_wrapper.type == "goal_classifier":
    from sac.wrappers import GoalClassifierLearnedVisualReward
    return GoalClassifierLearnedVisualReward(**kwargs)
  elif config.reward_wrapper.type == "distance_to_goal":
    kwargs["goal_emb"] = load_pickle(config.reward_wrapper.pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(config.reward_wrapper.pretrained_path, "distance_scale.pkl")
    from sac.wrappers import DistanceToGoalLearnedVisualReward
    return DistanceToGoalLearnedVisualReward(**kwargs)
  elif config.reward_wrapper.type == "holdr":
    kwargs["subtask_means"] = load_pickle(config.reward_wrapper.pretrained_path, "subtask_means.pkl")
    kwargs["distance_scale"] = load_pickle(config.reward_wrapper.pretrained_path, "distance_scale.pkl")
    from sac.wrappers import HOLDRLearnedVisualReward
    return HOLDRLearnedVisualReward(**kwargs)
  elif config.reward_wrapper.type == "reds":
    from sac.wrappers import REDSLearnedVisualReward
    return REDSLearnedVisualReward(**kwargs)
  else:
    return env

def make_vect_buffer(env, device, config):
    """Replay buffer factory.

    Args:
      env: A `gym.Env`.
      device: A `torch.device` object.
      config: RL config dict, must inherit from base config defined in
        `configs/rl_default.py`.

    Returns:
      ReplayBuffer.
    """
    # Handle both single and vector environments for action_shape
    if hasattr(env, 'single_action_space'):
      # Vector environment
      action_shape = env.single_action_space.shape
      obs_shape = env.single_observation_space.shape
    elif isinstance(env.action_space, (tuple, gym.spaces.Tuple)):
      # Vector environment with tuple action space
      action_shape = env.action_space[0].shape
      obs_shape = env.observation_space.shape
    else:
      # Single environment
      action_shape = env.action_space.shape
      obs_shape = env.observation_space.shape

    kwargs = {
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "capacity": config.replay_buffer_capacity,
        "device": device,
    }

    # Use standard replay buffer
    return replay_buffer.ReplayBuffer(**kwargs)
