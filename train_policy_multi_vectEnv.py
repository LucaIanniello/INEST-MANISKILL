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

"""Launch script for training RL policies with pretrained reward models."""


# Only import non-CUDA, non-torch, non-agent, non-utils modules at the top
import collections
import os.path as osp
from typing import Dict

from absl import app
from absl import flags
from absl import logging
from base_configs import validate_config
import gym
from ml_collections import config_dict
from ml_collections import config_flags
import numpy as np
import os

# pylint: disable=logging-fstring-interpolation
FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_string("env_name", None, "The environment name.")
flags.DEFINE_integer("seed", 0, "RNG seed.")
flags.DEFINE_boolean("resume", False, "Resume experiment from last checkpoint.")
flags.DEFINE_boolean("wandb", False, "Log on W&B.")
flags.DEFINE_string("device", "cuda:0", "Compute device.")  # Add device flag

config_flags.DEFINE_config_file(
    "config",
    "base_configs/rl.py",
    "File path to the training hyperparameter configuration.",
)



# Will be re-imported inside main()
def evaluate(policy, eval_env, num_episodes):
  """Evaluate the policy and dump rollout videos to disk."""
  import numpy as np
  import collections
  episode_rewards = []
  policy.eval()
  stats = collections.defaultdict(list)
  
  # Handle both vector and single environments
  is_vector_env = hasattr(eval_env, 'num_envs')
  
  episodes_completed = 0
  if is_vector_env:
    observations = eval_env.reset()[0]
    # print(f"[EVAL DEBUG] After reset, observations shape: {observations.shape}, type: {type(observations)}", flush=True)
    # print(f"[EVAL DEBUG] After reset, observations[0] shape: {observations[0].shape}, type: {type(observations[0])}", flush=True)
    episode_rewards_current = np.zeros(eval_env.num_envs)
    
    while episodes_completed < num_episodes:
      # Debug: Print observation shapes for first few episodes
      if episodes_completed < 3:
        for j in range(min(eval_env.num_envs, 2)):  # Only print first 2 envs
          obs_shape = observations[j].shape if hasattr(observations[j], 'shape') else type(observations[j])
          print(f"[EVAL DEBUG] Episode {episodes_completed}, Env {j}: observation shape = {obs_shape}")
      
      actions = []
      for j in range(eval_env.num_envs):
        # Ensure observations[j] has the correct shape
        obs_j = observations[j]
        # print(f"[EVAL DEBUG] Before policy.act: obs_j shape = {obs_j.shape}, type = {type(obs_j)}", flush=True)
        if hasattr(policy, 'module'):  # DDP case
          action = policy.module.act(obs_j, sample=False)
        else:  # Non-DDP case
          action = policy.act(obs_j, sample=False)
        actions.append(action)
      
      next_observations, rewards, dones, infos = eval_env.step(actions)
      episode_rewards_current += rewards
      
      for j in range(eval_env.num_envs):
        if dones[j]:
          if episodes_completed < num_episodes:
            episode_rewards.append(episode_rewards_current[j])
            episodes_completed += 1
            
            if j < len(infos) and "episode" in infos[j]:
              for k, v in infos[j]["episode"].items():
                stats[k].append(v)
              if "eval_score" in infos[j]:
                stats["eval_score"].append(infos[j]["eval_score"])
          
          episode_rewards_current[j] = 0.0
      
      # Update observations for next iteration (key fix!)
      observations = next_observations
  else:
    # Single environment evaluation (fallback)
    for _ in range(num_episodes):
      observation, done = eval_env.reset(), False
      if "holdr" in FLAGS.experiment_name:
        eval_env.reset_state()
      episode_reward = 0
      while not done:
        if hasattr(policy, 'module'):  # DDP case
          action = policy.module.act(observation, sample=False)
        else:  # Non-DDP case
          action = policy.act(observation, sample=False)
        observation, reward, done, info = eval_env.step(action)
        episode_reward += reward
      for k, v in info["episode"].items():
        stats[k].append(v)
      if "eval_score" in info:
        stats["eval_score"].append(info["eval_score"])
      episode_rewards.append(episode_reward)
      
  for k, v in stats.items():
    stats[k] = np.mean(v)
  return stats, episode_rewards



def main(_):
  # DDP-safe: import all CUDA, torch, agent, utils, wandb, dist, etc. here
  import torch
  import torch.distributed as dist
  import os.path as osp
  from torchkit import CheckpointManager
  from torchkit import experiment
  from torchkit import Logger
  from tqdm.auto import tqdm
  import wandb
  import utils
  from sac import agent
  import os
  from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
  import sys
  import time
  pid = os.getpid()
  print(f"[DDP INIT] PID={pid} RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)

  # Make sure we have a valid config that inherits all the keys defined in the base config.
  if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
      rank = int(os.environ["RANK"])
      world_size = int(os.environ["WORLD_SIZE"])
      local_rank = int(os.environ.get("LOCAL_RANK", 0))
      dist.init_process_group(backend="nccl", init_method="env://")
      print(f"[DDP INIT] PID={pid} RANK={rank} LOCAL_RANK={local_rank} initializing device (torch.cuda.device_count()={torch.cuda.device_count()})", flush=True)
      torch.cuda.set_device(local_rank)
      device = torch.device(f"cuda:{local_rank}")
      print(f"[DDP INIT] PID={pid} RANK={rank} Set device to cuda:{local_rank} successfully.", flush=True)
  else:
      rank = 0
      world_size = 1
      device = torch.device(FLAGS.device if torch.cuda.is_available() else "cpu")
  print(f"[DDP TRAINING START] PID={pid} RANK={rank} DEVICE={device} Starting training loop.", flush=True)


  activated_subtask_experiment = False
  validate_config(FLAGS.config, mode="rl")

  config = FLAGS.config
  exp_dir = osp.join(
      config.save_dir,
      FLAGS.experiment_name,
      str(FLAGS.seed),
  )
  
  print(f"[DDP EXPERIMENT] PID={pid} RANK={rank} Using experiment name: {FLAGS.experiment_name}", flush=True)
  print(f"[DDP EXPERIMENT] PID={pid} RANK={rank} Experiment directory: {exp_dir}", flush=True)
  
  # Only rank 0 creates the experiment directory and handles setup
  if rank == 0:
    utils.setup_experiment(exp_dir, config, FLAGS.resume)
    
    if FLAGS.wandb:
      wandb.init(project="EnvRewardTests", group="20MillionMultiGPUENV", name="20MillionMultiGPUENV", mode="online")
      wandb.config.update(FLAGS)
      wandb.run.log_code(".")
      wandb.config.update(config.to_dict(), allow_val_change=True)
  
  # Synchronize all processes before continuing
  if world_size > 1:
    dist.barrier()

  # Setup compute device.
  # if torch.cuda.is_available():
  #   device = torch.device(FLAGS.device)
  # else:
  #   logging.info("No GPU device found. Falling back to CPU.")
  #   device = torch.device("cpu")
  logging.info("Using device: %s", device)

  # Set RNG seeds.
  if FLAGS.seed is not None:
    logging.info("RL experiment seed: %d", FLAGS.seed)
    experiment.seed_rngs(FLAGS.seed)
    experiment.set_cudnn(config.cudnn_deterministic, config.cudnn_benchmark)
  else:
    logging.info("No RNG seed has been set for this RL experiment.")


 
  # Load vector environments with different seeds for each process to ensure diversity
  num_envs_per_process = config.get("num_envs_per_process", 4)  # Number of parallel envs per DDP process
  env_seed_start = FLAGS.seed + rank * 1000  # Different seed range for each process
  eval_seed_start = FLAGS.seed + rank * 1000 + 500
  
  print(f"[DDP ENV] PID={pid} RANK={rank} Creating {num_envs_per_process} parallel environments per process", flush=True)
  
  env = utils.make_vector_env(
      FLAGS.env_name,
      num_envs=num_envs_per_process,
      seed_start=env_seed_start,
      action_repeat=config.action_repeat,
      frame_stack=config.frame_stack,
  )
  eval_env = utils.make_vector_env(
      FLAGS.env_name,
      num_envs=1,  # Keep eval simple with single env
      seed_start=eval_seed_start,
      action_repeat=config.action_repeat,
      frame_stack=config.frame_stack,
      save_dir=osp.join(exp_dir, "video", "eval") if rank == 0 else None,  # Only rank 0 saves videos
  )
  
  # if config.reward_wrapper.pretrained_path:
  #   print("Using learned reward wrapper for vector environments.")
  #   env = utils.wrap_vector_learned_reward(env, FLAGS.config, device)
  #   eval_env = utils.wrap_vector_learned_reward(eval_env, FLAGS.config, device)


  # Dynamically set observation and action space values from vector env
  config.sac.obs_dim = env.single_observation_space.shape[0]
  config.sac.action_dim = env.single_action_space.shape[0]
  config.sac.action_range = [
      float(env.single_action_space.low.min()),
      float(env.single_action_space.high.max()),
  ]

  # Resave the config since the dynamic values have been updated at this point
  # and make it immutable for safety :)
  utils.dump_config(exp_dir, config)
  config = config_dict.FrozenConfigDict(config)

  # policy = agent.SAC(device, config.sac)
  policy = agent.SAC(device, config.sac)
  if world_size > 1:
    policy = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank])

  buffer = utils.make_vect_buffer(env.envs[0], device, config)  # Use first env for buffer creation

  # # Create checkpoint manager.
  checkpoint_dir = osp.join(exp_dir, "checkpoints")
  # checkpoint_manager = CheckpointManager(
  #     checkpoint_dir,
  #     policy=policy,
  #     **policy.optim_dict(),
  # )

  # logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)
  
  if rank == 0:
    logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)
    # If using DDP, get the underlying model for optim_dict
    optim_dict = policy.module.optim_dict() if world_size > 1 else policy.optim_dict()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        policy=policy,
        **optim_dict,
    )
  else:
    logger = None
    checkpoint_manager = None

  try:
    if rank == 0:
      start = checkpoint_manager.restore_or_initialize()
    else:
      start = 0
    if world_size > 1:
      start_tensor = torch.tensor([start], device=device)
      dist.broadcast(start_tensor, src=0)
      start = start_tensor.item()
    
    observations = env.reset()[0]  # Vector env returns (obs, infos)
    episode_rewards = np.zeros(env.num_envs)
    LOG_EVERY_N = 1000  # Print rank/device info every N steps
    
    # Each process runs a fraction of total steps for proper DDP speedup
    steps_per_process = config.num_train_steps // world_size
    total_steps = start + steps_per_process
    
    print(f"[DDP TRAINING] PID={pid} RANK={rank} Running {steps_per_process} steps (total across all processes: {config.num_train_steps})", flush=True)
    
    for i in tqdm(range(start, total_steps), initial=start):
      if (i % LOG_EVERY_N == 0):
        print(f"[DDP STEP] PID={pid} RANK={rank} DEVICE={device} STEP={i}", flush=True)
        # Print memory usage for diagnostics
        try:
          import torch
          if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            print(f"[MEMORY] PID={pid} RANK={rank} GPU={device} Allocated={mem_alloc/1e6:.2f}MB Reserved={mem_reserved/1e6:.2f}MB", flush=True)
        except Exception as e:
          print(f"[MEMORY] PID={pid} RANK={rank} Could not get GPU memory: {e}", flush=True)
        try:
          import psutil
          process = psutil.Process(pid)
          rss = process.memory_info().rss / 1e6
          print(f"[MEMORY] PID={pid} RANK={rank} CPU RSS={rss:.2f}MB", flush=True)
        except Exception as e:
          print(f"[MEMORY] PID={pid} RANK={rank} Could not get CPU memory: {e}", flush=True)
      
      for subenv in env.envs:
        subenv.index_seed_steps = i
      # env._subtask = 1 # Reset subtask to 0 at the beginning of each step.
            
      # Subtask Exploration while in the beginning of the training.   
      
      # Block and free exploration
      # if i == 30_000 or i == 900_000 or i == 1_500_000:
      #   activated_subtask_experiment = True
          
      # if activated_subtask_experiment:
      #   if i >= 300_000 and i < 600_000:
      #       env._subtask = 1
      #   elif i >= 900_000 and i < 1_200_000:
      #       env._subtask = 2
      #   elif i >= 1_500_000 and i < 1_800_000:
      #       env._subtask = 3
      #   elif i == 600_000 or i == 1_200_000 or i == 1_800_000:
      #       activated_subtask_experiment = False
      #       env._subtask = 0
      #   else:
      #       env._subtask = 0
      
      # # ConsecutionBlocks      
      #       # if i == 30_000 : #   ac# tivated_subtask_experiment = True
          
      # if# if activated_subtask_experiment:
      #   for subenv in env.envs:
      #       if i >= 30_000 and i < 830_000:
      #           subenv.stage_completed[0] = True
      #           subenv.stage_completed[1] = True 
      #           subenv.stage_completed[2] = False
      #       elif i >= 830_000 and i < 1_630_000:
      #           subenv.stage_completed[0] = True
      #           subenv.stage_completed[1] = False
      #           subenv.stage_completed[2] = False
      #       elif i >= 1_630_000 and i < 2_400_000:
      #           subenv.stage_completed[0] = False
      #           subenv.stage_completed[1] = False 
      #           subenv.stage_completed[2] = False
      #       elif i == 2_400_000:
      #           activated_subtask_experiment = False
      #           subenv.stage_completed[0] = False
      #           subenv.stage_completed[1] = False 
      #           subenv.stage_completed[2] = False
      #       else:
      #           subenv.stage_completed[0] = False
      #           subenv.stage_completed[1] = False 
      #           subenv.stage_completed[2] = False   
          
      # Vector environment handling
      if i < config.num_seed_steps:
        # Random actions for initial exploration
        actions = [env.single_action_space.sample() for _ in range(env.num_envs)]
      else:
        policy.eval()
        # Get actions for all environments
        actions = []
        for j in range(env.num_envs):
          if world_size > 1:
            action = policy.module.act(observations[j], sample=True)  # DDP case
          else:
            action = policy.act(observations[j], sample=True)  # Non-DDP case
          actions.append(action)
      
      # Step all environments
      next_observations, rewards, dones, infos = env.step(actions)
      episode_rewards += rewards
      
      # Process each environment's transition
      for j in range(env.num_envs):
        observation = observations[j]
        action = actions[j]
        reward = rewards[j]
        next_observation = next_observations[j]
        done = dones[j]
        mask = 0.0 if done else 1.0
        
        # Log rewards for rank 0
        if rank == 0 and FLAGS.wandb and j == 0:  # Only log first env to avoid spam
          wandb.log({
            "train/reward": reward,
            "train/step": i,
          }, step=i)

        # Insert into replay buffer
        if not config.reward_wrapper.pretrained_path:
          buffer.insert(observation, action, reward, next_observation, mask)
        else:
          # For learned rewards, we need pixels from the single environment
          pixels = env.envs[j].render(mode="rgb_array") if hasattr(env.envs[j], 'render') else None
          if pixels is not None:
            buffer.insert(observation, action, reward, next_observation, mask, pixels)
          else:
            buffer.insert(observation, action, reward, next_observation, mask)
        
        # Handle episode completion
        if done:
          if "holdr" in config.reward_wrapper.type:
            buffer.reset_state()
            if hasattr(env.envs[j], 'reset_state'):
              env.envs[j].reset_state()

          # Log episode info for rank 0
          if rank == 0 and j < len(infos) and "episode" in infos[j]:
            for k, v in infos[j]["episode"].items():
              logger.log_scalar(v, infos[j]["total"]["timesteps"], k, "training")
              if FLAGS.wandb:
                wandb.log({
                    f"train_done/{k}": v,
                    "train_done/step": i,
                }, step=i)
            if FLAGS.wandb:
              wandb.log({
                  "train_done/episode_reward": episode_rewards[j],
                  "train_done/step": i,
              }, step=i)
          if world_size > 1:
            dist.barrier()
          
          # Reset episode reward for this environment
          episode_rewards[j] = 0.0
      
      # Update observations for next iteration
      observations = next_observations
      
      # For vector environments, adjust training frequency to maintain same sample efficiency
      # Train every N steps where N = num_envs to match single environment sample efficiency
      should_train = (i >= config.num_seed_steps) and ((i + 1) % env.num_envs == 0)
      
      if should_train:
        policy.train()
        # Handle both DDP and non-DDP cases for policy updates
        if world_size > 1:
          train_info = policy.module.update(buffer, i)  # DDP case
        else:
          train_info = policy.update(buffer, i)  # Non-DDP case

        if (i + 1) % config.log_frequency == 0 and rank == 0:
          for k, v in train_info.items():
            # Use step count for logging instead of info
            logger.log_scalar(v, i, k, "training")
            if FLAGS.wandb:
              wandb.log({
                  f"train/{k}": v,
                  "train/step": i,
              }, step=i)
          if FLAGS.wandb:
            wandb.log({
              "train/total_episode_rewards": np.sum(episode_rewards),
                "train/step": i,
            }, step=i)
          logger.flush()
        # Synchronize after logging to ensure all processes are aligned
        if world_size > 1:
          dist.barrier()

      
      if (i + 1) % config.eval_frequency == 0 and rank == 0:
        eval_stats, eval_episode_rewards = evaluate(policy, eval_env, config.num_eval_episodes)
        for k, v in eval_stats.items():
          logger.log_scalar(
              v,
              i,  # Use step count for logging
              f"average_{k}s",
              "evaluation",
          )
          if FLAGS.wandb:
            wandb.log({
                f"eval/{k}": v,
                "train/step": i,
            }, step=i)
          if FLAGS.wandb:
            wandb.log({
                "eval/episode_reward": eval_episode_rewards,
                "train/step": i,
            }, step=i)
        logger.flush()
        
      if world_size > 1:
          dist.barrier()

      if (i + 1) % config.checkpoint_frequency == 0 and rank == 0:
        print(f"[DDP CHECKPOINT] PID={pid} RANK={rank} Saving checkpoint at step {i}", flush=True)
        checkpoint_manager.save(i)

  except KeyboardInterrupt:
    print(f"[DDP EXIT] PID={pid} RANK={rank} Caught keyboard interrupt. Saving before quitting.", flush=True)

  finally:
    if rank == 0:
        print(f"[DDP EXIT] PID={pid} RANK={rank} Saving final checkpoint and closing logger.", flush=True)
        checkpoint_manager.save(i)
        logger.close()
    if world_size > 1:
        print(f"[DDP EXIT] PID={pid} RANK={rank} Destroying process group.", flush=True)
        dist.destroy_process_group()
    # checkpoint_manager.save(i)  # pylint: disable=undefined-loop-variable
    # logger.close()
  # NOTE: If you ever use a PyTorch DataLoader for offline RL or imitation, wrap it with DistributedSampler for DDP:
  # from torch.utils.data.distributed import DistributedSampler
  # train_sampler = DistributedSampler(dataset)  # Pass sampler=train_sampler to DataLoader


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_name")
  flags.mark_flag_as_required("env_name")
  app.run(main)
