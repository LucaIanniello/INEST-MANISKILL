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
from sac import agent
import torch
from torchkit import CheckpointManager
from torchkit import experiment
from torchkit import Logger
from tqdm.auto import tqdm
import utils
import wandb

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_string("env_name", None, "The environment name.")
flags.DEFINE_integer("seed", 0, "RNG seed.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_boolean("resume", False, "Resume experiment from last checkpoint.")
flags.DEFINE_boolean("wandb", False, "Log on W&B.")

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


@experiment.pdb_fallback
def main(_):
  # Make sure we have a valid config that inherits all the keys defined in the
  # base config.
  activated_subtask_experiment = False
  validate_config(FLAGS.config, mode="rl")

  config = FLAGS.config
  exp_dir = osp.join(
      config.save_dir,
      FLAGS.experiment_name,
      str(FLAGS.seed),
  )
  utils.setup_experiment(exp_dir, config, FLAGS.resume)
  
  if FLAGS.wandb:
    wandb.init(project="EnvRewardTests", group="6MSingleGPUMultiEnvTEST", name="6MSingleGPUMultiEnvTEST", mode="online")
    wandb.config.update(FLAGS)
    wandb.run.log_code(".")
    wandb.config.update(config.to_dict(), allow_val_change=True)

  # Setup compute device.
  if torch.cuda.is_available():
    device = torch.device(FLAGS.device)
  else:
    logging.info("No GPU device found. Falling back to CPU.")
    device = torch.device("cpu")
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
  env_seed_start = FLAGS.seed + 1000  # Different seed range for each process
  eval_seed_start = FLAGS.seed + 1000 + 500
 
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
      save_dir=osp.join(exp_dir, "video", "eval") ,  # Only rank 0 saves videos
  )
  
  print("env:", env)
  print("env.action_space:", getattr(env, 'action_space', None))
  print("env.envs:", getattr(env, 'envs', None))
  for i, subenv in enumerate(getattr(env, 'envs', [])):
      print(f"Subenv {i}: {subenv}, action_space: {getattr(subenv, 'action_space', None)}")
  
  
  if config.reward_wrapper.pretrained_path:
    print("Using learned reward wrapper.")
    env = utils.wrap_learned_reward(env, FLAGS.config, device=device)
    eval_env = utils.wrap_learned_reward(eval_env, FLAGS.config, device=device)


  # Dynamically set observation and action space values.
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

  policy = agent.SAC(device, config.sac)

  buffer = utils.make_vect_buffer(env.envs[0], device, config)

  # Create checkpoint manager.
  checkpoint_dir = osp.join(exp_dir, "checkpoints")
  checkpoint_manager = CheckpointManager(
      checkpoint_dir,
      policy=policy,
      **policy.optim_dict(),
  )

  logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)

  try:
    start = checkpoint_manager.restore_or_initialize()
    observations = env.reset()[0]  # Vector env returns (obs, infos)
    episode_rewards = np.zeros(env.num_envs)
    
    # Debug: Print initial shapes and types
    print(f"[DEBUG] Initial observations shape: {observations.shape}")
    print(f"[DEBUG] Episode rewards shape: {episode_rewards.shape}")
    print(f"[DEBUG] Buffer capacity: {buffer.capacity}")
    print(f"[DEBUG] Number of environments: {env.num_envs}")
    print(f"[DEBUG] Training frequency adjusted: every {env.num_envs} steps")
    
    # Track learning statistics
    training_step_count = 0
    total_episodes_completed = 0
    
    for i in tqdm(range(start, config.num_train_steps), initial=start):
        
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
      # if i == 30_000:
      #   activated_subtask_experiment = True
          
      # if activated_subtask_experiment:
      #   if i >= 300_000 and i < 600_000:
      #       env._subtask = 1
      #   elif i >= 600_000 and i < 900_000:
      #       env._subtask = 2
      #   elif i >= 900_000 and i < 1_200_000:
      #       env._subtask = 3
      #   elif i == 1_200_000:
      #       activated_subtask_experiment = False
      #       env._subtask = 0
      #   else:
      #       env._subtask = 0
      
      # Pretrained Subtask Exploration
      # if activated_subtask_experiment:
      #   if i > 25_000 and i <= 50_000:
      #       env._subtask = 1
      #   elif i > 50_000 and i <= 75_000:
      #       env._subtask = 2
      #   elif i > 75_000 and i <= 100_000:
      #       env._subtask = 3
      #   elif i > 100_000:
      #       activated_subtask_experiment = False
      #       env._subtask = 0
      #   else:
      #       env._subtask = 0
        
            
          
      if i < config.num_seed_steps:
        #Pretrain Subtask Exploration
        # activated_subtask_experiment = True
        actions = [env.single_action_space.sample() for _ in range(env.num_envs)]
      else:
        policy.eval()
        actions = []
        for j in range(env.num_envs):
          # Add noise to policy actions for better exploration in vector envs
          action = policy.act(observations[j], sample=True)
          # Add small amount of noise for diversity between environments
          if np.random.random() < 0.1:  # 10% chance of random action
            action = env.single_action_space.sample()
          actions.append(action)
      
      # Step all environments
      next_observations, rewards, dones, infos = env.step(actions)
      episode_rewards += rewards
      
      # Handle automatic reset for done environments
      # Note: SyncVectorEnv should auto-reset, but let's be explicit about observation handling
      reset_indices = np.where(dones)[0]
      if len(reset_indices) > 0:
        # The next_observations already contain the reset observations for done envs
        pass  # SyncVectorEnv handles this automatically
      
      # Randomize the order of environment processing to reduce correlation
      env_indices = np.random.permutation(env.num_envs)
        
      for j in env_indices:
        observation = observations[j]
        action = actions[j]
        reward = rewards[j]
        next_observation = next_observations[j]
        done = dones[j]
        mask = 0.0 if done else 1.0
        
        # Log rewards for this environment
        if FLAGS.wandb and j == 0:  # Only log first env to avoid spam
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
        
        # Handle episode completion for this specific environment
        if done:
          total_episodes_completed += 1
          if "holdr" in config.reward_wrapper.type:
            buffer.reset_state()
          if hasattr(env.envs[j], 'reset_state'):
              env.envs[j].reset_state()

          # Log episode info
          if j < len(infos) and "episode" in infos[j]:
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
          # Reset episode reward for this environment
          episode_rewards[j] = 0.0
          
          # Debug: Print episode completion
          if total_episodes_completed % 100 == 0:  # Print every 100 episodes
            print(f"[DEBUG] Step {i}, Env {j} completed episode #{total_episodes_completed}, training steps: {training_step_count}")
      
      # Update observations for next iteration
      observations = next_observations
      
      # For vector environments, adjust training frequency to maintain same sample efficiency
      # Train every N steps where N = num_envs to match single environment sample efficiency
      should_train = (i >= config.num_seed_steps) and ((i + 1) % env.num_envs == 0)
      
      if should_train:
        policy.train()
        # For vector environments, we should train less frequently to match single env sample efficiency
        # Train only once per step, not once per environment
        train_info = policy.update(buffer, i)  # Non-DDP case
        training_step_count += 1

        if (i + 1) % config.log_frequency == 0:
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
              "train/training_step_count": training_step_count,
              "train/total_episodes_completed": total_episodes_completed,
                "train/step": i,
            }, step=i)
          logger.flush()
          
          # Print training progress
          print(f"[TRAINING] Step {i}, Training steps: {training_step_count}, Episodes: {total_episodes_completed}, Avg reward: {np.mean(episode_rewards):.3f}")

      if (i + 1) % config.eval_frequency == 0:
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

      if (i + 1) % config.checkpoint_frequency == 0:
        checkpoint_manager.save(i)

  except KeyboardInterrupt:
    print("Caught keyboard interrupt. Saving before quitting.")

  finally:
    checkpoint_manager.save(i)  # pylint: disable=undefined-loop-variable
    logger.close()


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_name")
  flags.mark_flag_as_required("env_name")
  app.run(main)
