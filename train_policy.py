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
import gymnasium as gym
import mani_skill.envs
from ml_collections import config_dict
from ml_collections import config_flags
import numpy as np
from sac import agent
from sac import replay_buffer
import torch
from torchkit import CheckpointManager
from torchkit import experiment
from torchkit import Logger
from tqdm.auto import tqdm
import utils
from utils import flatten_observation
import wandb

import os
import json
import inspect
from types import MethodType

import time

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


def safe_render(env, mode="rgb_array", **kwargs):
    """Call env.render safely, handling signature mismatches between Gym and ManiSkill wrappers.
    
    Some ManiSkill wrappers implement render(self) while Gym expects render(self, mode, **kwargs).
    This function tries both calling conventions and returns the result.
    """
    try:
        # Try Gym convention first (with mode)
        return env.render(mode=mode, **kwargs)
    except TypeError as e:
        if "positional argument" in str(e) or "unexpected keyword argument" in str(e):
            # Fallback to ManiSkill convention (no args)
            try:
                return env.render()
            except Exception:
                # If both fail, re-raise the original error
                raise e
        else:
            # Re-raise non-signature related errors
            raise e


def patch_env_render_compatibility(env):
    """Automatically patch any wrapper in the env chain that has render signature mismatch.
    
    This finds wrappers with render(self) signature and replaces them with compatibility
    wrappers that accept mode and **kwargs but call the underlying no-arg render.
    """
    def make_compatible_render(original_render):
        """Create a compatible render method from an incompatible one."""
        def compatible_render(self, mode="rgb_array", **kwargs):
            try:
                # First try to call original with mode (in case it was updated)
                sig = inspect.signature(original_render)
                if "mode" in sig.parameters or len(sig.parameters) > 1:
                    return original_render(mode=mode, **kwargs)
                else:
                    # Original only accepts self, call without args
                    return original_render()
            except TypeError:
                # Fallback to no-args call
                return original_render()
        return compatible_render
    
    # Walk the wrapper chain and patch incompatible render methods
    current_env = env
    patched_count = 0
    
    while current_env is not None:
        render_method = getattr(current_env, 'render', None)
        if render_method is not None:
            try:
                sig = inspect.signature(render_method)
                params = list(sig.parameters.keys())
                
                # Check if this render method only accepts 'self' (no mode parameter)
                if len(params) <= 1 and "mode" not in sig.parameters:
                    # Patch this wrapper's render method
                    compatible_method = make_compatible_render(render_method)
                    current_env.render = MethodType(compatible_method, current_env)
                    print(f"Patched render method on {type(current_env).__name__}")
                    patched_count += 1
            except (ValueError, TypeError):
                # Can't inspect signature, skip
                pass
        
        # Move to next wrapper in chain
        if hasattr(current_env, 'env'):
            current_env = current_env.env
        elif hasattr(current_env, 'unwrapped') and current_env.unwrapped != current_env:
            current_env = current_env.unwrapped
        else:
            break
    
    print(f"Patched {patched_count} wrapper(s) for render compatibility")
    return env


def evaluate(
    policy,
    env,
    num_episodes,
):
  """Evaluate the policy and dump rollout videos to disk."""
  episode_rewards = []
  policy.eval()
  stats = collections.defaultdict(list)
  last_episode_frames = []
  last_episode_rewards = []
  last_episode_actions = []
  exp_dir = os.path.dirname(os.path.dirname(env.save_dir))
    
  for i in range(num_episodes):
    observation = env.reset()
    done = False
    # Flatten the observation to handle ManiSkill's object dtype arrays
    observation = flatten_observation(observation)
    if "holdr" in FLAGS.experiment_name:
      # Reset the buffer and environment state for holdr.
      env.reset_state()
    episode_reward = 0
    count=0
    while not done:
      # Capture frame for last episode only
            if i == num_episodes - 1:
                frame = safe_render(env)
                last_episode_frames.append(frame)
            
            action = policy.act(observation, sample=False)
            
            if i == num_episodes - 1:
                if isinstance(action, torch.Tensor):
                    action_np = action.cpu().numpy()
                else:
                    action_np = action  # Already numpy array
                    
                last_episode_actions.append(action_np.tolist())
                
            
            base_env= env.unwrapped
            base_env.index_seed_steps = count
            count+=1
            result = env.step(action)
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                observation, reward, done, info = result
            # Flatten the observation for the next step
            observation = flatten_observation(observation)
            episode_reward += reward
            
            # Capture reward for last episode only
            if i == num_episodes - 1:
                last_episode_rewards.append(reward)
    for k, v in info["episode"].items():
      stats[k].append(v)
    if "eval_score" in info:
      stats["eval_score"].append(info["eval_score"])
      print(f"Episode {i} eval score: {stats['eval_score']}")
    episode_rewards.append(episode_reward)
    
    actions_file = os.path.join(exp_dir, "last_evaluation_actions.json")
        
    action_data = {
        "actions": last_episode_actions,
        "total_reward": sum(last_episode_rewards),
    }
    
    with open(actions_file, 'w') as f:
        json.dump(action_data, f, indent=2)
    
    logging.info(f"Saved last evaluation actions to {actions_file}")
    
    # Log video and reward plot to wandb
    if last_episode_frames and FLAGS.wandb:
        # Convert frames to proper format (time, channel, height, width)
        frames = np.array([frame.transpose(2, 0, 1) for frame in last_episode_frames])
        wandb.log({
            "eval/last_eval_video": wandb.Video(frames, fps=30, format="mp4"),
            "eval/step": i  # Use current training step
        })
        
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(last_episode_rewards)
            plt.title("Reward Evolution - Last Evaluation Episode")
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.tight_layout()
            wandb.log({
                "eval/last_reward_plot": wandb.Image(plt),
                "eval/step": i
            })
            plt.close()
        except ImportError:
            pass  # matplotlib not available
  for k, v in stats.items():
    stats[k] = np.mean(v)
    
  if FLAGS.wandb:
        wandb.log({
            "eval/mean_episode_reward": np.mean(episode_rewards),
            "eval/episode_rewards": episode_rewards,
            "eval/step": i,
        })
        if "eval_score" in stats:
            wandb.log({
                "eval/mean_eval_score": stats["eval_score"],
                "eval/eval_scores": stats.get("eval_score", []),
                "eval/step": i,
            })
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
    if FLAGS.resume:
        wandb_id = "1u7lqxky"
        wandb.init(project="NewEnv", group="NewModel_pyramid_12", name="NewModel_12", id=wandb_id, mode="online", resume="must")
    else:
        wandb.init(project="NewEnv", group="NewModel_pyramid_42", name="NewModel_pyramid_42", mode="online")
    wandb.config.update(FLAGS, allow_val_change=True)
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


 
  # Load env.
  env = utils.make_env(
      FLAGS.env_name,
      FLAGS.seed,
      action_repeat=config.action_repeat,
      frame_stack=config.frame_stack,
  )
  eval_env = utils.make_env(
      FLAGS.env_name,
      FLAGS.seed + 45,
      action_repeat=config.action_repeat,
      frame_stack=config.frame_stack,
      save_dir=osp.join(exp_dir, "video", "eval"),
  )
  
  # Patch render compatibility for both environments
  print("Patching training env render compatibility...")
  env = patch_env_render_compatibility(env)
  print("Patching eval env render compatibility...")
  eval_env = patch_env_render_compatibility(eval_env)
  
  # if config.reward_wrapper.pretrained_path:
  #   print("Using learned reward wrapper.")
  #   env = utils.wrap_learned_reward(env, FLAGS.config, device=device)
  #   eval_env = utils.wrap_learned_reward(eval_env, FLAGS.config, device=device)


  # Dynamically set observation and action space values.
  # Get a sample observation to determine the flattened size
  sample_obs = env.reset() 
  flattened_sample = flatten_observation(sample_obs)
  
  sample_next_obs = env.step(env.action_space.sample())[0]
  flattened_next_sample = flatten_observation(sample_next_obs)
  
  config.sac.obs_dim = flattened_sample.shape[0]
  config.sac.action_dim = env.action_space.shape[0]
  config.sac.action_range = [
      float(env.action_space.low.min()),
      float(env.action_space.high.max()),
  ]

  # Resave the config since the dynamic values have been updated at this point
  # and make it immutable for safety :)
  utils.dump_config(exp_dir, config)
  config = config_dict.FrozenConfigDict(config)

  policy = agent.SAC(device, config.sac)
  
  print("Sample observation flattened shape:", flattened_sample.shape)
  print("Observation space shape:", flattened_sample.shape[0])

  buffer = utils.make_buffer(env, device, config, flattened_obs_shape=(flattened_sample.shape[0],), flattened_next_obs_shape=(flattened_next_sample.shape[0],))

  # Create checkpoint manager.
  checkpoint_dir = osp.join(exp_dir, "checkpoints")
  
  # Fix for HPC cross-device link issue - use same filesystem for temp files
  temp_dir = osp.join(exp_dir, "tmp")
  os.makedirs(temp_dir, exist_ok=True)
  original_tmpdir = os.environ.get('TMPDIR', None)
  os.environ['TMPDIR'] = temp_dir
  
  try:
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        policy=policy,
        **policy.optim_dict(),
    )
  finally:
    # Restore original TMPDIR
    if original_tmpdir is not None:
      os.environ['TMPDIR'] = original_tmpdir
    else:
      os.environ.pop('TMPDIR', None)

  logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)

  try:
    start = checkpoint_manager.restore_or_initialize()
    observation = env.reset()
    done = False
    # Flatten the observation to handle ManiSkill's object dtype arrays
    observation = flatten_observation(observation)
    episode_reward = 0
    training_frames = []
    training_rewards = []
    video_log_frequency = config.eval_frequency * 2  # Log videos less frequently than evaluation
    should_record_video = False
    
    for i in tqdm(range(start, config.num_train_steps), initial=start):
      env.index_seed_step = i
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
        
      if i % video_log_frequency == 0 and not should_record_video:
          should_record_video = True
          
      if i < config.num_seed_steps:
        #Pretrain Subtask Exploration
        # activated_subtask_experiment = True
        action = env.action_space.sample()  
      else:
        policy.eval()
        action = policy.act(observation, sample=True)
        
      if should_record_video:
          frame = safe_render(env)
          training_frames.append(frame) 
          
      # next_observation, reward, done, info = env.step(action, exp_dir = exp_dir, rank = 0, flag="train")
      result = env.step(action)
      if len(result) == 5:
          next_observation, reward, terminated, truncated, info = result
          done = terminated or truncated
      else:
          next_observation, reward, done, info = result
      episode_reward += reward
      
      if FLAGS.wandb:
          if 'coverage_stats' in info:
            # Debug print to see what's in coverage_stats
            if i % 1000 == 0:  # Print every 1000 steps to avoid spam
              print(f"Step {i}: Coverage stats: {info['coverage_stats']}")
            wandb.log({f"exploration/{k}": v for k, v in info['coverage_stats'].items()}, step=i)
          
          wandb.log({
              "train/reward": reward,
              "train/step": i,
          }, step=i)
        
      if should_record_video:
          training_rewards.append(reward)
          
      if not done or "TimeLimit.truncated" in info:
        mask = 1.0
      else:
        mask = 0.0
        
      if FLAGS.wandb:
        wandb.log({
        "train/reward": reward,
        "train/step": i,
        }, step=i)

      # if not config.reward_wrapper.pretrained_path:
      #   # print("No reward wrapper specified. Using default reward.")
      #   buffer.insert(observation, action, reward, next_observation, mask)
      # else:
      #   buffer.insert(
      #       observation,
      #       action,
      #       reward,
      #       next_observation,
      #       mask,
      #       env.render(mode="rgb_array"),
      #   )
      # observation = next_observation
      
      # Flatten next_observation before storing in buffer and updating observation
      next_observation_flattened = flatten_observation(next_observation)
      print("Next observation flattened shape:", next_observation_flattened.shape)
      
      # Convert CUDA tensors to CPU numpy arrays if needed
      if hasattr(reward, 'cpu'):
        reward_cpu = reward.cpu().numpy()
      else:
        reward_cpu = reward
        
      if hasattr(action, 'cpu'):
        action_cpu = action.cpu().numpy()
      else:
        action_cpu = action
      
      buffer.insert(observation, action_cpu, reward_cpu, next_observation_flattened, mask)
      observation = next_observation_flattened

      if done:
        if should_record_video and training_frames  and FLAGS.wandb:
            try:
                # Convert frames to proper format
                frames = np.array([frame.transpose(2, 0, 1) for frame in training_frames])
                wandb.log({
                    "train/training_video": wandb.Video(frames, fps=20, format="mp4"),
                    "train/step": i
                })
                
                # Plot reward evolution for this training episode
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 6))
                    plt.plot(training_rewards)
                    plt.title(f"Training Episode Reward Evolution - Step {i}")
                    plt.xlabel("Episode Step")
                    plt.ylabel("Reward")
                    plt.grid(True, alpha=0.3)
                    
                    # Add some statistics
                    total_reward = sum(training_rewards)
                    avg_reward = np.mean(training_rewards)
                    plt.axhline(y=avg_reward, color='r', linestyle='--', alpha=0.7, 
                              label=f'Avg: {avg_reward:.3f}')
                    plt.legend()
                    plt.tight_layout()
                    
                    wandb.log({
                        "train/training_reward_plot": wandb.Image(plt),
                        "train/episode_total_reward": total_reward,
                        "train/episode_avg_reward": avg_reward,
                        "train/episode_length": len(training_rewards),
                        "train/step": i
                    })
                    plt.close()
                except ImportError:
                    pass  # matplotlib not available
                
                print(f"[VIDEO LOG] Training video logged at step {i}, episode length: {len(training_frames)} frames, total reward: {sum(training_rewards):.3f}")
                
            except Exception as e:
                print(f"[VIDEO LOG ERROR] Failed to log training video: {e}")
            
            # Reset for next recording
            training_frames = []
            training_rewards = []
            should_record_video = False
            
        observation = env.reset()
        done = False
        # Flatten the observation after reset
        observation = flatten_observation(observation)
        print("observation after reset flattened shape:", observation.shape)
        # if "holdr" in config.reward_wrapper.type:
        #   # print("Resetting buffer and environment state.")
        #   # buffer.reset_state()
        #   env.reset_state()

        for k, v in info["episode"].items():
          logger.log_scalar(v, info["total"]["timesteps"], k, "training")
          if FLAGS.wandb:
            wandb.log({
                f"train_done/{k}": v,
                "train_done/step": i,
            }, step=i)
        if FLAGS.wandb:
            wandb.log({
                "train_done/episode_reward": episode_reward,
                "train_done/step": i,
            }, step=i)
        episode_reward = 0
        

      if i >= config.num_seed_steps:
        policy.train()
        train_info = policy.update(buffer, i)

        if (i + 1) % config.log_frequency == 0:
          for k, v in train_info.items():
            logger.log_scalar(v, info["total"]["timesteps"], k, "training")
            if FLAGS.wandb:
              wandb.log({
                  f"train/{k}": v,
                  "train/step": i,
              }, step=i)
          if FLAGS.wandb:
            wandb.log({
              "train/episode_reward": episode_reward,
                "train/step": i,
            }, step=i)
          logger.flush()

      if (i + 1) % config.eval_frequency == 0:
        eval_stats, episode_rewards = evaluate(policy, eval_env, config.num_eval_episodes)
        for k, v in eval_stats.items():
          logger.log_scalar(
              v,
              info["total"]["timesteps"],
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
                "eval/episode_reward": episode_rewards,
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
