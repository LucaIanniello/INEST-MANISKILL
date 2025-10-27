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

"""Compute and store the mean goal embedding using a trained model."""

import os
import typing

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm
import utils
from xirl import common
from xirl.models import SelfSupervisedModel
import pdb
import matplotlib.pyplot as plt
from sac.wrappers import HOLDRLearnedVisualReward
import json

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean(
    "restore_checkpoint", True,
    "Restore model checkpoint. Disabling loading a checkpoint is useful if you "
    "want to measure performance at random initialization.")

ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


def embed(
    model,
    downstream_loader,
    device,
):
  """Embed the stored trajectories and compute mean goal embedding."""
  for class_name, class_loader in downstream_loader.items():
    logging.info("Embedding %s.", class_name)
    for batch in tqdm(iter(class_loader), leave=False):
      print(batch['video_name'])
      out = model.infer(batch["frames"].to(device))
      # out = model.module.infer(batch["frames"].to(device))
      emb = out.numpy().embs
      break
    break   
  return emb


def setup():
  """Load the latest embedder checkpoint and dataloaders."""
  config = utils.load_config_from_dir(FLAGS.experiment_path)
  model = common.get_model(config)
  downstream_loaders = common.get_downstream_dataloaders(config, True)["valid"]
  checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
  if FLAGS.restore_checkpoint:
    checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
    global_step = checkpoint_manager.restore_or_initialize()
    logging.info("Restored model from checkpoint %d.", global_step)
  else:
    logging.info("Skipping checkpoint restore.")
  return model, downstream_loaders

def compute_embedding_distance(emb, goal_emb, subtask_idx):
    dist = np.linalg.norm(emb - goal_emb)
    # dist *= self._distance_scale[subtask_idx]
    mod_dist = distance_reward(dist)
    return dist, mod_dist

def distance_reward(d, alpha=0.001, beta=0.01, gamma=1e-3):
    return -alpha * d**2 - beta * np.sqrt(d**2 + gamma)

def check_subtask_completion(dist, current_reward, subtask, subtask_solved_counter,
                             subtask_threshold, subtask_hold_steps,
                             non_decreasing_reward, num_subtasks):
    prev_reward = 0.0
    # Logic mirrors the provided _check_subtask_completion
    if subtask == 0:
        if dist > -0.02:
            subtask_solved_counter += 1
            if subtask_solved_counter >= subtask_hold_steps:
                subtask = min(num_subtasks - 1, subtask + 1)
                subtask_solved_counter = 0
                if non_decreasing_reward:
                    prev_reward = current_reward
        else:
            subtask_solved_counter = 0
    elif subtask == 1:
        # Hardcoded threshold for subtask 1, as in your example
        if dist > -0.02:
            subtask_solved_counter += 1
            if subtask_solved_counter >= subtask_hold_steps:
                subtask = min(num_subtasks - 1, subtask + 1)
                subtask_solved_counter = 0
                if non_decreasing_reward:
                    prev_reward = current_reward
        else:
            subtask_solved_counter = 0
    elif subtask == 2:
        # Hardcoded threshold for subtask 2, as in your example
        if dist > -0.01:
            subtask_solved_counter += 1
            if subtask_solved_counter >= subtask_hold_steps:
                subtask = 3
                subtask_solved_counter = 0
                if non_decreasing_reward:
                    prev_reward = current_reward
        else:
            subtask_solved_counter = 0
    # You can add more elifs for further subtasks if needed
    return prev_reward, subtask, subtask_solved_counter


def main(_):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  model, downstream_loader = setup()
  model.to(device).eval()
  rews = []
  print(FLAGS.experiment_path)
  # if "xirl_embodiment" in FLAGS.experiment_path:
  #   print("Using distance to goal reward function")
  #   goal_emb = utils.load_pickle(FLAGS.experiment_path, "goal_emb.pkl")
  #   distance_scale = utils.load_pickle(FLAGS.experiment_path,
  #                                           "distance_scale.pkl")
  #   embs = embed(model, downstream_loader, device)
  
  #   for emb in embs:
  #     dist = np.linalg.norm(emb - goal_emb)
  #     dist = -1.0 * dist * distance_scale
  #     rews.append(dist)
      
  # # elif "holdr" in FLAGS.experiment_path:
  print("Using HOLDR reward function")
  subtask_means = utils.load_pickle(FLAGS.experiment_path, "subtask_means.pkl")
  distance_scale = utils.load_pickle(FLAGS.experiment_path, "distance_scale.pkl")
  embs = embed(model, downstream_loader, device)
  subtask = 0
  non_decreasing_reward = False
  subtask_cost = 2.0
  subtask_threshold = 5.5 
  subtask_hold_steps = 1
  distance_normalizer = 5
  subtask_solved_counter = 0
  
  cosine_similarity_emb_subtask_1_vec = []
  cosine_similarity_emb_subtask_2_vec = []
  cosine_similarity_emb_subtask_3_vec = []
  previous_reward = -10.0
  
  # Track the previous subtask to detect transitions
  previous_subtask = 0
  transition_period = 10  # Number of frames to smooth the transition
  transition_counter = 0
  transition_baseline_distance = None  # Will store the initial distance after transition
  previous_distance = None  # Track previous distance to detect when distance starts decreasing
  monotonic_reward = -10.0  # Ensure reward never decreases (initialize with low value)
  
  i = 0
  for emb in embs: 
    
    subtask_1_mean , subtask_2_mean, subtask_3_mean = subtask_means
          
    cosine_similarity_emb_subtask_1 = np.dot(emb, subtask_1_mean) / (np.linalg.norm(emb) * np.linalg.norm(subtask_1_mean))
    cosine_similarity_emb_subtask_2 = np.dot(emb, subtask_2_mean) / (np.linalg.norm(emb) * np.linalg.norm(subtask_2_mean))
    cosine_similarity_emb_subtask_3 = np.dot(emb, subtask_3_mean) / (np.linalg.norm(emb) * np.linalg.norm(subtask_3_mean))
    
    cosine_similarity_emb_subtask_1_vec.append(cosine_similarity_emb_subtask_1)
    cosine_similarity_emb_subtask_2_vec.append(cosine_similarity_emb_subtask_2)
    cosine_similarity_emb_subtask_3_vec.append(cosine_similarity_emb_subtask_3)
    
    if subtask >= 3:
          reward = subtask * subtask_cost
          print(f"Frame: {i}, Subtask {subtask}, Reward{reward}, Dist{dist} Cosine Similarity: {cosine_similarity_emb_subtask_1}, {cosine_similarity_emb_subtask_2}, {cosine_similarity_emb_subtask_3}")
          rews.append(reward)
    else:
          goal_emb = subtask_means[subtask]
          old_dist, dist = compute_embedding_distance(emb, goal_emb, subtask) 
          
          step_reward = dist  # Base on distance to goal
          bonus_reward = subtask * subtask_cost
          current_reward = step_reward + bonus_reward
                
          
          # Use the monotonic reward as our actual reward
          reward = current_reward
          # reward = np.tanh(reward/0.5)
          # if reward < previous_reward:
          #     reward = previous_reward
          # else:
          #     previous_reward = reward
          print(f"Frame: {i}, Subtask {subtask}, Reward{reward}, Dist{dist}, Cosine Similarity: {cosine_similarity_emb_subtask_1}, {cosine_similarity_emb_subtask_2}, {cosine_similarity_emb_subtask_3}")
          # print(f"Reward: {reward}, Subtask: {subtask}, Distance: {dist}")
          rews.append(reward)      
          prev_reward, subtask, subtask_solved_counter = check_subtask_completion(
              dist, reward, subtask, subtask_solved_counter,
              subtask_threshold, subtask_hold_steps,
              non_decreasing_reward, len(subtask_means))
    i += 1
      
  reward_save_path = os.path.join("/home/liannello/xirl/experiment_results/3Subtask_plots/Egocentric/", "reward.json")
  with open(reward_save_path, "w") as f:
        json.dump(rews, f)
  print(f"Saved rewards to: {reward_save_path}")
  
  cosine_sim_dict = {
      "Key": {
          "Subtask_1": np.asarray(cosine_similarity_emb_subtask_1_vec).tolist(),
          "Subtask_2": np.asarray(cosine_similarity_emb_subtask_2_vec).tolist(),
          "Subtask_3": np.asarray(cosine_similarity_emb_subtask_3_vec).tolist()
      }
  }
  
  cosine_save_path = os.path.join("/home/liannello/xirl/experiment_results/6Subtask/6Subtask_Pretrain_results/Egocentric/", "cosine.json")
  with open(cosine_save_path, "w") as f:
        json.dump(cosine_sim_dict, f)
  print(f"Saved rewards to: {cosine_save_path}")
  # Save reward plot
  plt.figure()
  plt.plot(rews)
  plt.title("Reward vs Time")
  plt.xlabel("Step")
  plt.ylabel("Reward")
  plt.grid(True)
  
  plt.figure()
  plt.plot(cosine_similarity_emb_subtask_1_vec, label="Subtask 1")
  plt.plot(cosine_similarity_emb_subtask_2_vec, label="Subtask 2")
  plt.plot(cosine_similarity_emb_subtask_3_vec, label="Subtask 3")
  plt.title("Cosine Similarity vs Time")
  plt.xlabel("Step")
  plt.ylabel("Cosine Similarity")
  plt.legend(loc="lower left")
  plt.grid(True)
  cosine_save_path = os.path.join("/home/liannello/xirl/experiment_results/3Subtask_plots/Egocentric/", "CosineSimilarity_SmoothTransitions.png")
  plt.savefig(cosine_save_path, bbox_inches='tight')
  print(f"Saved cosine similarity plot to: {cosine_save_path}")
  plt.close()

  # Save the plot instead of showing it
  save_path = os.path.join("/home/liannello/xirl/experiment_results/3Subtask_plots/Egocentric/", "ALLO_NEWDIST_TEST.png")
  plt.savefig(save_path, bbox_inches='tight')
  print(f"Saved reward plot to: {save_path}")
  plt.close()

if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_path")
  app.run(main)
