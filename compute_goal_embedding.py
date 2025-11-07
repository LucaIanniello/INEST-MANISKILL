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

def read_subgoal_from_file(file_path):
    with open(file_path, 'r') as f:
        subgoal_data = json.load(f)

    # Process the values to extract the last numbers
    processed_data = {}
    for key, paths in subgoal_data.items():
      processed_data[key] = [int(path.split('/')[-1].split('.')[0]) for path in paths]

    return processed_data

def embed(
    model,
    downstream_loader,
    device,
):
  """Embed the stored trajectories and compute mean goal embedding."""
  goal_embs = []
  init_embs = []
  for class_name, class_loader in downstream_loader.items():
    logging.info("Embedding %s.", class_name)
    for batch in tqdm(iter(class_loader), leave=False):
      out = model.module.infer(batch["frames"].to(device))
      # out = model.module.infer(batch["frames"].to(device))
      emb = out.numpy().embs
      init_embs.append(emb[0, :])
      goal_embs.append(emb[-1, :])
  goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
  dist_to_goal = np.linalg.norm(
      np.stack(init_embs, axis=0) - goal_emb, axis=-1).mean()
  distance_scale = 1.0 / dist_to_goal
  return goal_emb, distance_scale

def embed_subtasks(
    model,
    downstream_loader,
    device,
    subgoal_data
):
  """Embed the stored trajectories and compute mean goal embedding."""
  init_embs = []
  all_subgoal_frames_embs = []
  for class_name, class_loader in downstream_loader.items():
    logging.info("Embedding %s.", class_name)
    for batch in tqdm(iter(class_loader), leave=False):
      video_id = batch["video_name"][0].split("/")[-1]
      subgoal_frames = subgoal_data[video_id]
      out = model.infer(batch["frames"].to(device))
      # out = model.module.infer(batch["frames"].to(device))
      emb = out.numpy().embs
      init_embs.append(emb[0, :])
      video_subgoal_embs = []
      for idx in subgoal_frames:
        print(f"Embedding subgoal frame {idx} for video {video_id}")
        video_subgoal_embs.append(emb[idx, :])
      all_subgoal_frames_embs.append(video_subgoal_embs)
  all_subgoal_frames_embs = np.array(all_subgoal_frames_embs)

  # pdb.set_trace()
  
# Compute the mean embedding for each subtask (column) across all videos (rows)
  num_subtasks = len(all_subgoal_frames_embs[1])
  subtask_means = []
# Loop over each subtask
  subtask_means = np.mean(all_subgoal_frames_embs, axis=0)
  # pdb.set_trace()
  # Compute the distance vector
  dist_to_goal = []
  per_subtask_dists = []
  # Distance between the initial embedding and the first subtask
  dist = np.linalg.norm(
      np.stack(init_embs, axis=0) - subtask_means[0], axis=-1
  )
  dist_to_goal.append(np.mean(dist))
  per_subtask_dists.append(dist)
  
  # Distances between consecutive subtasks
  for i in range(1, num_subtasks):
      dist = np.linalg.norm(
          subtask_means[i - 1] - subtask_means[i], axis=-1
      )
      dist_to_goal.append(dist)
      
      video_dists = np.linalg.norm(
          all_subgoal_frames_embs[:, i - 1, :] - all_subgoal_frames_embs[:, i, :], axis=1
      )
      per_subtask_dists.append(video_dists)
      
  dist_to_goal = np.array(dist_to_goal)  # Convert to a NumPy array
  distance_scale = 1.0 / dist_to_goal
  
  subtask_thresholds = np.array([np.percentile(dists, 80) for dists in per_subtask_dists]) * distance_scale
  for i, dists in enumerate(per_subtask_dists):
    print(f"Subtask {i} distance stats: min={np.min(dists)}, max={np.max(dists)}, mean={np.mean(dists)}, median={np.median(dists)}, 80th={np.percentile(dists, 80)}, 30th={np.percentile(dists, 30)}, 50th={np.percentile(dists, 50)}")
  return subtask_means, distance_scale

def setup():
  """Load the latest embedder checkpoint and dataloaders."""
  config = utils.load_config_from_dir(FLAGS.experiment_path)
  model = common.get_model(config)
  downstream_loaders = common.get_downstream_dataloaders(config, False)["train"]
  checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
  if FLAGS.restore_checkpoint:
    checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
    global_step = checkpoint_manager.restore_or_initialize()
    logging.info("Restored model from checkpoint %d.", global_step)
  else:
    logging.info("Skipping checkpoint restore.")
  return model, downstream_loaders


def main(_):
  # Add these at the very beginning
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  np.random.seed(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  
  # Also fix the unused embed() function for consistency
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, downstream_loader = setup()
  model.to(device).eval()

  subgoal_file_path = "/home/liannello/.maniskill/demos/StackPyramid-v1/motionplanning/subtask_frames.json"
  
  subgoal_data = read_subgoal_from_file(subgoal_file_path)
  subtask_means, distance_scale = embed_subtasks(model, downstream_loader, device, subgoal_data)
  utils.save_pickle(FLAGS.experiment_path, subtask_means, "subtask_means.pkl")
  utils.save_pickle(FLAGS.experiment_path, distance_scale, "distance_scale.pkl")
    
  # # if "holdr_embodiment" in FLAGS.experiment_path:
  #   subgoal_data = read_subgoal_from_file(subgoal_file_path)
  #   subtask_means, distance_scale = embed_subtasks(model, downstream_loader, device, subgoal_data)
  #   utils.save_pickle(FLAGS.experiment_path, subtask_means, "subtask_means.pkl")
  #   utils.save_pickle(FLAGS.experiment_path, distance_scale, "distance_scale.pkl")
  # #   # utils.save_pickle(FLAGS.experiment_path, subtask_thresholds, "subtask_thresholds.pkl")
  # else:
  #   goal_emb, distance_scale = embed(model, downstream_loader, device)
  #   utils.save_pickle(FLAGS.experiment_path, goal_emb, "goal_emb.pkl")
  #   utils.save_pickle(FLAGS.experiment_path, distance_scale, "distance_scale.pkl")


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_path")
  app.run(main)
