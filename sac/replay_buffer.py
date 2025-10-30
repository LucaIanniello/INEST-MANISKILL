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

"""Lightweight in-memory replay buffer.

Adapted from https://github.com/ikostrikov/jaxrl.
"""

import abc
import collections
from typing import Optional, Tuple

import pdb
import cv2
import numpy as np
import torch
from xirl.models import SelfSupervisedModel

Batch = collections.namedtuple(
    "Batch", ["obses", "actions", "rewards", "next_obses", "masks"]
)
TensorType = torch.Tensor
ModelType = SelfSupervisedModel


class ReplayBuffer:
  """Buffer to store environment transitions."""

  def __init__(
      self,
      obs_shape,
      next_obs_shape,
      action_shape,
      capacity,
      device,
  ):
    """Constructor.

    Args:
      obs_shape: The dimensions of the observation space.
      action_shape: The dimensions of the action space
      capacity: The maximum length of the replay buffer.
      device: The torch device wherein to return sampled transitions.
    """
    self.capacity = capacity
    self.device = device

    # Always use float32 for state-based observations, uint8 only for actual images
    # Since we're using flatten_observation which converts everything to float32,
    # we should use float32 for the buffer as well
    obs_dtype = np.float32
    self.obses = self._empty_arr(obs_shape, obs_dtype)
    self.next_obses = self._empty_arr(next_obs_shape, obs_dtype)
    self.actions = self._empty_arr(action_shape, np.float32)
    self.rewards = self._empty_arr((1,), np.float32)
    self.masks = self._empty_arr((1,), np.float32)

    self.idx = 0
    self.size = 0

  def _empty_arr(self, shape, dtype):
    """Creates an empty array of specified shape and type."""
    return np.empty((self.capacity, *shape), dtype=dtype)

  def _to_tensor(self, arr):
    """Convert an ndarray to a torch Tensor and move it to the device."""
    return torch.as_tensor(arr, device=self.device, dtype=torch.float32)

  def insert(
      self,
      obs,
      action,
      reward,
      next_obs,
      mask,
  ):
    """Insert an episode transition into the buffer."""
    np.copyto(self.obses[self.idx], obs)
    np.copyto(self.actions[self.idx], action)
    np.copyto(self.rewards[self.idx], reward)
    np.copyto(self.next_obses[self.idx], next_obs)
    np.copyto(self.masks[self.idx], mask)

    self.idx = (self.idx + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size):
    """Sample an episode transition from the buffer."""
    idxs = np.random.randint(low=0, high=self.size, size=(batch_size,))

    return Batch(
        obses=self._to_tensor(self.obses[idxs]),
        actions=self._to_tensor(self.actions[idxs]),
        rewards=self._to_tensor(self.rewards[idxs]),
        next_obses=self._to_tensor(self.next_obses[idxs]),
        masks=self._to_tensor(self.masks[idxs]),
    )

  def __len__(self):
    return self.size


class ReplayBufferLearnedReward(abc.ABC, ReplayBuffer):
  """Buffer that replaces the environment reward with a learned one.

  Subclasses should implement the `_get_reward_from_image` method.
  """

  def __init__(
      self,
      model,
      res_hw = None,
      batch_size = 64,
      **base_kwargs,
  ):
    """Constructor.

    Args:
      model: A model that ingests RGB frames and returns embeddings. Should be a
        subclass of `xirl.models.SelfSupervisedModel`.
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
      batch_size: How many samples to forward through the model to compute the
        learned reward. Controls the size of the staging lists.
      **base_kwargs: Base keyword arguments.
    """
    super().__init__(**base_kwargs)

    self.model = model
    self.res_hw = res_hw
    self.batch_size = batch_size

    self._reset_staging()

  def _reset_staging(self):
    self.obses_staging = []
    self.next_obses_staging = []
    self.actions_staging = []
    self.rewards_staging = []
    self.masks_staging = []
    self.pixels_staging = []

  def _pixel_to_tensor(self, arr):
    arr = torch.from_numpy(arr).permute(2, 0, 1).float()[None, None, Ellipsis]
    arr = arr / 255.0
    arr = arr.to(self.device)
    return arr

  @abc.abstractmethod
  def _get_reward_from_image(self):
    """Forward the pixels through the model and compute the reward."""

  def insert(
      self,
      obs,
      action,
      reward,
      next_obs,
      mask,
      pixels,
  ):
    """The insert method in the ReplayBufferLearnedReward class is responsible for adding new experiences 
    to the replay buffer. This method also handles the computation of learned rewards using a model that 
    processes image data."""
    
    if len(self.obses_staging) < self.batch_size:
      self.obses_staging.append(obs)
      self.next_obses_staging.append(next_obs)
      self.actions_staging.append(action)
      self.rewards_staging.append(reward)
      self.masks_staging.append(mask)
      if self.res_hw is not None:
        h, w = self.res_hw
        pixels = cv2.resize(pixels, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
      self.pixels_staging.append(pixels)
    else:
      for obs_s, action_s, reward_s, next_obs_s, mask_s in zip(
          self.obses_staging,
          self.actions_staging,
          self._get_reward_from_image(),
          self.next_obses_staging,
          self.masks_staging,
      ):
        super().insert(obs_s, action_s, reward_s, next_obs_s, mask_s)
      self._reset_staging()


class ReplayBufferDistanceToGoal(ReplayBufferLearnedReward):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      goal_emb,
      distance_scale = 1.0,
      **base_kwargs,
  ):
    super().__init__(**base_kwargs)

    self.goal_emb = goal_emb
    self.distance_scale = distance_scale

  def _get_reward_from_image(self):
    image_tensors = [self._pixel_to_tensor(i) for i in self.pixels_staging]
    image_tensors = torch.cat(image_tensors, dim=1)
    embs = self.model.infer(image_tensors).numpy().embs
    # embs = self.model.module.infer(image_tensors).numpy().embs
    dists = -1.0 * np.linalg.norm(embs - self.goal_emb, axis=-1)
    dists *= self.distance_scale
    return dists


class ReplayBufferGoalClassifier(ReplayBufferLearnedReward):
  """Replace the environment reward with the output of a goal classifier."""
  
  """The model.infer method is used to forward the pixels through the model and compute the reward.
  The sigmoid applied then allows to apply the sigmoid activation function. 
  The sigmoid is used because the network used is a Multi-Layer Perceptron (MLP) with a single output unit."""
  
  """prob.item() allows to convert a probability tensor in a scalar value"""

  def _get_reward_from_image(self):
    image_tensors = [self._pixel_to_tensor(i) for i in self.pixels_staging]
    image_tensors = torch.cat(image_tensors, dim=1)
    prob = torch.sigmoid(self.model.infer(image_tensors).embs)
    # prob = torch.sigmoid(self.model.module.infer(image_tensors).embs)
    return prob.detach().cpu().numpy()

class ReplayBufferHOLDR(ReplayBufferLearnedReward):
    """Replay buffer that replaces the environment reward with HOLDR-based rewards."""

    def __init__(
        self,
        subtask_means,
        distance_scale,
        subtask_threshold=5.0,
        subtask_cost=2.0,
        subtask_hold_steps=1,
        **base_kwargs,
    ):
        """
        Args:
            subtask_means: Mean embeddings for each subtask (shape: [num_subtasks, emb_dim]).
            distance_scale: Scaling factors for distances (shape: [num_subtasks]).
            subtask_threshold: Distance threshold for subtask completion.
            subtask_cost: Cost shaping term for subtasks.
            subtask_hold_steps: Number of consecutive steps required to complete a subtask.
            **base_kwargs: Additional arguments for the base ReplayBufferLearnedReward class.
        """
        super().__init__(**base_kwargs)

        self._subtask_means = np.atleast_2d(subtask_means)  # (num_subtasks, emb_dim)
        self._distance_scale = distance_scale               # (num_subtasks,)
        self._num_subtasks = len(subtask_means)

        # Subtask tracking
        self._subtask = 0
        self._subtask_threshold = subtask_threshold
        self._subtask_cost = subtask_cost
        self._subtask_hold_steps = subtask_hold_steps
        self._subtask_solved_counter = 0
        self._non_decreasing_reward = False
        

    def reset_state(self):
        """Reset subtask tracking variables."""
        self._subtask = 0
        self._subtask_solved_counter = 0


    def _compute_embedding_distance(self, emb, goal_emb, subtask_idx):
        """Compute the scaled distance between the embedding and the goal embedding."""
        dist = np.linalg.norm(emb - goal_emb)
        # dist *= self._distance_scale[subtask_idx]
        dist = self._distance_reward(dist)
        return dist
      
    def _distance_reward(self, d, alpha=0.001, beta=0.01, gamma=1e-3):
        """Compute the distance-based reward."""
        return -alpha * d**2 - beta * np.sqrt(d**2 + gamma)

    def _check_subtask_completion(self, dist, current_reward):
      if self._subtask == 0:
        if dist > -0.1:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = min(self._num_subtasks - 1, self._subtask + 1)
                self._subtask_solved_counter = 0
                if self._non_decreasing_reward:
                    self._prev_reward = current_reward
        else:
            self._subtask_solved_counter = 0
      elif self._subtask == 1:
        if dist > - 0.15:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = min(self._num_subtasks - 1, self._subtask + 1)
                self._subtask_solved_counter = 0
                if self._non_decreasing_reward:
                    self._prev_reward = current_reward
        else:
            self._subtask_solved_counter = 0
      elif self._subtask == 2:
        if dist > -0.25:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = min(self._num_subtasks - 1, self._subtask + 1)
                self._subtask_solved_counter = 0
                if self._non_decreasing_reward:
                    self._prev_reward = current_reward
        else:
            self._subtask_solved_counter = 0
            
    def _get_reward_from_image(self):
        print("REPLAY-BUFFER: Computing HOLDR-based reward from image.")
        """Compute the HOLDR-based reward for the current batch of pixels."""
        image_tensors = [self._pixel_to_tensor(i) for i in self.pixels_staging]
        image_tensors = torch.cat(image_tensors, dim=1)
        embs = self.model.infer(image_tensors).numpy().embs  # Shape: (batch_size, emb_dim)
        # embs = self.model.module.infer(image_tensors).numpy().embs
        
        rewards = []
        for emb in embs:
            if self._subtask >= self._num_subtasks-1:
                reward = self._subtask_cost * self._subtask
            else:
                            
                goal_emb = self._subtask_means[self._subtask]
                dist = self._compute_embedding_distance(emb, goal_emb, self._subtask)
                      
                step_reward = dist  # Base on distance to goal
                bonus_reward = self._subtask * self._subtask_cost
                reward = step_reward + bonus_reward
                self._check_subtask_completion(dist, reward)
            
            rewards.append(reward)

        # pdb.set_trace()
        return np.array(rewards)

class ReplayBufferREDS(ReplayBufferLearnedReward):
    """Replay buffer that replaces the environment reward with REDS-based rewards."""

    def __init__(
        self,
        subtask_phrases=None,
        **base_kwargs,
    ):
        """
        Args:
            subtask_phrases: List of subtask phrases (optional, will use default if None).
            **base_kwargs: Additional arguments for the base ReplayBufferLearnedReward class.
        """
        super().__init__(**base_kwargs)
        # Use default phrases if not provided
        if subtask_phrases is None:
            subtask_phrases = [
                "The robot moves the red block in the goal zone",
                "The robot moves the blue block in the goal zone",
                "The robot moves the yellow block in the goal zone"
            ]
        self.text_phrases = subtask_phrases
        self.text_features = []
        for phrase in self.text_phrases:
          # Pass as a batch of 1 video, 1 phrase
          text_feature_list = self.model.encode_text([[phrase]])
          # text_feature_list is a list of 1 tensor of shape (1, D)
          text_feature = text_feature_list[0][0]  # shape: (D,)
          self.text_features.append(text_feature)
        self.text_features = torch.stack(self.text_features, dim=0).to(self.device)
        
    def cos_sim(self, x1, x2):
        normed_x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
        normed_x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
        return torch.matmul(normed_x1, normed_x2.T)
      
    def text_score(self, image_features, text_features, logit=1.0):
        return (self.cos_sim(image_features, text_features) + 1) / 2 * logit

    def _get_reward_from_image(self):
        """Compute the REDS-based reward for the current batch of pixels."""
        image_tensors = [self._pixel_to_tensor(i) for i in self.pixels_staging]
        image_tensors = torch.cat(image_tensors, dim=1)  # (1, B, C, H, W)
        image_features = self.model.encode_video(image_tensors)  # (B, D) or (1, B, D)

        # Compute cosine similarity with each subtask embedding
        cont_matrix = self.text_score(image_features, self.text_features)  # (B, N)
        diag_cont_matrix = torch.diagonal(cont_matrix, dim1=-2, dim2=-1)

        # Add bias to prevent phase switching
        N = self.text_features.shape[0]
        eps = 5e-2
        bias = torch.linspace(eps * (N - 1), 0.0, N, device=diag_cont_matrix.device)  # (N,)
        diag_cont_matrix += bias  # (B, N)

        # For each sample, select the subtask with max similarity
        target_text_indices = torch.argmax(diag_cont_matrix, dim=-1)

        # Gather the corresponding subtask embedding for each sample
        task_embeddings = self.text_features[target_text_indices]  # (B, D)
        if image_features.dim() == 3:
            image_features = image_features.squeeze(0)  # (1, D)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)  # (1, D)
        if task_embeddings.dim() == 1:
            task_embeddings = task_embeddings.unsqueeze(0)  # (1, D)
        # Compute reward for each sample
        reward = self.model.predict_reward(image_features, task_embeddings)
        # Handle case where reward is a list of tensors
        if isinstance(reward, list):
            if len(reward) == 1:
                reward = reward[0]
            else:
                reward = torch.stack(reward)
        return reward.detach().cpu().numpy()
