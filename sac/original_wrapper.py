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

"""Environment wrappers."""

import abc
import collections
import os
import time
import typing

import cv2
import gym
import imageio
import numpy as np
import torch
from xirl.models import SelfSupervisedModel
from xirl.trainers.reds import REDSRewardTrainer
from collections import defaultdict

TimeStep = typing.Tuple[np.ndarray, float, bool, dict]
ModelType = SelfSupervisedModel
TensorType = torch.Tensor
DistanceFuncType = typing.Callable[[float], float]
InfoMetric = typing.Mapping[str, typing.Mapping[str, typing.Any]]


class FrameStack(gym.Wrapper):
  """Stack the last k frames of the env into a flat array.

  This is useful for allowing the RL policy to infer temporal information.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, k):
    """Constructor.

    Args:
      env: A gym env.
      k: The number of frames to stack.
    """
    super().__init__(env)

    assert isinstance(k, int), "k must be an integer."

    self._k = k
    self._frames = collections.deque([], maxlen=k)

    shp = env.observation_space.shape
    self.observation_space = gym.spaces.Box(
        low=env.observation_space.low.min(),
        high=env.observation_space.high.max(),
        shape=((shp[0] * k,) + shp[1:]),
        dtype=env.observation_space.dtype,
    )

  def reset(self):
    obs = self.env.reset()
    for _ in range(self._k):
      self._frames.append(obs)
    return self._get_obs()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._frames.append(obs)
    return self._get_obs(), reward, done, info

  def _get_obs(self):
    assert len(self._frames) == self._k
    return np.concatenate(list(self._frames), axis=0)


class ActionRepeat(gym.Wrapper):
  """Repeat the agent's action N times in the environment.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env, repeat):
    """Constructor.

    Args:
      env: A gym env.
      repeat: The number of times to repeat the action per single underlying env
        step.
    """
    super().__init__(env)

    assert repeat > 1, "repeat should be greater than 1."
    self._repeat = repeat

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._repeat):
      obs, rew, done, info = self.env.step(action)
      total_reward += rew
      if done:
        break
    return obs, total_reward, done, info


class RewardScale(gym.Wrapper):
  """Scale the environment reward."""

  def __init__(self, env, scale):
    """Constructor.

    Args:
      env: A gym env.
      scale: How much to scale the reward by.
    """
    super().__init__(env)

    self._scale = scale

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    reward *= self._scale
    return obs, reward, done, info


class EpisodeMonitor(gym.ActionWrapper):
  """A class that computes episode metrics.

  At minimum, episode return, length and duration are computed. Additional
  metrics that are logged in the environment's info dict can be monitored by
  specifying them via `info_metrics`.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(self, env):
    super().__init__(env)

    self._reset_stats()
    self.total_timesteps: int = 0

  def _reset_stats(self):
    self.reward_sum: float = 0.0
    self.episode_length: int = 0
    self.start_time = time.time()

  def step(self, action):
    obs, rew, done, info = self.env.step(action)

    self.reward_sum += rew
    self.episode_length += 1
    self.total_timesteps += 1
    info["total"] = {"timesteps": self.total_timesteps}

    if done:
      info["episode"] = dict()
      info["episode"]["return"] = self.reward_sum
      info["episode"]["length"] = self.episode_length
      info["episode"]["duration"] = time.time() - self.start_time

    return obs, rew, done, info

  def reset(self):
    self._reset_stats()
    return self.env.reset()


class VideoRecorder(gym.Wrapper):
  """Wrapper for rendering and saving rollouts to disk.

  Reference: https://github.com/ikostrikov/jaxrl/
  """

  def __init__(
      self,
      env,
      save_dir,
      resolution = (256, 256),
      fps = 30,
  ):
    super().__init__(env)

    self.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    self.height, self.width = resolution
    self.fps = fps
    self.enabled = True
    self.current_episode = 0
    self.frames = []

  def step(self, action):
    frame = self.env.render(mode="rgb_array")
    if frame.shape[:2] != (self.height, self.width):
      frame = cv2.resize(
          frame,
          dsize=(self.width, self.height),
          interpolation=cv2.INTER_CUBIC,
      )
    self.frames.append(frame)
    observation, reward, done, info = self.env.step(action)
    if done:
      filename = os.path.join(self.save_dir, f"{self.current_episode}.mp4")
      imageio.mimsave(filename, self.frames, fps=self.fps)
      self.frames = []
      self.current_episode += 1
    return observation, reward, done, info


# ========================================= #
# Learned reward wrappers.
# ========================================= #

# Note: While the below classes provide a nice wrapper API, they are not
# efficient for training RL policies as rewards are computed individually at
# every `env.step()` and so cannot take advantage of batching on the GPU.
# For actually training policies, it is better to use the learned replay buffer
# implementations in `sac.replay_buffer.py`. These store transitions in a
# staging buffer which is forwarded as a batch through the GPU.


class LearnedVisualReward(abc.ABC, gym.Wrapper):
  """Base wrapper class that replaces the env reward with a learned one.

  Subclasses should implement the `_get_reward_from_image` method.
  """

  def __init__(
      self,
      env,
      model,
      device,
      index_seed_step = 0,
      res_hw = None,
  ):
    """Constructor.

    Args:
      env: A gym env.
      model: A model that ingests RGB frames and returns embeddings. Should be a
        subclass of `xirl.models.SelfSupervisedModel`.
      device: Compute device.
      res_hw: Optional (H, W) to resize the environment image before feeding it
        to the model.
    """
    super().__init__(env)

    self._device = device
    self._model = model.to(device).eval()
    self._res_hw = res_hw

  def _to_tensor(self, x):
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None, None, Ellipsis]
    # TODO(kevin): Make this more generic for other preprocessing.
    x = x / 255.0
    x = x.to(self._device)
    return x

  def _render_obs(self):
    """Render the pixels at the desired resolution."""
    # TODO(kevin): Make sure this works for mujoco envs.
    pixels = self.env.render(mode="rgb_array")
    if self._res_hw is not None:
      h, w = self._res_hw
      pixels = cv2.resize(pixels, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    return pixels

  @abc.abstractmethod
  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""

  def step(self, action):
    obs, env_reward, done, info = self.env.step(action)
    # We'll keep the original env reward in the info dict in case the user would
    # like to use it in conjunction with the learned reward.
    info["env_reward"] = env_reward
    pixels = self._render_obs()
    learned_reward = self._get_reward_from_image(pixels)
    
    return obs, learned_reward, done, info


class DistanceToGoalLearnedVisualReward(LearnedVisualReward):
  """Replace the environment reward with distances in embedding space."""

  def __init__(
      self,
      goal_emb,
      distance_scale = 1.0,
      **base_kwargs,
  ):
    """Constructor.

    Args:
      goal_emb: The goal embedding.
      distance_scale: Scales the distance from the current state embedding to
        that of the goal state. Set to `1.0` by default.
      **base_kwargs: Base keyword arguments.
    """
    super().__init__(**base_kwargs)

    self._goal_emb = np.atleast_2d(goal_emb)
    self._distance_scale = distance_scale

  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    # print("Computing reward from image dist.")
    image_tensor = self._to_tensor(image)
    emb = self._model.infer(image_tensor).numpy().embs
    # emb = self._model.module.infer(image_tensor).numpy().embs
    dist = -1.0 * np.linalg.norm(emb - self._goal_emb)
    dist *= self._distance_scale
    return dist


class GoalClassifierLearnedVisualReward(LearnedVisualReward):
  """Replace the environment reward with the output of a goal classifier."""

  def _get_reward_from_image(self, image):
    """Forward the pixels through the model and compute the reward."""
    print("Computing reward from image.")
    image_tensor = self._to_tensor(image)
    prob = torch.sigmoid(self._model.infer(image_tensor).embs)
    # prob = torch.sigmoid(self._model.module.infer(image_tensor).embs)
    return prob.item()


class HOLDRLearnedVisualReward(LearnedVisualReward):
    def __init__(
        self,
        subtask_means,
        distance_scale,
        index_seed_step = 0,
        subtask_threshold=5.0,
        subtask_cost=2.0,
        subtask_hold_steps=1,
        intrinsic_scale=0.2,
        k_nearest=10,
        max_memory=5_000,
        coverage_grid_size=100,
        coverage_threshold=0.1,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        self._subtask_means = np.atleast_2d(subtask_means)  
        self._distance_scale = distance_scale               
        self._num_subtasks = len(subtask_means)
        
        self.index_seed_step = index_seed_step

        # Subtask tracking
        self._subtask = 0
        self._subtask_cost = subtask_cost
        self._subtask_hold_steps = subtask_hold_steps
        self._subtask_solved_counter = 0
        
        self._intrinsic_scale = intrinsic_scale
        self._k_nearest = k_nearest
        self._max_memory = max_memory
        self._embedding_memory_per_subtask = defaultdict(list)
        
        # Coverage tracking attributes
        self._coverage_grid_size = coverage_grid_size
        self._coverage_threshold = coverage_threshold
        self._embedding_dim = None 
        self._coverage_grid = None
        self._visited_states = set()
        self._state_visit_counts = {}
        self._unique_states_visited = 0
        self._total_steps = 0
        
        # Similarity-preserving grid mapping
        self._use_similarity_grid = True
        self._grid_dims = 2  # Use 2D grid for better visualization
        self._embedding_buffer = []
        self._buffer_size = 1000  # Collect embeddings before computing mapping
        self._similarity_mapping_fitted = False
        self._embedding_mean = None
        self._projection_matrix = None
        
        # Coverage metrics storage
        self._coverage_history = []
        self._novelty_history = []
        self._subtask_coverage = {}
        
        self._visited_states_per_subtask = defaultdict(set)
        self._norm_fitted = False
        
        self.subtask_switch_step = 0
        self.increase_intrinsic_scale_after_subtask = 0.2

    def _initialize_coverage_grid(self, embedding_dim):
        """Initialize coverage tracking structures based on embedding dimension"""
        self._embedding_dim = embedding_dim
        
        if self._use_similarity_grid:
            # Use 2D grid for similarity-preserving mapping
            self._coverage_grid = np.zeros([self._coverage_grid_size, self._coverage_grid_size])
            print(f"WRAPPER: Initialized 2D similarity-preserving coverage grid {self._coverage_grid.shape} for embedding dim {embedding_dim}")
        else:
            # Fallback: use coordinate-wise mapping with limited dimensions
            grid_dims = min(embedding_dim, 3)
            self._coverage_grid = np.zeros([self._coverage_grid_size] * grid_dims)
            print(f"WRAPPER: Initialized coordinate-wise coverage grid {self._coverage_grid.shape} for embedding dim {embedding_dim}")
    
    def _fit_similarity_mapping(self):
        """Fit a PCA-like mapping to preserve embedding similarities in 2D grid"""
        if len(self._embedding_buffer) < min(self._buffer_size, 100):
            return False
            
        try:
            embeddings = np.array(self._embedding_buffer)
            print(f"WRAPPER: Fitting similarity mapping on {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
            
            # Center the embeddings
            self._embedding_mean = embeddings.mean(axis=0)
            centered = embeddings - self._embedding_mean
            
            # Compute covariance matrix
            cov_matrix = np.cov(centered.T)
            
            # Eigendecomposition for PCA
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalues (descending)
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Keep top 2 components for 2D grid mapping
            self._projection_matrix = eigenvecs[:, :self._grid_dims]
            
            # Project all embeddings to get the bounds for normalization
            projected = centered @ self._projection_matrix
            self._projection_min = projected.min(axis=0)
            self._projection_max = projected.max(axis=0)
            
            # Calculate explained variance
            explained_variance = eigenvals[:self._grid_dims].sum() / eigenvals.sum()
            print(f"WRAPPER: Similarity mapping fitted. Explained variance: {explained_variance:.3f}")
            print(f"WRAPPER: Projection bounds: min={self._projection_min}, max={self._projection_max}")
            
            self._similarity_mapping_fitted = True
            return True
            
        except Exception as e:
            print(f"WRAPPER ERROR: Failed to fit similarity mapping: {e}")
            self._use_similarity_grid = False
            return False
        
    def _fit_running_stats(self, emb):
        if not self._norm_fitted:
            self._running_min = emb.copy()
            self._running_max = emb.copy()
            self._norm_fitted = True
        else:
            self._running_min = np.minimum(self._running_min, emb)
            self._running_max = np.maximum(self._running_max, emb)

    def _get_grid_coordinates(self, embedding):
        if self._embedding_dim is None:
            self._initialize_coverage_grid(len(embedding))
        
        if self._use_similarity_grid:
            return self._get_similarity_grid_coordinates(embedding)
        else:
            return self._get_coordinate_wise_grid_coordinates(embedding)
    
    def _get_similarity_grid_coordinates(self, embedding):
        """Map embedding to 2D grid coordinates preserving similarity structure"""
        # Collect embeddings for fitting the similarity mapping
        if not self._similarity_mapping_fitted:
            self._embedding_buffer.append(embedding.copy())
            if len(self._embedding_buffer) >= min(self._buffer_size, 1000):
                fitted = self._fit_similarity_mapping()
                if not fitted:
                    return self._get_coordinate_wise_grid_coordinates(embedding)
            else:
                # Use fallback until we have enough data
                return self._get_coordinate_wise_grid_coordinates(embedding)
        
        try:
            # Project embedding to 2D space using fitted PCA
            centered = embedding - self._embedding_mean
            projected = centered @ self._projection_matrix
            
            # Normalize to [0, 1] using fitted bounds with some margin
            margin = 0.1 * (self._projection_max - self._projection_min)
            extended_min = self._projection_min - margin
            extended_max = self._projection_max + margin
            
            normalized = (projected - extended_min) / (extended_max - extended_min + 1e-8)
            normalized = np.clip(normalized, 0.0, 1.0)
            
            # Convert to grid coordinates
            coords = (normalized * (self._coverage_grid_size - 1)).astype(int)
            coords = np.clip(coords, 0, self._coverage_grid_size - 1)
            
            return tuple(int(coords.flatten()[i]) for i in range(len(coords.flatten())))
            
        except Exception as e:
            print(f"WRAPPER WARNING: Similarity grid mapping failed: {e}")
            return self._get_coordinate_wise_grid_coordinates(embedding)
    
    def _get_coordinate_wise_grid_coordinates(self, embedding):
        """Fallback: simple coordinate-wise mapping"""
        self._fit_running_stats(embedding)

        denom = (self._running_max - self._running_min + 1e-6)
        normalized = np.clip((embedding - self._running_min) / denom, 0.0, 1.0)
        coords = (normalized * (self._coverage_grid_size - 1)).astype(int)
        coords = np.clip(coords, 0, self._coverage_grid_size - 1)
        
        # Match grid dimensions
        grid_dims = len(self._coverage_grid.shape)
        coord_tuple = tuple(int(coords.flatten()[i]) for i in range(min(len(coords.flatten()), grid_dims)))
        
        # Pad with zeros if needed
        if len(coord_tuple) < grid_dims:
            coord_tuple = coord_tuple + (0,) * (grid_dims - len(coord_tuple))
            
        return coord_tuple

    def _update_coverage_metrics(self, emb):
        """Update various coverage metrics"""
        self._total_steps += 1
        
        # Grid-based coverage
        grid_coords = self._get_grid_coordinates(emb)
        was_new_cell = False
        
        # Ensure grid_coords is a valid index for the grid
        try:
            if self._coverage_grid[grid_coords] == 0:
                self._coverage_grid[grid_coords] = 1
                self._unique_states_visited += 1
                was_new_cell = True
        except (IndexError, ValueError) as e:
            print(f"WRAPPER ERROR: Invalid grid coordinates {grid_coords} for grid shape {self._coverage_grid.shape}: {e}")
            # Fallback: don't update grid coverage for this step
            pass
        
        # Hash-based unique state counting (more precise)
        rounded_emb = np.round(emb.flatten(), decimals=3)
        state_hash = hash(tuple(rounded_emb.tolist()))

        was_new_state = False
        if state_hash not in self._visited_states:
            self._visited_states.add(state_hash)
            was_new_state = True
            
        self._visited_states_per_subtask[self._subtask].add(state_hash)
        
        # Calculate current coverage metrics
        grid_coverage = np.sum(self._coverage_grid > 0) / self._coverage_grid.size
        unique_state_ratio = len(self._visited_states) / max(self._total_steps, 1)
        
        # Debug print for first few steps
        if self._total_steps <= 10:
            print(f"WRAPPER DEBUG Step {self._total_steps}: "
                  f"grid_coords={grid_coords}, was_new_cell={was_new_cell}, "
                  f"grid_coverage={grid_coverage:.4f}, unique_states={len(self._visited_states)}, "
                  f"similarity_fitted={self._similarity_mapping_fitted}")
        
        # Periodic status updates
        if self._total_steps % 5000 == 0:
            print(f"WRAPPER STATUS Step {self._total_steps}: "
                  f"Grid coverage: {grid_coverage:.1%}, "
                  f"Unique states: {len(self._visited_states)}, "
                  f"Similarity mapping: {'fitted' if self._similarity_mapping_fitted else 'not fitted'}")
        
        # Store metrics
        self._coverage_history.append({
            'step': self._total_steps,
            'subtask': self._subtask,
            'grid_coverage': grid_coverage,
            'unique_states': len(self._visited_states),
            'unique_ratio': unique_state_ratio,
            'total_grid_cells': np.sum(self._coverage_grid > 0)
        })
        
    def _compute_intrinsic_reward(self, emb):
        current_subtask = self._subtask
        memory = self._embedding_memory_per_subtask[current_subtask]

        if len(memory) >= self._max_memory:
            memory.pop(0)
        memory.append(emb.copy())

        # If not enough samples yet → encourage exploration
        if len(memory) <= self._k_nearest:
            novelty_reward = 1.0
        else:
            # Calculate distances to all memories except the just added one
            dists = [np.linalg.norm(emb - mem) for mem in memory[:-1]]
            
            # Sort distances and get the k nearest
            sorted_dists = sorted(dists)
            k = min(self._k_nearest, len(sorted_dists))
            k_nearest_dists = sorted_dists[:k]
            
            # Compute mean of k nearest distances as the novelty reward
            mean_dist = np.mean(k_nearest_dists)
            novelty_reward = min(mean_dist, 10.0)

        # Track novelty history
        self._novelty_history.append({
            'step': self._total_steps,
            'subtask': current_subtask,
            'novelty_reward': float(novelty_reward)
        })

        return novelty_reward


    def get_coverage_stats(self):
        """Get comprehensive coverage statistics"""
        if not self._coverage_history:
            return {}
        
        latest = self._coverage_history[-1]
        
        stats = {
            'total_steps': self._total_steps,
            'unique_states_visited': len(self._visited_states),
            'grid_coverage_percentage': latest['grid_coverage'] * 100,
            'unique_state_ratio': latest['unique_ratio'],
            'current_subtask': self._subtask,
            'average_novelty': np.mean([h['novelty_reward'] for h in self._novelty_history[-100:]]) if self._novelty_history else 0,
            'similarity_mapping_fitted': self._similarity_mapping_fitted,
            'using_similarity_grid': self._use_similarity_grid,
        }
        
        # Add grid shape information
        if self._coverage_grid is not None:
            stats['grid_shape'] = list(self._coverage_grid.shape)
            stats['total_grid_cells'] = self._coverage_grid.size
            stats['visited_grid_cells'] = int(np.sum(self._coverage_grid > 0))
        
        # Per-subtask statistics
        subtask_stats = {}
        for s in range(self._num_subtasks + 1):
          subtask_stats[f'subtask_{s}_coverage'] = (
              [h['grid_coverage'] for h in self._coverage_history if h['subtask'] == s][-1] * 100
              if any(h['subtask'] == s for h in self._coverage_history) else 0
          )
          subtask_stats[f'subtask_{s}_unique_states'] = len(self._visited_states_per_subtask[s])
        stats.update(subtask_stats)
        return stats
    
    def get_coverage_visualization_data(self):
        """Get data for visualizing the coverage grid"""
        if self._coverage_grid is None:
            return None
            
        viz_data = {
            'coverage_grid': self._coverage_grid.copy(),
            'grid_shape': self._coverage_grid.shape,
            'total_cells': self._coverage_grid.size,
            'visited_cells': int(np.sum(self._coverage_grid > 0)),
            'coverage_percentage': (np.sum(self._coverage_grid > 0) / self._coverage_grid.size) * 100,
            'similarity_mapping_fitted': self._similarity_mapping_fitted,
            'using_similarity_grid': self._use_similarity_grid,
        }
        
        if self._similarity_mapping_fitted and hasattr(self, '_projection_matrix'):
            viz_data['projection_bounds'] = {
                'min': self._projection_min.tolist(),
                'max': self._projection_max.tolist()
            }
            
        return viz_data
      
    def save_coverage_data(self, filepath):
        """Save coverage data for analysis"""
        import json
        import numpy as np

        def _to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        def _clean_dict_list(dict_list):
            return [{k: _to_serializable(v) for k, v in d.items()} for d in dict_list]

        coverage_data = {
            'coverage_history': _clean_dict_list(self._coverage_history),
            'novelty_history': _clean_dict_list(self._novelty_history),
            'stats': {k: _to_serializable(v) for k, v in self.get_coverage_stats().items()},
            'visualization_data': {k: _to_serializable(v) for k, v in (self.get_coverage_visualization_data() or {}).items()},
            'parameters': {
                'k_nearest': int(self._k_nearest),
                'intrinsic_scale': float(self._intrinsic_scale),
                'max_memory': int(self._max_memory),
                'coverage_grid_size': int(self._coverage_grid_size),
                'use_similarity_grid': bool(self._use_similarity_grid),
                'grid_dims': int(self._grid_dims) if hasattr(self, '_grid_dims') else 2,
            }
        }

        with open(filepath, 'w') as f:
            json.dump(coverage_data, f, indent=2)
            
        print(f"WRAPPER: Saved coverage data to {filepath}")
        
        # Also save a simple coverage grid visualization file
        if self._coverage_grid is not None:
            grid_file = filepath.replace('.json', '_grid.npy')
            np.save(grid_file, self._coverage_grid)
            print(f"WRAPPER: Saved coverage grid to {grid_file}")

        
    def reset_state(self):
        print("WRAPPER: Resetting HOLDRLearnedVisualReward state.")
        # Reset subtask tracking
        self._subtask = 0
        self._subtask_solved_counter = 0
        
        # Optionally reset coverage tracking (uncomment if you want fresh coverage each episode)
        # self._coverage_grid = np.zeros_like(self._coverage_grid) if self._coverage_grid is not None else None
        # self._visited_states = set()
        # self._visited_states_per_subtask = defaultdict(set)
        # self._embedding_memory = []
        # self._total_steps = 0
        # self._unique_states_visited = 0
        # print("WRAPPER: Also reset coverage tracking.")
       
    def _compute_embedding_distance(self, emb, goal_emb, subtask_idx):
        dist = np.linalg.norm(emb - goal_emb)
        #dist *= self._distance_scale[subtask_idx]
        dist = self._distance_reward(dist)
        return dist
    
    def _distance_reward(self, d, alpha=0.001, beta=0.01, gamma=1e-3):
      return -alpha * d**2 - beta * np.sqrt(d**2 + gamma)
    
    def _check_subtask_completion(self, dist, current_reward):
      if self._subtask == 0:
        if dist > -0.03:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = self._subtask + 1
                self._subtask_solved_counter = 0
                self.subtask_switch_step = self.index_seed_step
        else:
            self._subtask_solved_counter = 0
      elif self._subtask == 1:
        if dist > -0.03:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = self._subtask + 1
                self._subtask_solved_counter = 0
                self.subtask_switch_step = self.index_seed_step
        else:
            self._subtask_solved_counter = 0
      elif self._subtask == 2:
        if dist > -0.06:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = self._subtask + 1
                self._subtask_solved_counter = 0
                self.subtask_switch_step = self.index_seed_step
        else:
            self._subtask_solved_counter = 0
            
      elif self._subtask == 3:
        if dist > -0.03:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = self._subtask + 1
                self._subtask_solved_counter = 0
                self.subtask_switch_step = self.index_seed_step
        else:
            self._subtask_solved_counter = 0
      elif self._subtask == 4:
        if dist > -0.03:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = self._subtask + 1
                self._subtask_solved_counter = 0
                self.subtask_switch_step = self.index_seed_step
        else:
            self._subtask_solved_counter = 0
      elif self._subtask == 5:
        if dist > -0.06:
            self._subtask_solved_counter += 1
            if self._subtask_solved_counter >= self._subtask_hold_steps:
                self._subtask = self._subtask + 1
                self._subtask_solved_counter = 0
                self.subtask_switch_step = self.index_seed_step
        else:
            self._subtask_solved_counter = 0
            
            
    def _get_reward_from_image(self, image, flag):
        image_tensor = self._to_tensor(image)
        emb = self._model.infer(image_tensor).numpy().embs  # Shape: (emb_dim,)
        # emb = self._model.module.infer(image_tensor).numpy().embs

        if flag == "train":
            self._update_coverage_metrics(emb)

        if self._subtask >= self._num_subtasks:
            reward = self._subtask_cost * self._subtask
            # print(f"WRAPPER- Step:{self.index_seed_step}, reward: {reward}, subtask: {self._subtask}")
        else:        
            goal_emb = self._subtask_means[self._subtask]
            dist = self._compute_embedding_distance(emb, goal_emb, self._subtask)
        
            bonus_reward = self._subtask * self._subtask_cost
            reward = dist + bonus_reward
                    
            # Check if the subtask is completed
            self._check_subtask_completion(dist, reward)
        
            # print(f"WRAPPER- Step:{self.index_seed_step}, reward: {reward}, subtask: {self._subtask}. distance: {dist}")
            
        intrinsic_bonus = self._compute_intrinsic_reward(emb)
        if self.subtask_switch_step > 0 and (self.index_seed_step - self.subtask_switch_step) < 7:
            reward += (self.increase_intrinsic_scale_after_subtask + self._intrinsic_scale) * intrinsic_bonus
        else:
            reward += self._intrinsic_scale * intrinsic_bonus
        return reward
      


    def step(self, action, rank, exp_dir, flag):
        obs, env_reward, done, info = self.env.step(action)
        info["env_reward"] = env_reward
        pixels = self._render_obs()
        learned_reward = self._get_reward_from_image(pixels, flag)
        
        if self.index_seed_step % 20000 == 0 and flag == "train":
            coverage_stats = self.get_coverage_stats()
            info['coverage_stats'] = coverage_stats
            if rank == 0:
                self.save_coverage_data('coverage_analysis.json')

                if self._coverage_grid is not None and self._coverage_grid.ndim == 2:
                    try:
                        import matplotlib.pyplot as plt
                        # Create directory if it does not exist
                        coverage_dir = os.path.join(exp_dir, 'grid_coverage')
                        os.makedirs(coverage_dir, exist_ok=True)

                        plt.figure(figsize=(8, 8))
                        plt.imshow(self._coverage_grid, cmap='Blues', origin='lower')
                        plt.title(f'Coverage Grid at Step {self.index_seed_step}\n'
                                  f'Coverage: {coverage_stats.get("grid_coverage_percentage", 0):.1f}% '
                                  f'({coverage_stats.get("visited_grid_cells", 0)}/{coverage_stats.get("total_grid_cells", 0)} cells)')
                        plt.xlabel('Grid X (Projected Embedding Dim 1)')
                        plt.ylabel('Grid Y (Projected Embedding Dim 2)')
                        plt.colorbar(label='Visited (1) / Not Visited (0)')

                        viz_file = os.path.join(coverage_dir, f'coverage_grid_step_{self.index_seed_step}.png')
                        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"Saved coverage visualization to {viz_file}")
                    except ImportError:
                        pass  # matplotlib not available

        return obs, learned_reward, done, info


      
class REDSLearnedVisualReward(LearnedVisualReward):
    """Replace the environment reward with the output of a REDS model."""

    def __init__(
        self,
        **base_kwargs,
    ):
        """Constructor."""
        super().__init__(**base_kwargs)
        self.text_phrases = ["The robot picks the red block", "The robot push the red block in the green zone","The robot picks the blue block","The robot push the blue block in the green zone","The robot picks the yellow block", "The robot push the yellow block in the green zone","All the blocks are in the green zone in the correct order"]
        self.text_features = []
        for phrase in self.text_phrases:
          # Pass as a batch of 1 video, 1 phrase
          text_feature_list = self._model.encode_text([[phrase]])
          # text_feature_list is a list of 1 tensor of shape (1, D)
          text_feature = text_feature_list[0][0]  # shape: (D,)
          self.text_features.append(text_feature)
        self.text_features = torch.stack(self.text_features, dim=0).to(self._device)
        
    def cos_sim(self, x1, x2):
        normed_x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
        normed_x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
        return torch.matmul(normed_x1, normed_x2.T)
    
    def text_score(self, image_features, text_features, logit=1.0):
        return (self.cos_sim(image_features, text_features) + 1) / 2 * logit
      
    def _get_reward_from_image(self, image):
        if isinstance(image, np.ndarray):
            # Convert to torch tensor and permute to (C, H, W)
            image = torch.from_numpy(image).float().permute(2, 0, 1)  # (C, H, W)
            image = image / 255.0  # normalize if needed

        # Add batch and time dimensions: (1, 1, C, H, W)
        image = image.unsqueeze(0).unsqueeze(0)
        image = image.to(self._device)
        # print("Computing reward from image.")
        """Forward the pixels through the model and compute the reward."""
        image_features = self._model.encode_video(image)
        cont_matrix = self.text_score(image_features, self.text_features)
        diag_cont_matrix = cont_matrix[0]

        N = self.text_features.shape[0]
        eps = 5e-2
        bias = torch.linspace(eps * (N - 1), 0.0, N, device=diag_cont_matrix.device)
        diag_cont_matrix += bias
        target_text_indices = torch.argmax(diag_cont_matrix).item()
        task_embedding = self.text_features[target_text_indices]
        if image_features.dim() == 3:
            image_features = image_features.squeeze(0)  # (1, D)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)  # (1, D)
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0)  # (1, D)
        reward = self._model.predict_reward([image_features], [task_embedding])
        return reward[0].item() if reward[0].numel() == 1 else reward[0]
        


  

