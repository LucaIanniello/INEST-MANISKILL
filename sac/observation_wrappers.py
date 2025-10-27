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

"""Observation wrappers for handling complex observation spaces."""

import numpy as np
import gymnasium as gym
from collections import OrderedDict
import torch


class FlattenObservation(gym.ObservationWrapper):
    """Wrapper to flatten complex observation spaces into a single numpy array.
    
    This wrapper handles:
    - Nested dictionaries 
    - Mixed numpy arrays and tensors
    - Object-dtype arrays containing nested structures
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get a sample observation to determine the flattened size
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gymnasium reset API that returns (obs, info)
            
        flattened_obs = self._flatten_observation(obs)
        
        # Update the observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flattened_obs.shape,
            dtype=np.float32
        )
    
    def observation(self, observation):
        """Flatten the observation into a numpy array."""
        return self._flatten_observation(observation)
    
    def _flatten_observation(self, obs):
        """Recursively flatten observations to a numpy array."""
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.object_:
                # Handle object arrays containing nested structures
                return self._flatten_object_array(obs)
            else:
                return obs.flatten().astype(np.float32)
        elif isinstance(obs, torch.Tensor):
            return obs.detach().cpu().numpy().flatten().astype(np.float32)
        elif isinstance(obs, (dict, OrderedDict)):
            return self._flatten_dict(obs)
        elif isinstance(obs, (list, tuple)):
            return self._flatten_sequence(obs)
        elif isinstance(obs, (int, float, bool, np.number)):
            return np.array([float(obs)], dtype=np.float32)
        else:
            # Try to convert to array and flatten
            try:
                return np.array(obs).flatten().astype(np.float32)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not flatten observation of type {type(obs)}: {e}")
                return np.array([0.0], dtype=np.float32)  # Return a default value
    
    def _flatten_object_array(self, obj_array):
        """Flatten an object array containing nested structures."""
        flattened_parts = []
        
        # Try to iterate through the object array
        try:
            if obj_array.ndim == 0:
                # Single object
                flattened_parts.append(self._flatten_observation(obj_array.item()))
            else:
                # Array of objects
                for item in obj_array.flat:
                    flattened_parts.append(self._flatten_observation(item))
        except Exception as e:
            print(f"Error processing object array: {e}")
            # If we can't process it, return a default array
            return np.array([0.0], dtype=np.float32)
        
        if flattened_parts:
            return np.concatenate(flattened_parts)
        else:
            return np.array([], dtype=np.float32)
    
    def _flatten_dict(self, obs_dict):
        """Flatten a dictionary of observations."""
        flattened_parts = []
        
        # Sort keys for consistent ordering
        for key in sorted(obs_dict.keys()):
            value = obs_dict[key]
            flattened_value = self._flatten_observation(value)
            if flattened_value.size > 0:
                flattened_parts.append(flattened_value)
        
        if flattened_parts:
            return np.concatenate(flattened_parts)
        else:
            return np.array([], dtype=np.float32)
    
    def _flatten_sequence(self, obs_seq):
        """Flatten a sequence (list/tuple) of observations."""
        flattened_parts = []
        
        for item in obs_seq:
            flattened_value = self._flatten_observation(item)
            if flattened_value.size > 0:
                flattened_parts.append(flattened_value)
        
        if flattened_parts:
            return np.concatenate(flattened_parts)
        else:
            return np.array([], dtype=np.float32)


class FixManiSkillObservation(gym.ObservationWrapper):
    """Specific wrapper for ManiSkill environments that may return object arrays."""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Test the observation space by getting a sample observation
        try:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
        except Exception as e:
            print(f"Warning: Could not sample observation during wrapper initialization: {e}")
            # Fallback to a default observation space
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(48,),  # ManiSkill StackCube typically has 48-dim observations
                dtype=np.float32
            )
            return
            
        # Process the sample observation to determine the correct space
        flattened_obs = self._process_observation(obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flattened_obs.shape,
            dtype=np.float32
        )
        print(f"FixManiSkillObservation: Created observation space with shape {flattened_obs.shape}")
    
    def observation(self, observation):
        """Process the observation to ensure it's a proper numpy array."""
        try:
            processed_obs = self._process_observation(observation)
            
            # Final checks
            if not isinstance(processed_obs, np.ndarray):
                print(f"ERROR: Processed observation is not numpy array: {type(processed_obs)}")
                processed_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            elif processed_obs.dtype == np.object_:
                print(f"ERROR: Observation wrapper returning object dtype! Falling back to zeros.")
                processed_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            elif processed_obs.shape != self.observation_space.shape:
                print(f"ERROR: Observation shape mismatch. Expected {self.observation_space.shape}, got {processed_obs.shape}")
                # Try to reshape or pad/truncate
                if processed_obs.size >= np.prod(self.observation_space.shape):
                    processed_obs = processed_obs.flatten()[:np.prod(self.observation_space.shape)].reshape(self.observation_space.shape)
                else:
                    print(f"Not enough elements to fill expected shape, padding with zeros")
                    padded = np.zeros(self.observation_space.shape, dtype=np.float32)
                    padded[:min(processed_obs.size, padded.size)] = processed_obs.flatten()[:min(processed_obs.size, padded.size)]
                    processed_obs = padded
            
            return processed_obs.astype(np.float32)
            
        except Exception as e:
            print(f"ERROR in observation wrapper: {e}")
            print(f"Original observation type: {type(observation)}")
            if hasattr(observation, 'shape'):
                print(f"Original observation shape: {observation.shape}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _process_observation(self, obs):
        """Process observation to handle ManiSkill-specific issues."""
        print(f"Processing observation: type={type(obs)}")
        
        # First check if it's a list/tuple containing mixed types (the actual issue)
        if isinstance(obs, (list, tuple)):
            print(f"Observation is a {type(obs).__name__} with length {len(obs)}")
            return self._handle_mixed_list(obs)
        elif isinstance(obs, np.ndarray):
            print(f"Observation is numpy array: shape={obs.shape}, dtype={obs.dtype}")
            if obs.dtype == np.object_:
                # This is the problematic case - object array
                print("Processing object array...")
                return self._handle_object_array(obs)
            else:
                # Normal array, ensure it's float32
                print("Normal array, converting to float32")
                return obs.astype(np.float32)
        elif isinstance(obs, torch.Tensor):
            print(f"Observation is torch tensor: shape={obs.shape}")
            return obs.detach().cpu().numpy().astype(np.float32)
        else:
            print(f"Observation is other type: {type(obs)}, trying fallback flattening")
            # Try the standard flattening approach
            return self._flatten_complex_obs(obs)
    
    def _handle_mixed_list(self, obs_list):
        """Handle mixed list/tuple containing arrays and dicts."""
        print(f"Processing mixed list with {len(obs_list)} items")
        flattened_values = []
        
        for i, item in enumerate(obs_list):
            print(f"  Item {i}: type={type(item)}")
            if isinstance(item, np.ndarray):
                print(f"    Array shape={item.shape}, dtype={item.dtype}")
                if item.dtype != np.object_:
                    # This is a regular numpy array
                    flat_item = item.flatten().astype(np.float32)
                    print(f"    Adding {flat_item.size} elements from array")
                    flattened_values.append(flat_item)
                else:
                    # Recursively handle object arrays within the list
                    print(f"    Object array detected, processing recursively")
                    result = self._handle_object_array(item)
                    if result.size > 0:
                        print(f"    Adding {result.size} elements from object array")
                        flattened_values.append(result.flatten())
            elif isinstance(item, torch.Tensor):
                flat_item = item.detach().cpu().numpy().flatten().astype(np.float32)
                print(f"    Adding {flat_item.size} elements from tensor")
                flattened_values.append(flat_item)
            elif isinstance(item, dict):
                # Skip dictionary entries like {'reconfigure': False}
                print(f"    Skipping dictionary: {item}")
                continue
            elif isinstance(item, (int, float, bool, np.number)):
                print(f"    Adding single scalar value: {item}")
                flattened_values.append(np.array([float(item)], dtype=np.float32))
            else:
                print(f"    Unknown item type, skipping: {type(item)}")
                continue
        
        if flattened_values:
            result = np.concatenate(flattened_values)
            print(f"Successfully flattened mixed list to shape: {result.shape}")
            return result
        else:
            print("No valid arrays found in mixed list, returning default")
            return np.array([0.0], dtype=np.float32)

    def _handle_object_array(self, obj_array):
        """Handle numpy object arrays from ManiSkill."""
        try:
            # Try to extract and concatenate all numeric values
            flattened_values = []
            
            def extract_numbers(item):
                if isinstance(item, (np.ndarray, torch.Tensor)):
                    if isinstance(item, torch.Tensor):
                        item = item.detach().cpu().numpy()
                    if item.dtype != np.object_:
                        return item.flatten().astype(np.float32)
                elif isinstance(item, (dict, OrderedDict)):
                    values = []
                    for key in sorted(item.keys()):
                        val = extract_numbers(item[key])
                        if val is not None and val.size > 0:
                            values.append(val)
                    return np.concatenate(values) if values else None
                elif isinstance(item, (list, tuple)):
                    values = []
                    for val in item:
                        extracted = extract_numbers(val)
                        if extracted is not None and extracted.size > 0:
                            values.append(extracted)
                    return np.concatenate(values) if values else None
                elif isinstance(item, (int, float, bool, np.number)):
                    return np.array([float(item)], dtype=np.float32)
                return None
            
            if obj_array.ndim == 0:
                # Single object
                result = extract_numbers(obj_array.item())
            else:
                # Multiple objects
                for item in obj_array.flat:
                    result = extract_numbers(item)
                    if result is not None:
                        flattened_values.append(result)
            
            if isinstance(result, np.ndarray) and result.size > 0:
                return result
            elif flattened_values:
                return np.concatenate(flattened_values)
            else:
                # Fallback
                return np.array([0.0], dtype=np.float32)
                
        except Exception as e:
            print(f"Error handling object array: {e}")
            print(f"Object array shape: {obj_array.shape}, dtype: {obj_array.dtype}")
            if obj_array.size > 0:
                print(f"First item type: {type(obj_array.flat[0])}")
            # Return a default observation
            return np.array([0.0], dtype=np.float32)
    
    def _flatten_complex_obs(self, obs):
        """Fallback flattening for complex observations."""
        if isinstance(obs, (dict, OrderedDict)):
            values = []
            for key in sorted(obs.keys()):
                val = self._process_observation(obs[key])
                if val.size > 0:
                    values.append(val.flatten())
            return np.concatenate(values) if values else np.array([0.0], dtype=np.float32)
        
        elif isinstance(obs, (list, tuple)):
            values = []
            for item in obs:
                val = self._process_observation(item)
                if val.size > 0:
                    values.append(val.flatten())
            return np.concatenate(values) if values else np.array([0.0], dtype=np.float32)
        
        else:
            try:
                return np.array(obs, dtype=np.float32).flatten()
            except:
                return np.array([0.0], dtype=np.float32)