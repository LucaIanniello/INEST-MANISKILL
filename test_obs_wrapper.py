#!/usr/bin/env python3

"""Test script to verify observation wrapper fixes."""

import sys
sys.path.append('/home/liannello/xirl/xirl_conda')

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from sac.observation_wrappers import FixManiSkillObservation
from utils import make_env

def test_observation_wrapper():
    """Test the observation wrapper with ManiSkill environment."""
    print("Creating ManiSkill environment...")
    
    # Create the environment using our utils function
    env = make_env(
        env_name="StackCube-v1", 
        seed=42,
        add_episode_monitor=False,
        action_repeat=1,
        frame_stack=1
    )
    
    print(f"Environment created. Observation space: {env.observation_space}")
    
    # Test reset
    print("\nTesting environment reset...")
    try:
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
            print(f"Reset returned tuple: obs shape={obs.shape}, dtype={obs.dtype}")
        else:
            obs = reset_result
            print(f"Reset returned obs: shape={obs.shape}, dtype={obs.dtype}")
        
        print(f"Observation range: min={obs.min()}, max={obs.max()}")
        
        # Test if observation can be converted to torch tensor
        try:
            obs_tensor = torch.as_tensor(obs)
            print(f"✓ Successfully converted to tensor: {obs_tensor.shape}")
        except Exception as e:
            print(f"✗ Failed to convert to tensor: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        return False
    
    # Test a few steps
    print("\nTesting environment steps...")
    try:
        for i in range(3):
            action = env.action_space.sample()
            step_result = env.step(action)
            
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            
            print(f"Step {i}: obs shape={next_obs.shape}, dtype={next_obs.dtype}, reward={reward}")
            
            # Test tensor conversion
            try:
                obs_tensor = torch.as_tensor(next_obs)
                print(f"  ✓ Tensor conversion successful")
            except Exception as e:
                print(f"  ✗ Tensor conversion failed: {e}")
                return False
                
            if done:
                print("Episode ended, resetting...")
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    next_obs, info = reset_result
                else:
                    next_obs = reset_result
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        return False
    
    finally:
        env.close()

if __name__ == "__main__":
    success = test_observation_wrapper()
    sys.exit(0 if success else 1)