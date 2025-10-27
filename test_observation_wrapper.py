#!/usr/bin/env python3

import gymnasium as gym
import mani_skill.envs
import numpy as np
from sac.observation_wrappers import FixManiSkillObservation

def test_observation_wrapper():
    """Test the observation wrapper to see what kinds of observations we get."""
    print("Creating ManiSkill StackCube environment...")
    
    # Create the environment
    env = gym.make(
        "StackCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array"  # Changed from "human" to avoid GUI issues
    )
    
    print("Environment created. Getting sample observations...")
    
    # Get a few sample observations
    for i in range(3):
        print(f"\n--- Reset {i+1} ---")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, info = obs
            print(f"Reset returned tuple: obs type={type(obs)}, info type={type(info)}")
        else:
            print(f"Reset returned: {type(obs)}")
        
        print(f"Observation type: {type(obs)}")
        if isinstance(obs, np.ndarray):
            print(f"Observation dtype: {obs.dtype}, shape: {obs.shape}")
            if obs.dtype == np.object_:
                print(f"Object array contents (first few items): {obs.flat[:3] if obs.size > 0 else 'empty'}")
        elif isinstance(obs, (list, tuple)):
            print(f"Observation is {type(obs).__name__} with length: {len(obs)}")
            for j, item in enumerate(obs[:3]):  # Show first 3 items
                print(f"  Item {j}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}, dtype={getattr(item, 'dtype', 'N/A')}")
        else:
            print(f"Observation: {obs}")
    
    print("\n--- Testing with wrapper ---")
    wrapped_env = FixManiSkillObservation(env)
    
    for i in range(3):
        print(f"\nWrapped Reset {i+1}:")
        obs = wrapped_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        print(f"Wrapped observation type: {type(obs)}, dtype: {obs.dtype}, shape: {obs.shape}")
        print(f"Observation space: {wrapped_env.observation_space}")
    
    print("\nTesting step function:")
    action = wrapped_env.action_space.sample()
    result = wrapped_env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        print(f"Step observation type: {type(obs)}, dtype: {obs.dtype}, shape: {obs.shape}")
    else:
        obs, reward, done, info = result
        print(f"Step observation type: {type(obs)}, dtype: {obs.dtype}, shape: {obs.shape}")

if __name__ == "__main__":
    test_observation_wrapper()