import math
import numpy as np
import imageio
from pathlib import Path
import os


from sweepToTop import SweepToTopEnv 
import xmagical.entities as en 
from xmagical.entities.embodiments.gripper import NonHolonomicGripperEmbodiment
import random
import json

def get_block_positions_and_colors(state):
    block_positions = {}
    colors_set = [None, None, None]  
    canonical_positions = [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.0)]
    color_enum = {"red": en.ShapeColor.RED, "blue": en.ShapeColor.BLUE, "yellow": en.ShapeColor.YELLOW}

    for i, color_name in enumerate(["red", "blue", "yellow"]):
        base = 2 + i*5
        bx, by = state[base], state[base+1]
        r, b, y = state[base+2:base+5]
        if r == 1 and b == 0 and y == 0:
            block_positions["red"] = (bx, by)
            color = "red"
        elif r == 0 and b == 1 and y == 0:
            block_positions["blue"] = (bx, by)
            color = "blue"
        elif r == 0 and b == 0 and y == 1:
            block_positions["yellow"] = (bx, by)
            color = "yellow"
        
    # print("Block positions:", block_positions)

    for color, pos in block_positions.items():
        if color == "red":
            if pos[0] == -0.5:
                colors_set[0] = color_enum[color]
            elif pos[0] == 0.0:
                colors_set[1] = color_enum[color]
            elif pos[0] == 0.5:
                colors_set[2] = color_enum[color]
        if color == "blue":
            if pos[0] == -0.5:
                colors_set[0] = color_enum[color]
            elif pos[0] == 0.0:
                colors_set[1] = color_enum[color]
            elif pos[0] == 0.5:
                colors_set[2] = color_enum[color]
        if color == "yellow":
            if pos[0] == -0.5:
                colors_set[0] = color_enum[color]
            elif pos[0] == 0.0:
                colors_set[1] = color_enum[color]
            elif pos[0] == 0.5:
                colors_set[2] = color_enum[color]

    return block_positions, colors_set
            

def replicate_actions(env, action_file, save_dir: Path):
    count = 0
    with open(action_file, 'r') as f:
        actions = json.load(f)
    env.reset()
    frames = []

    for i, a in enumerate(actions):
        if i % 10 == 0:
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            imageio.imwrite(str(save_dir / f"{count}.png"), frame)
            count += 1
        env.step(a)
        
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    imageio.imwrite(str(save_dir / f"{count}.png"), frame)

    return frames


if __name__ == "__main__":
    root_dir = Path("/home/lianniello/egocentric_bad_trajectory")
    videos_root = root_dir / "videos"
    train_ego_root = root_dir / "frames" / "train" / "gripper"
    # validation_ego_root = root_dir / "frames" / "valid" / "gripper"
    
    train_allo_root = Path("/home/lianniello/allocentric_bad_trajectory/frames/train/gripper")
    # validation_allo_root = Path("/home/lianniello/new_env_dataset/frames/valid/gripper")
    
    for repo_dir in train_allo_root.iterdir():
        colors_set = []
        repo_name = repo_dir.name  
        # print(f"Processing {repo_name}...") 
        state_path = train_allo_root / repo_name / f"{repo_name}_states.json"
        action_file = train_allo_root / repo_name / f"{repo_name}_actions.json"
        with open(state_path, 'r') as f:
            states = json.load(f)
        
        initial_state = states[0]
        # print(f"Initial state for {repo_name}: {initial_state}")
        _, colors_set = get_block_positions_and_colors(initial_state)
        # print(f"Colors set for {repo_name}: {colors_set}")
        
        env = SweepToTopEnv(
            robot_cls=NonHolonomicGripperEmbodiment,
            use_state=True, 
            colors_set=colors_set,
        )

        print(f"Generating video {repo_name}...")

        video_dir = videos_root / repo_name
        frame_dir = train_ego_root / repo_name
        video_path = video_dir / f"{repo_name}.mp4"

        # Make sure frame and video dirs exist
        frame_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)
        frames = replicate_actions(env, action_file, frame_dir)
        imageio.mimwrite(str(video_path), frames, fps=30)
        
    # for repo_dir in validation_allo_root.iterdir():
    #     colors_set = []
    #     repo_name = repo_dir.name  
    #     # print(f"Processing {repo_name}...") 
    #     state_path = validation_allo_root / repo_name / f"{repo_name}_states.json"
    #     action_file = validation_allo_root / repo_name / f"{repo_name}_actions.json"
    #     with open(state_path, 'r') as f:
    #         states = json.load(f)
        
    #     initial_state = states[0]
    #     # print(f"Initial state for {repo_name}: {initial_state}")
    #     _, colors_set = get_block_positions_and_colors(initial_state)
    #     # print(f"Colors set for {repo_name}: {colors_set}")
        
    #     env = SweepToTopEnv(
    #         robot_cls=NonHolonomicGripperEmbodiment,
    #         use_state=True, 
    #         colors_set=colors_set,
    #     )

    #     print(f"Generating video {repo_name}...")

    #     video_dir = videos_root / repo_name
    #     frame_dir = validation_ego_root / repo_name
    #     video_path = video_dir / f"{repo_name}.mp4"

    #     # Make sure frame and video dirs exist
    #     frame_dir.mkdir(parents=True, exist_ok=True)
    #     video_dir.mkdir(parents=True, exist_ok=True)
    #     frames = replicate_actions(env, action_file, frame_dir)
    #     imageio.mimwrite(str(video_path), frames, fps=30)
        
        
        

        