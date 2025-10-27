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
import pdb

DEFAULT_GOAL_XYHW = (-1.2, 1.16, 0.4, 2.4)
HOME_POSE = (0.0, -0.6)
UP_DOWN_MAG = 0.4
ANG_TOT = 0.05
k_rho      = 0.8
k_alpha    = 3.0
max_omega  = 1.0 

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

def noisy_step(env, action, rho, 
               sigma_far=(0.05, 0.10, 0.0), 
               sigma_near=(0.01, 0.02, 0.0),
               rho_threshold=0.4):
    # interpolate noise scale based on rho
    α = np.clip((rho - rho_threshold) / (1.0 - rho_threshold), 0.0, 1.0)
    # α=1 when rho>=1.0; α=0 when rho<=rho_threshold
    sigma = tuple(α * sf + (1-α) * sn 
                  for sf, sn in zip(sigma_far, sigma_near))
    
    noise = np.random.normal(scale=sigma, size=3)
    noisy_action = action + noise
    # clip to valid ranges: [v, ω, grip] ∈ [0, UP_DOWN_MAG], [−max_ω,max_ω], [−1,1]
    noisy_action[0] = np.clip(noisy_action[0], 0.0, UP_DOWN_MAG)
    noisy_action[1] = np.clip(noisy_action[1], -max_omega, max_omega)
    noisy_action[2] = np.clip(noisy_action[2], -1.0, 1.0)
    
    return env.step(noisy_action), noisy_action


def get_block_positions(states_array):
    state = env.get_state()
    if len(states_array) == 0:
        states_array.append(state)
    block_positions = {}
    for i, color_name in enumerate(["red","blue","yellow"]):
        base = 2 + i*5
        bx, by = state[base], state[base+1]
        r, b, y = state[base+2:base+5]
        if r == 1 and b == 0 and y == 0:
            block_positions["red"] = (bx, by)
        elif r == 0 and b == 1 and y == 0:
            block_positions["blue"] = (bx, by)
        elif r == 0 and b == 0 and y == 1:
            block_positions["yellow"] = (bx, by)
      
    return block_positions
    

def in_goal(goal_lower_center, pos_y, goal_y_max) -> bool:
    return goal_lower_center <= pos_y <= goal_y_max

def move_to(env, target_xy, color, observations_array, actions_array, states_array, reward_array , OPEN_CLOSE_MAG=0.0, tol=0.1, max_steps=400):
    actions = []
    for step in range(max_steps):
        state = env.get_state()
        states_array.append(state)
        robot_x, robot_y = state[0], state[1]
        cos_th, sin_th = state[17], state[18]
        dx = target_xy[0] - robot_x
        dy = target_xy[1] - robot_y
        rho = np.hypot(dx, dy)
    
        phi = np.arctan2(dy, dx)
        theta_r = np.arctan2(sin_th, cos_th) + np.pi/2
        phi_rel = ((phi - theta_r + np.pi) % (2*np.pi)) - np.pi
        
        if rho < tol and abs(phi_rel) < tol:
            for i in range(10):
                action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                actions.append(action)
                actions_array.append(action.tolist())
                obs, rew, done, info = env.step(action)
                observations_array.append(obs)
                reward_array.append(rew)                
                if done:
                    print("Env terminated early.")
                    return actions
            break
        v = np.clip(k_rho * rho, 0.0, UP_DOWN_MAG)
        omega = np.clip(k_alpha * phi_rel, -max_omega, max_omega)
        action = np.array([v, omega, OPEN_CLOSE_MAG], dtype=np.float32)

        (obs, rew, done, info), noisy_act = noisy_step(env, action, rho)
        observations_array.append(obs)
        reward_array.append(rew)
        actions_array.append(noisy_act.tolist())

        # 4) record the *noisy* command
        actions.append(noisy_act)

        if done:
            print("Env terminated early.")
            break

    return actions


def push_to_goal(env, target_x, target_y, goal_y_min, goal_y_max, color, observations_array, actions_array, states_array, reward_array, max_steps=400):
    actions = []
    step = 0    
    while step < (max_steps) and not in_goal(goal_y_min, target_y, goal_y_max):
        block_positions = get_block_positions(states_array=states_array)
        _, target_y = block_positions[color]
        target_xy = (target_x, goal_y_max)
        actions += move_to(env, target_xy, color, observations_array, actions_array, states_array, reward_array, OPEN_CLOSE_MAG=1.0, max_steps=1)
        step += 1
    for step in range(15):
        action = [-1.0, 0.0, 0.0] # Move back
        actions.append(np.array(action, dtype=np.float32))
        obs, rew, done, info = env.step(actions[-1])
        observations_array.append(obs)
        actions_array.append(actions[-1].tolist())
        reward_array.append(rew)
    if done:
        print("Env terminated early.")
        return actions
    return actions

def record_trajectory(env, actions, save_dir: Path, video_id: str, states_array, render_kwargs=None):
    env.reset()
    count = 0
    frames = []
    subgoal_frame_indices = {}
    recorded_subgoals = {"blue": False, "yellow": False, "red": False}

    # Goal area bounds
    goal_x, goal_y, goal_h, _ = DEFAULT_GOAL_XYHW 
    goal_y_min = goal_y - goal_h
    goal_y_max = goal_y
    goal_center_y = (goal_y_min + goal_y_max) / 2
    goal_lower_center = (goal_center_y + goal_y_min) / 2

    for i, a in enumerate(actions):
        frame = env.render(mode="rgb_array", **(render_kwargs or {}))
        frames.append(frame)

        # Save every 10th frame
        if i % 10 == 0:
            imageio.imwrite(str(save_dir / f"{count}.png"), frame)

            # Check block positions for subgoals
            block_positions = get_block_positions(states_array)
            for color in ["blue", "yellow", "red"]:
                if not recorded_subgoals[color]:
                    _, by = block_positions[color]
                    if in_goal(goal_lower_center, by, goal_y_max):
                        subgoal_frame_indices[color] = str(save_dir / f"{count}.png")
                        recorded_subgoals[color] = True
            count += 1

        env.step(a)

    # Save final frame
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    imageio.imwrite(str(save_dir / f"{count}.png"), frame)

    return frames, subgoal_frame_indices


def generate_video(env, out_path: Path, video_id: str, subgoal_frames_dict: dict, video_path, observations, actions, states,reward,  fps: int = 30):
    env.reset()
    all_actions = []
    frames = []

    goal_x, goal_y, goal_h, _ = DEFAULT_GOAL_XYHW 
    goal_y_min = goal_y - goal_h
    goal_y_max = goal_y
    goal_center_y = (goal_y_min + goal_y_max) / 2
    goal_lower_center = (goal_center_y + goal_y_min) / 2

    starting_block_positions = get_block_positions(states_array=states)

    for color in ["blue", "yellow", "red"]:
        block_positions = get_block_positions(states_array=states)
        bx, by = block_positions[color]
        starting_x, _ = starting_block_positions[color]
        if starting_x == -0.5:
            f_bx = np.random.uniform(starting_x - 0.1, starting_x + 0.1)
        elif starting_x == 0.5:
            f_bx = np.random.uniform(starting_x - 0.1, starting_x + 0.1)
        elif starting_x == 0.0:
            f_bx = np.random.uniform(starting_x - 0.1, starting_x + 0.1)
        all_actions += move_to(env, (bx, by), color, observations_array=observations, actions_array=actions, states_array=states, reward_array=reward)
        all_actions += push_to_goal(env, f_bx, by, goal_lower_center, goal_y_max, color, observations_array=observations, actions_array=actions, states_array=states, reward_array=reward)

    final_block_positions = get_block_positions(states_array=states)
   

    if all(in_goal(goal_y_min, final_block_positions[color][1], goal_y_max) for color in ["blue", "yellow", "red"]):
        save_dir = out_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        frames, subgoal_frames = record_trajectory(env, all_actions, frame_dir, video_id, states_array=states)

        imageio.mimsave(str(out_path), frames, fps=fps)

        # Convert to list with fixed order
        subgoal_frames_dict[video_id] = [
            subgoal_frames.get("blue", -1),
            subgoal_frames.get("yellow", -1),
            subgoal_frames.get("red", -1)
        ]
        
        with open(str(frame_dir / f"{video_id}_observations.json"), "w") as f:
            json.dump(to_serializable(observations), f)
        with open(str(frame_dir / f"{video_id}_actions.json"), "w") as f:
            json.dump(to_serializable(actions), f)
        with open(str(frame_dir / f"{video_id}_rewards.json"), "w") as f:
            json.dump(to_serializable(rewards), f)
        with open(str(frame_dir / f"{video_id}_states.json"), "w") as f:
            json.dump(to_serializable(states), f)

        return True
    else:
        return False



if __name__ == "__main__":
    root_dir = Path("/home/lianniello/egocentric_bad_trajectory")
    videos_root = root_dir / "videos"
    frames_root = root_dir / "frames" / "train" / "gripper"
    subgoal_frames_dict = {}
    
    observations = []
    actions = []
    states = []
    rewards = []
    

    colors_set = [en.ShapeColor.RED, en.ShapeColor.BLUE, en.ShapeColor.YELLOW]


    for i in range(1100,1105):
        video_id = f"{i}"
        # random.shuffle(colors_set)
        # if colors_set[1] == en.ShapeColor.BLUE:
        #     colors_set[1] = colors_set[2]
        #     colors_set[2] = en.ShapeColor.BLUE
        
        observations = []
        actions = []
        states = []
        rewards = []
    
        env = SweepToTopEnv(
            robot_cls=NonHolonomicGripperEmbodiment,
            use_state=True, 
            colors_set=colors_set,
        )

        print(f"Generating video {video_id}...")
        print(f"Colors: {colors_set}")

        video_dir = videos_root / video_id
        frame_dir = frames_root / video_id
        video_path = video_dir / f"{video_id}.mp4"

        # Make sure frame and video dirs exist
        frame_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        success = generate_video(env, video_path, video_id, subgoal_frames_dict, video_dir, observations, actions, states, rewards)

        if success:
            file_size = os.path.getsize(video_path)
            if file_size > 130 * 1024:
                print(f"Video {video_path} exceeds size limit ({file_size / 1024:.2f} KB). Removing and retrying...")
                success = False

        while not success:
            print(f"Failed to generate valid video for {video_id}. Cleaning up...")

            # Clean contents of video directory
            if video_dir.exists():
                for file in video_dir.iterdir():
                    file.unlink()

            # Clean contents of frame directory
            if frame_dir.exists():
                for file in frame_dir.glob("*.png"):
                    file.unlink()


            # Remove any previous entry in subgoal dict
            subgoal_frames_dict.pop(video_id, None)
            # Remove previously generated JSON files
            for suffix in ["observations", "actions", "rewards", "states"]:
                json_file = video_dir / f"{video_id}_{suffix}.json"
                if json_file.exists():
                    json_file.unlink()

            observations.clear()
            actions.clear()
            states.clear()
            rewards.clear()
            # Retry
            success = generate_video(env, video_path, video_id, subgoal_frames_dict, video_dir, observations, actions, states, rewards)

            if success:
                file_size = os.path.getsize(video_path)
                if file_size > 130 * 1024:
                    print(f"Video {video_path} exceeds size limit ({file_size / 1024:.2f} KB). Removing and retrying...")
                    if video_path.exists():
                        os.remove(video_path)
                    success = False

    # Save subgoal dictionary
    with open(root_dir / "subgoal_frames.json", "w") as f:
        json.dump(subgoal_frames_dict, f, indent=2)
        

        
    
