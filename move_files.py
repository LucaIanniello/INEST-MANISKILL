import os
import shutil

# Root directories
source_root = "/home/lianniello/new_env_dataset/frames"
target_root = "/home/lianniello/egocentric_dataset/frames"

# File suffixes to copy
suffixes = ["_observations.json", "_actions.json", "_states.json", "_rewards.json"]

# Process both train and valid splits
for split in ["train/gripper", "valid/gripper"]:
    source_split = os.path.join(source_root, split)
    target_split = os.path.join(target_root, split)

    for video_id in os.listdir(target_split):
        if not video_id.isdigit():
            continue

        video_source_path = os.path.join(source_split, video_id)
        video_target_path = os.path.join(target_split, video_id)

        for suffix in suffixes:
            filename = f"{video_id}{suffix}"
            source_file = os.path.join(video_source_path, filename)
            target_file = os.path.join(video_target_path, filename)

            if os.path.exists(source_file):
                shutil.copy2(source_file, target_file)
                print(f"Copied {filename} to {target_file}")
            else:
                print(f"Missing: {source_file}")
