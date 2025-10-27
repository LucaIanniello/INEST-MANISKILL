import os
import json
from glob import glob

# === CONFIGURATION ===
root_dirs = [
    "/home/lianniello/egocentric_dataset/frames/train/gripper",
    "/home/lianniello/egocentric_dataset/frames/valid/gripper"
]
subtask_file = "/home/lianniello/egocentric_dataset/subgoal_frames.json"

# === PHRASES FOR EACH STAGE ===
phrases = [
    "The robot moves the red block in the goal zone",
    "The robot moves the blue block in the goal zone",
    "The robot moves the yellow block in the goal zone"
]

# === LOAD SUBTASK SWITCHES ===
with open(subtask_file, 'r') as f:
    subtask_data = json.load(f)

# === REVERSE MAP: video_id → switch frame indices ===
switch_dict = {}
for entry in subtask_data.values():
    if not entry:
        continue
    sample_path = entry[0]
    video_id = os.path.normpath(sample_path).split(os.sep)[-2]  # e.g., "0"
    switch_dict[video_id] = sorted(
        [int(os.path.splitext(os.path.basename(p))[0]) for p in entry]
    )

# === PROCESS ALL VIDEO DIRECTORIES ===
for root in root_dirs:
    for video_dir in sorted(os.listdir(root)):
        full_path = os.path.join(root, video_dir)
        if not os.path.isdir(full_path):
            continue

        video_id = video_dir
        if video_id not in switch_dict:
            print(f"Warning: No switch data for video {video_id}, skipping.")
            continue

        switch_points = switch_dict[video_id]

        # Get frame indices
        frame_files = sorted(glob(os.path.join(full_path, "*.png")))
        frame_indices = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in frame_files])

        text_array = []
        for idx in frame_indices:
            if idx < switch_points[0]:
                text_array.append(phrases[0])
            elif idx < switch_points[1]:
                text_array.append(phrases[1])
            else:
                text_array.append(phrases[2])

        # Save output in the same video directory
        output_path = os.path.join(full_path, f"{video_id}_text.json")
        with open(output_path, 'w') as out_f:
            json.dump(text_array, out_f, indent=2)

        print(f"Saved: {output_path}")
