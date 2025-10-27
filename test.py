import os
import json
from pathlib import Path

def sample_rewards_with_final(dataset_root, step=10):
    dataset_root = Path(dataset_root)

    for video_folder in sorted(dataset_root.iterdir()):
        if not video_folder.is_dir() or not video_folder.name.isdigit():
            continue

        video_id = video_folder.name
        reward_file = video_folder / f"{video_id}_actions.json"
        output_file = video_folder / f"{video_id}_sampled_actions.json"

        if not reward_file.exists():
            print(f"⚠️ Missing: {reward_file}")
            continue

        # Load original rewards
        with open(reward_file, "r") as f:
            rewards = json.load(f)

        if not isinstance(rewards, list):
            print(f"❌ Invalid format in {reward_file} — expected a list")
            continue

        # Sample every `step` and add the final one if needed
        sampled = rewards[::step]
        if len(rewards) > 0 and (len(rewards) - 1) % step != 0:
            sampled.append(rewards[-1])

        # Save sampled rewards
        with open(output_file, "w") as f:
            json.dump(sampled, f)

        print(f"✅ {video_id}: saved {len(sampled)} rewards → {output_file.name}")

# Example usage
sample_rewards_with_final("/home/lianniello/allocentric_bad_trajectory/frames/train/gripper")
