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

"""Teleop the agent and visualize the learned reward.
python interact_reward.py --embodiment Gripper --task MatchRegions [--pretrained_reward] [--dense_reward]
"""

from absl import app
from absl import flags
from configs.constants import EMBODIMENTS, TASKS
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from ml_collections import config_flags
import utils
from xmagical.utils import KeyboardEnvInteractor
import matplotlib.pyplot as plt 
import pdb
from ultralytics import YOLO
import cv2
import glob
import os
from natsort import natsorted  # Import natsort for natural sorting
from PIL import Image, ImageDraw
import random
import string
import numpy as np
import json

from datetime import datetime

FLAGS = flags.FLAGS

flags.DEFINE_enum("task", "sweep_to_top", TASKS,
                  "The agent task.")
flags.DEFINE_enum("embodiment", "gripper", EMBODIMENTS,
                  "The agent embodiment.")
# flags.DEFINE_enum("obs_space", "State", OBSERVATION_SPACE,
#                   "The agent observation space.")
# flags.DEFINE_enum("view", "Allo", VIEW_MODE,
#                   "The agent view.")
# flags.DEFINE_enum("variant", "Demo", VARIANT,
#                   "The agent variant.")

flags.DEFINE_boolean(
    "exit_on_done", True,
    "By default, env will terminate if done is True. Set to False to interact "
    "for as long as you want and press esc key to exit.")

config_flags.DEFINE_config_file(
    "config",
    "base_configs/rl.py",
    "File path to the training hyperparameter configuration.",
)

flags.DEFINE_integer("seed", 0, "RNG seed.")

flags.DEFINE_boolean(
    "use_objdet", False,
    "Use object detector for computing bboxes")

flags.DEFINE_boolean(
    "random_pos", False,
    "Randomize automatically target positions")

flags.DEFINE_string("objdet_path", "./obj_detect/best.pt", "Model path for object detector")

flags.DEFINE_string("env_config", "./env_configs/mr_0.json", "Model path for object detector")


names = {
    0: 'goal', 
    1: 'circle', 
    2: 'penthagon',  # Default pentagon
    3: 'rect', 
    4: 'robot', 
    5: 'star',
}

ordered_names = {1: 'goal', 2: 'robot', 3: 'star', 4: 'rect', 5: 'penthagon1', 6: 'circle', 7: 'penthagon2'}
# Define the desired order of classes
order = {
    'goal': 1,
    'robot': 2,
    'star': 3,
    'rect': 4,
    'penthagon1': 5,
    'circle': 6,
    'penthagon2': 7,
}

considered_objects = [1, 4, 6, 2, 7, 3, 5] # 2, 6 # rect, robot, star # [1,2,4,6]

# Function to get the class name from the label
def get_class_name(label, names):
    return names[int(label)]

# Function to reduce bbox size
def reduce_bbox(bbox, reduction_percent, img_shape):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate the reduction
    reduction_width = width * reduction_percent / 100
    reduction_height = height * reduction_percent / 100

    # Apply the reduction
    new_x1 = max(0, x1 + reduction_width / 2)  # Ensure not going out of bounds
    new_y1 = max(0, y1 + reduction_height / 2)
    new_x2 = min(img_shape[1], x2 - reduction_width / 2)  # img_shape[1] = image width
    new_y2 = min(img_shape[0], y2 - reduction_height / 2)  # img_shape[0] = image height

    return [new_x1, new_y1, new_x2, new_y2]

def reorder_bboxes(bboxes, labels, img):
    # Initialize arrays for ordered bboxes and labels
    ordered_bboxes = []
    ordered_labels = []

    if len(bboxes) != len(ordered_names.keys()):
       print(f"DANGER! Detection fails (detected {len(bboxes)} objects).")

    # Process each bounding box and label
    for bbox, label in zip(bboxes, labels):
        class_name = get_class_name(label, names)

        # Reduce bbox size for specific classes
        if class_name in ['penthagon', 'circle']:
            bbox = reduce_bbox(bbox, reduction_percent=15, img_shape=img.shape)
        elif class_name == 'star':
            bbox = reduce_bbox(bbox, reduction_percent=20, img_shape=img.shape)
        
        if class_name == 'penthagon':
            # Determine the specific pentagon type by color
            center_y = int((bbox[3] + bbox[1]) / 2)
            center_x = int((bbox[2] + bbox[0]) / 2)
            color = img[center_y, center_x]
            
            if np.array_equal(color, [123, 213, 254]):  # penthagon2 color
                class_name = 'penthagon2'
            elif np.array_equal(color, [211, 185, 135]):  # penthagon1 color
                class_name = 'penthagon1'
            else:
                print("Detected wrongly a penthagon, skipping this bbox.")
                pdb.set_trace()
                continue  # Skip this bounding box and move to the next one
        
        # Append to ordered list
        ordered_bboxes.append((bbox, order[class_name]))

    # Sort the bounding boxes based on the desired order
    ordered_bboxes.sort(key=lambda x: x[1])

    # Separate the bboxes and labels
    sorted_bboxes = [item[0] for item in ordered_bboxes]
    sorted_labels = [item[1] for item in ordered_bboxes]

    return np.array(sorted_bboxes), np.array(sorted_labels)

def get_current_date():
	return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

def get_random_string(n=5):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

random_id = get_current_date() + '_' + get_random_string(5)
vid_path = "/tmp/xirl/tests/dataset/" + random_id
os.makedirs(vid_path, exist_ok=True)
processed_vid_path = "/tmp/xirl/tests/processed_dataset/" + random_id
os.makedirs(processed_vid_path, exist_ok=True)
processed_frames = []


# Function to save bounding boxes in a .txt file with consistent order
def save_bboxes(bboxes, labels, output_txt_path, i):
    # Create a dictionary to group bboxes by label
    bbox_dict = {id: bbox[:4] for bbox, id in zip(bboxes, labels)}  # List for each label
    if len(bbox_dict.keys()) != len(ordered_names.keys()):
       print(f"DANGER! Detection fails (detected {len(bbox_dict.keys())} objects) at timestep {i}.")
    # Write to file in consistent order
    with open(output_txt_path, 'w') as f:
        for label_int in bbox_dict.keys():
            if label_int in considered_objects:
              x1, y1, x2, y2 = bbox_dict[label_int]
              f.write(f"{label_int} {x1} {y1} {x2} {y2}\n")
            else:   # No detection for this label
              print(f"DANGER! Detection fails (no detected {ordered_names[int(label_int)]}) in frame path {frame_path}.")


def save_bboxes_from_state(bboxes, output_txt_path):
    # Create a dictionary to group bboxes by label
    bbox_dict = {considered_objects[i]: bboxes[i][:4] for i in range(len(considered_objects))}  # List for each label

    # Write to file in consistent order
    with open(output_txt_path, 'w') as f:
        for label_int in bbox_dict.keys():
            if label_int in considered_objects:
              x1, y1, x2, y2 = bbox_dict[label_int]
              f.write(f"{label_int} {x1} {y1} {x2} {y2}\n")
                
# Function to create GIF from frames
def create_gif_from_frames(frame_folder, output_gif_path):
    frame_files = glob.glob(os.path.join(frame_folder, "tracked_*.png"))
    frame_files = natsorted(frame_files)  # Natural sorting for frame order
    frames = [Image.open(frame_file) for frame_file in frame_files]
    
    if frames:  # Check if any frames were found
        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"GIF created at {output_gif_path}")
    else:
        print("No frames found to create GIF.")

def compute(obs, i):
    model = YOLO(FLAGS.objdet_path)
    obs = np.ascontiguousarray(obs)
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    results = model.predict(obs, iou=0.4, conf=0.20, verbose=True)
    #results = model.track(obs, iou=0.4, conf=0.20, persist=True, verbose=True)
    # Extract bounding boxes and labels
    bboxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    labels = results[0].boxes.cls.cpu().numpy()  # Class labels
    #track_ids = results[0].boxes.id.int().cpu().tolist()
    bboxes, labels = reorder_bboxes(bboxes, labels, obs)
    # Draw the bounding boxes on the frame
    img_with_bboxes = results[0].plot()
    # Normalize bboxes before saving
    norm_bboxes = bboxes/obs.shape[0]

    # Save bounding boxes to .txt (use zfill to match frame naming)
    txt_path = os.path.join(vid_path, f"{str(i[0]).zfill(6)}.txt")
    save_bboxes(norm_bboxes, labels, txt_path, i)

    # Save original frame
    frame_path = os.path.join(vid_path, f"{str(i[0])}.png")
    cv2.imwrite(frame_path, obs)

    # Save processed frame
    processed_frame_path = os.path.join(processed_vid_path, f"{str(i[0])}.png")
    cv2.imwrite(processed_frame_path, img_with_bboxes)
    processed_frames.append(img_with_bboxes)

def convert_coordinates(pos):
      pos[0] = (pos[0] + 1.1) / 2.2 * 384
      pos[1] = 384 - ((pos[1] + 1.1) / 2.2) * 384
      return pos

def state_to_bboxes(obs):
    agent_in_bboxes = True
    distr_in_bboxes = True
    gripper_size = 60   
    target_1_size = 25
    target_2_size = 20
    goal_h = 0.65* 384/ 2 
    goal_w = 0.55* 384/ 2 
    goal_x = 0.1
    goal_y = 0.7
    dist_1_size = 25
    dist_2_size = 25
    dist_3_size = 25

    # Position adjustment offsets for fine-tuning
    gripper_offset = np.array([0, 0])     # Move gripper down
    target_1_offset = np.array([0, 0])  # Move target_1 up-left
    target_2_offset = np.array([0, 0])    # Move target_2 down-right
    goal_offset = np.array([0, 0]) 
    
    state = np.copy(obs)
    bboxes = np.zeros((1, 4))
    #adapt to the env!!!
    gripper_pos = state[:2] 
    target_1_pos = state[2:4] # STAR
    target_2_pos = state[4:6] # RECT
    goal_pos = np.array([goal_x, goal_y])
    dist_1_pos = state[6:8] # PENT (blue)
    dist_2_pos = state[8:10] # CIRC (yellow)
    dist_3_pos = state[10:12] # PENT (yellow)

    gripper_pos = convert_coordinates(gripper_pos) + gripper_offset
    target_1_pos = convert_coordinates(target_1_pos) + target_1_offset # STAR
    target_2_pos = convert_coordinates(target_2_pos) + target_2_offset # RECT
    dist_1_pos = convert_coordinates(dist_1_pos)
    dist_2_pos = convert_coordinates(dist_2_pos)
    dist_3_pos = convert_coordinates(dist_3_pos)

    goal_pos = convert_coordinates(goal_pos) + goal_offset
    
    bboxes[0] = goal_pos[0], goal_pos[1], goal_pos[0] + goal_w, goal_pos[1] + goal_h
    if agent_in_bboxes:
      bboxes = np.append(bboxes, [[gripper_pos[0] - gripper_size, gripper_pos[1] - gripper_size, gripper_pos[0] + gripper_size, gripper_pos[1] + gripper_size]], axis=0)
      
    bboxes = np.append(bboxes, [[target_1_pos[0] - target_1_size, target_1_pos[1] - target_1_size, target_1_pos[0] + target_1_size, target_1_pos[1] + target_1_size]], axis=0)
    bboxes = np.append(bboxes, [[target_2_pos[0] - target_2_size, target_2_pos[1] - target_2_size, target_2_pos[0] + target_2_size, target_2_pos[1] + target_2_size]], axis=0)

    if distr_in_bboxes:
      bboxes = np.append(bboxes, [[dist_1_pos[0] - dist_1_size, dist_1_pos[1] - dist_1_size, dist_1_pos[0] + dist_1_size, dist_1_pos[1] + dist_1_size]], axis=0)
      bboxes = np.append(bboxes, [[dist_2_pos[0] - dist_2_size, dist_2_pos[1] - dist_2_size, dist_2_pos[0] + dist_2_size, dist_2_pos[1] + dist_2_size]], axis=0)
      bboxes = np.append(bboxes, [[dist_3_pos[0] - dist_3_size, dist_3_pos[1] - dist_3_size, dist_3_pos[0] + dist_3_size, dist_3_pos[1] + dist_3_size]], axis=0)
    
    bboxes =  np.reshape(bboxes, (1, len(bboxes), 4)) 
    return bboxes

def vis_bboxes(bboxes, img, state):
    
    # Position adjustment offsets for fine-tuning
    gripper_offset = np.array([0, 0])     # Move gripper down
    target_1_offset = np.array([0, 0])  # Move target_1 up-left
    target_2_offset = np.array([0, 0])    # Move target_2 down-right

    gripper_pos = state[:2] 
    target_1_pos = state[2:4] # STAR
    target_2_pos = state[4:6] # RECT
    dist_1_pos = state[6:8] # PENT (blue)
    dist_2_pos = state[8:10] # CIRC (yellow)
    dist_3_pos = state[10:12] # PENT (yellow)

    gripper_pos = convert_coordinates(gripper_pos) + gripper_offset
    target_1_pos = convert_coordinates(target_1_pos) + target_1_offset
    target_2_pos = convert_coordinates(target_2_pos) + target_2_offset
    dist_1_pos = convert_coordinates(dist_1_pos)
    dist_2_pos = convert_coordinates(dist_2_pos)
    dist_3_pos = convert_coordinates(dist_3_pos)

    # if img:
    #   img = np.ones((384, 384, 3)) * 255
    img = cv2.resize(
          img,
          dsize=(384, 384),
          interpolation=cv2.INTER_CUBIC,
      )
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255), 6: (122, 255, 255)}
    for i in range(bboxes.shape[0]):
      x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]

      img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)

    # img = cv2.circle(img, (int(target_2_pos[0]),int(target_2_pos[1])), radius=0, color=(255, 0, 0), thickness=10)
    # img = cv2.circle(img, (int(gripper_pos[0]),int(gripper_pos[1])), radius=0, color=(0, 255, 0), thickness=10)   
    # img = cv2.circle(img, (int(target_1_pos[0]),int(target_1_pos[1])), radius=0, color=(0, 0, 255), thickness=10)   

    return img

def compute_from_state(image, state, i):
    bboxes_curr = state_to_bboxes(state)[0]
    bboxes_normalized = np.copy(bboxes_curr)
    bboxes_normalized[:, [0, 2]] /= 384  # Normalize x-coordinates (x1 and x2)
    bboxes_normalized[:, [1, 3]] /= 384  # Normalize y-coordinates (y1 and y2)

    img_with_bboxes = vis_bboxes(bboxes_curr, image, state)

    # Save bounding boxes to .txt (use zfill to match frame naming)
    txt_path = os.path.join(vid_path, f"{str(i[0]).zfill(6)}.txt")
    save_bboxes_from_state(bboxes_normalized, txt_path)

    # Save original frame
    frame_path = os.path.join(vid_path, f"{str(i[0])}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Save processed frame
    processed_frame_path = os.path.join(processed_vid_path, f"{str(i[0])}.png")
    cv2.imwrite(processed_frame_path, cv2.cvtColor(img_with_bboxes, cv2.COLOR_RGB2BGR))
    processed_frames.append(img_with_bboxes)

def main(_):
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[FLAGS.embodiment]
  #env_name = f"{FLAGS.task}-{FLAGS.embodiment}-{FLAGS.obs_space}-{FLAGS.view}-{FLAGS.variant}-v0"
  # with open(FLAGS.env_config, 'r') as file:
  #   config = json.load(file)
  # if FLAGS.random_pos:
  #   # Extract all positions into a list
  #   positions = []

  #   for item in config['target']:
  #       positions.append(item['pos'])

  #   for item in config['distractor']:
  #       positions.append(item['pos'])

  #   # Shuffle the positions
  #   random.shuffle(positions)

  #   # Reassign positions to targets and distractors
  #   pos_idx = 0
  #   for item in config['target']:
  #       item['pos'] = positions[pos_idx]
  #       pos_idx += 1

  #   for item in config['distractor']:
  #       item['pos'] = positions[pos_idx]
  #       pos_idx += 1
  
  #env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[FLAGS.embodiment]
  env = utils.make_env(env_name, seed=FLAGS.seed)
  viewer = KeyboardEnvInteractor(action_dim=env.action_space.shape[0])

  obs_state = env.reset()
  obs = env.render("rgb_array")
  viewer.imshow(obs)
  plt.imshow(obs)
  plt.show(block=False)
  # Access the window manager for the plot window
  plot_window = plt.gcf().canvas.manager.window
  # Set the window position using geometry (x, y position)
  plot_window.geometry("+300+300")  # Move the window to position (300, 300)

  plt.pause(0.5)
  

  i = [0]
  if FLAGS.use_objdet:
    compute(obs, i)
  else:
    compute_from_state(obs, obs_state, i)
  rews = []

  def step(action):
    obs_state, rew, done, info = env.step(action)
    rews.append(rew)
    if obs_state.ndim != 3:
      obs = env.render("rgb_array")
          
    if FLAGS.use_objdet:
      compute(obs, i)
    else:
      compute_from_state(obs, obs_state, i)

    if done:
      print(f"Done, score {info['eval_score']:.2f}/1.00")
      print("Episode metrics: ")
      for k, v in info["episode"].items():
        print(f"\t{k}: {v}")
      if FLAGS.exit_on_done:
        return
    i[0] += 1
    return obs

  viewer.run_loop(step)

  gif_output_path = os.path.join(processed_vid_path, "output.gif")
  pil_images = [Image.fromarray(image) for image in processed_frames]
  pil_images[0].save(gif_output_path, save_all=True, append_images=pil_images[1:], duration=100, loop=0)

  utils.plot_reward(rews)

if __name__ == "__main__":
  app.run(main)
