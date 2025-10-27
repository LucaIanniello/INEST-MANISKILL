from typing import Any, Dict, Tuple

import numpy as np
from gym import spaces

import xmagical.entities as en 
from xmagical.entities import EntityIndex
from xmagical.base_env import BaseEnv
import math
import random


DEFAULT_ROBOT_POSE = ((0.0, -0.6), 0.0)
DEFAULT_BLOCK_COLOR = en.ShapeColor.RED
DEFAULT_BLOCK_SHAPE = en.ShapeType.SQUARE
DEFAULT_BLOCK_POSES = [
    ((-0.5, 0.0), 0.0),
    ((0.0, 0.0), 0.0),
    ((0.5, 0.0), 0.0),
]
DEFAULT_GOAL_COLOR = en.ShapeColor.GREEN
DEFAULT_GOAL_XYHW = (-1.2, 1.16, 0.4, 2.4)
# Max possible L2 distance (arena diagonal 2*sqrt(2)).
D_MAX = 2.8284271247461903

class SweepToTopEnv(BaseEnv):
    """Sweep 3 debris entities to the goal zone at the top of the arena."""

    def __init__(
        self,
        use_state: bool = False,
        use_dense_reward: bool = False,
        use_color_reward: bool = False,
        rand_layout_full: bool = False,
        rand_shapes: bool = False,
        rand_colors: bool = False,
        index_seed_steps: int = 0,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            use_state: Whether to use states rather than pixels for the
                observation space.
            use_dense_reward: Whether to use a dense reward or a sparse one.
            rand_layout_full: Whether to randomize the poses of the debris.
            rand_shapes: Whether to randomize the shapes of the debris.
            rand_colors: Whether to randomize the colors of the debris and the
                goal zone.
        """
        super().__init__(**kwargs)

        self.use_state = True
        self.use_dense_reward = False
        # self.use_dense_reward = False
        self.use_color_reward = True
        self.rand_layout_full = rand_layout_full
        self.rand_shapes = rand_shapes
        self.rand_colors = rand_colors
        self.num_debris = 3
        self.stage_completed = [0] * self.num_debris
        self.starting_position = [0] * self.num_debris
        self.actual_goal_stage = 0 #0 is red, 1 is blue, 2 is yellow
        self.last_color_reward = 0
        self.index_seed_steps = index_seed_steps
        
        if self.use_state:
            # Redefine the observation space if we are using states as opposed
            # to pixels.
            # C is the number of states for the robot 
            c = 4 if self.action_dim == 2 else 5
            debris_features = 5
            base_dim = c + debris_features * self.num_debris + 2 * self.num_debris  # robot + (pos+color) + (dist to robot & dist to goal)
            
            goal_dim = self.num_debris + 1  # one-hot index of current goal block + 1 goal_finished

            low = np.array([-1.0] * base_dim + [0.0] * goal_dim, dtype=np.float32)
            high = np.array([+1.0] * base_dim + [1.0] * goal_dim, dtype=np.float32)

            self.observation_space = spaces.Box(low, high, dtype=np.float32)
            
    def get_finger_peak_distance(self) -> float:
        #Calculate the distance between the peaks of the two fingers.
        finger_peaks = []
        for finger_body, finger_side in zip(self._robot.finger_bodies, [-1, 1]):
            # Calculate the peak position of the finger in world coordinates.
            finger_angle = finger_body.angle
            finger_length = self._robot.finger_upper_length + self._robot.finger_lower_length
            peak_x = finger_body.position.x + finger_length * math.cos(finger_angle)
            peak_y = finger_body.position.y + finger_length * math.sin(finger_angle)
            finger_peaks.append((peak_x, peak_y))
        
        # Compute the Euclidean distance between the two peaks.
        peak1, peak2 = finger_peaks
        distance = math.sqrt((peak2[0] - peak1[0]) ** 2 + (peak2[1] - peak1[1]) ** 2)
        return distance

    def is_block_gripped(self, block_x, block_y) -> bool:
        #Check if the block is likely gripped by the robot fingers.
        lf_pos = np.array(self._robot.finger_bodies[0].position)
        rf_pos = np.array(self._robot.finger_bodies[1].position)
        pinch_center = (lf_pos + rf_pos) / 2.0
        grip_width = self.get_finger_peak_distance()

        block_pos = np.array([block_x, block_y])
        dist_to_center = np.linalg.norm(block_pos - pinch_center)

        # Conditions for a 'grip'
        is_near_center = dist_to_center < 0.4
        is_gripper_closed = grip_width < 0.28 
        return is_near_center and is_gripper_closed

    #ORIGINAL 
    def on_reset(self) -> None:
        print("Resetting SweepToTopEnv")
        robot_pos, robot_angle = DEFAULT_ROBOT_POSE
        robot = self._make_robot(robot_pos, robot_angle)

        goal_color = DEFAULT_GOAL_COLOR
        if self.rand_colors:
            goal_color = self.rng.choice(en.SHAPE_COLORS)
        sensor = en.GoalRegion(
            *DEFAULT_GOAL_XYHW,
            goal_color,
            dashed=False,
        )
        self.add_entities([sensor])
        self.__sensor_ref = sensor
        
              
        # Not randomized block positions.
        y_coords = [pose[0][1] for pose in DEFAULT_BLOCK_POSES]
        x_coords = [pose[0][0] for pose in DEFAULT_BLOCK_POSES]
        
        angles = [pose[1] for pose in DEFAULT_BLOCK_POSES]
       
        self.starting_position = y_coords
        
        debris_shapes = [DEFAULT_BLOCK_SHAPE] * self.num_debris
        colors_set = [en.ShapeColor.RED, en.ShapeColor.BLUE, en.ShapeColor.YELLOW]
        self.rng.shuffle(colors_set)
        debris_colors = colors_set[: self.num_debris]
       
        self.__debris_shapes = [
            self._make_shape(
                shape_type=shape,
                color_name=color,
                init_pos=(x, y),
                init_angle=angle,
            )
            for (x, y, angle, shape, color) in zip(
                x_coords,
                y_coords,
                angles,
                debris_shapes,
                debris_colors,
            )
        ]
        self.add_entities(self.__debris_shapes)

        # Add robot last for draw order reasons.
        self.add_entities([robot])

        # Block lookup index.
        self.__ent_index = en.EntityIndex(self.__debris_shapes)
        
        self.stage_completed = [0] * self.num_debris
        self.actual_goal_stage = 0
        self.last_color_reward = 0
        
    #SUBTASK
    # def on_reset(self) -> None:
    #     print("Resetting SweepToTopEnv with subtask logic")
    #     goal_color = DEFAULT_GOAL_COLOR
    #     if self.rand_colors:
    #         goal_color = self.rng.choice(en.SHAPE_COLORS)
    #     sensor = en.GoalRegion(
    #         *DEFAULT_GOAL_XYHW,
    #         goal_color,
    #         dashed=False,
    #     )
    #     self.add_entities([sensor])
    #     self.__sensor_ref = sensor
        
    #     y_coords = [pose[0][1] for pose in DEFAULT_BLOCK_POSES]
    #     x_coords = [pose[0][0] for pose in DEFAULT_BLOCK_POSES]
        
    #     angles = [pose[1] for pose in DEFAULT_BLOCK_POSES]
       
    #     self.starting_position = y_coords
        
    #     debris_shapes = [DEFAULT_BLOCK_SHAPE] * self.num_debris
    #     colors_set = [en.ShapeColor.RED, en.ShapeColor.BLUE, en.ShapeColor.YELLOW]
    #     self.rng.shuffle(colors_set)
    #     debris_colors = colors_set[: self.num_debris]
        
    #     first_subtask = True
    #     second_subtask = False
    #     third_subtask = False
        
    #     if self.index_seed_steps >= 30_000 and self.index_seed_steps < 530_000:
    #         print("Subtask 3 activated")
    #         first_subtask = False
    #         second_subtask = False
    #         third_subtask = True
    #     elif self.index_seed_steps >= 830_000 and self.index_seed_steps < 1_330_000:
    #         print("Subtask 2 activated")
    #         first_subtask = False
    #         second_subtask = True
    #         third_subtask = False
    #     elif self.index_seed_steps >= 1_630_000 and self.index_seed_steps < 2_130_000:
    #         print("Subtask 1 activated")
    #         first_subtask = True
    #         second_subtask = False
    #         third_subtask = False
    #     else:
    #         print("No subtask activated")
    #         first_subtask = False
    #         second_subtask = False
    #         third_subtask = False
                  
    #     if first_subtask:
    #         self.__debris_shapes = [
    #             self._make_shape(
    #                 shape_type=shape,
    #                 color_name=color,
    #                 init_pos=(x, 1.0) if color == en.ShapeColor.RED else (x, y),
    #                 init_angle=0.0 if color == en.ShapeColor.RED else angle,
    #             )
    #             for (x, y, angle, shape, color) in zip(
    #                 x_coords,
    #                 y_coords,
    #                 angles,
    #                 debris_shapes,
    #                 debris_colors,
    #             )
    #         ]
    #         for (x, color) in zip(x_coords, debris_colors):
    #             if color == en.ShapeColor.RED:
    #                 x_red = x
    #         robot_pos, robot_angle = (x_red, 0.55), 0.0
    #         robot = self._make_robot(robot_pos, robot_angle)
            
    #     elif second_subtask:
    #         self.__debris_shapes = [
    #             self._make_shape(
    #                 shape_type=shape,
    #                 color_name=color,
    #                 init_pos=(x, 1.0) if color == en.ShapeColor.RED or color == en.ShapeColor.BLUE else (x, y),
    #                 init_angle=0.0 if color == en.ShapeColor.RED or color == en.ShapeColor.BLUE else angle,
    #             )
    #             for (x, y, angle, shape, color) in zip(
    #                 x_coords,
    #                 y_coords,
    #                 angles,
    #                 debris_shapes,
    #                 debris_colors,
    #             )
    #         ]
    #         for (x, color) in zip(x_coords, debris_colors):
    #             if color == en.ShapeColor.BLUE:
    #                 x_blue = x
    #         robot_pos, robot_angle = (x_blue, 0.55), 0.0
    #         robot = self._make_robot(robot_pos, robot_angle)
            
    #     elif third_subtask:
    #         self.__debris_shapes = [
    #             self._make_shape(
    #                 shape_type=shape,
    #                 color_name=color,
    #                 init_pos=(x, 1.0) if color == en.ShapeColor.RED or color == en.ShapeColor.BLUE or color == en.ShapeColor.YELLOW else (x, y),
    #                 init_angle=0.0 if color == en.ShapeColor.RED or color == en.ShapeColor.BLUE or color == en.ShapeColor.YELLOW else angle,
    #             )
    #             for (x, y, angle, shape, color) in zip(
    #                 x_coords,
    #                 y_coords,
    #                 angles,
    #                 debris_shapes,
    #                 debris_colors,
    #             )
    #         ]
    #         for (x, color) in zip(x_coords, debris_colors):
    #             if color == en.ShapeColor.YELLOW:
    #                 x_yellow = x
    #         robot_pos, robot_angle = (x_yellow, 0.55), 0.0
    #         robot = self._make_robot(robot_pos, robot_angle)
            
    #     else:
    #         self.__debris_shapes = [
    #             self._make_shape(
    #                 shape_type=shape,
    #                 color_name=color,
    #                 init_pos=(x, y),
    #                 init_angle=angle
    #             )
    #             for (x, y, angle, shape, color) in zip(
    #                 x_coords,
    #                 y_coords,
    #                 angles,
    #                 debris_shapes,
    #                 debris_colors,
    #             )
    #         ]
    #         robot_pos, robot_angle = DEFAULT_ROBOT_POSE
    #         robot = self._make_robot(robot_pos, robot_angle)
            
    #     self.add_entities(self.__debris_shapes)

    #     # Add robot last for draw order reasons.
    #     self.add_entities([robot])

    #     # Block lookup index.
    #     self.__ent_index = en.EntityIndex(self.__debris_shapes)
        
    #     self.stage_completed = [False] * self.num_debris 
    #     self.actual_goal_stage = 0
    #     self.last_color_reward = 0
        
    #COMPLETELLY RANDOM FOR ROBOT AND BLOCKS
    # RANDOMIZED ROBOT AND BLOCKS WITH CONSTRAINTS
    # def on_reset(self) -> None:
    #     """
    #     Randomize robot and block positions at each reset.
    #     - No entity (robot or block) in the goal zone
    #     - All entities at least 0.4 units apart (including from the goal zone)
    #     - All entities at least 0.2 units from the arena walls
    #     - Arena bounds consistent with BaseEnv
    #     """
    #     # Use arena bounds from BaseEnv
    #     arena_l, arena_r, arena_b, arena_t = self.ARENA_BOUNDS_LRBT
    #     wall_buffer = 0.2
    #     arena_x_min, arena_x_max = arena_l + wall_buffer, arena_r - wall_buffer
    #     arena_y_min, arena_y_max = arena_b + wall_buffer, arena_t - wall_buffer

    #     def is_in_goal_zone(pos, margin=0.4):
    #         goal_x, goal_y, goal_h, goal_w = DEFAULT_GOAL_XYHW
    #         x, y = pos
    #         # Expand goal zone by margin in all directions
    #         in_x = (goal_x - margin) <= x <= (goal_x + goal_w + margin)
    #         in_y = (goal_y - goal_h - margin) <= y <= (goal_y + margin)
    #         return in_x and in_y

    #     def is_far_enough(pos, others, min_dist=0.4):
    #         for o in others:
    #             if np.linalg.norm(np.array(pos) - np.array(o)) < min_dist:
    #                 return False
    #         return True

    #     # Sample positions for robot and 3 blocks
    #     entities_pos = []
    #     max_tries = 1000
    #     for i in range(4):  # 1 robot + 3 blocks
    #         for _ in range(max_tries):
    #             x = self.rng.uniform(arena_x_min, arena_x_max)
    #             y = self.rng.uniform(arena_y_min, arena_y_max)
    #             pos = (x, y)
    #             if is_in_goal_zone(pos, margin=0.4):
    #                 continue
    #             if not is_far_enough(pos, entities_pos, min_dist=0.4):
    #                 continue
    #             entities_pos.append(pos)
    #             break
    #         else:
    #             raise RuntimeError("Failed to sample non-overlapping positions for robot and blocks.")

    #     robot_pos = entities_pos[0]
    #     robot_angle = self.rng.uniform(-np.pi, np.pi)
    #     robot = self._make_robot(robot_pos, robot_angle)

    #     goal_color = DEFAULT_GOAL_COLOR
    #     if self.rand_colors:
    #         goal_color = self.rng.choice(en.SHAPE_COLORS)
    #     sensor = en.GoalRegion(
    #         *DEFAULT_GOAL_XYHW,
    #         goal_color,
    #         dashed=False,
    #     )
    #     self.add_entities([sensor])
    #     self.__sensor_ref = sensor

    #     # Randomize block angles
    #     block_angles = [self.rng.uniform(-np.pi, np.pi) for _ in range(self.num_debris)]

    #     debris_shapes = [DEFAULT_BLOCK_SHAPE] * self.num_debris
    #     colors_set = [en.ShapeColor.RED, en.ShapeColor.BLUE, en.ShapeColor.YELLOW]
    #     self.rng.shuffle(colors_set)
    #     debris_colors = colors_set[: self.num_debris]

    #     # Assign block positions (entities_pos[1:4])
    #     self.starting_position = [pos[1] for pos in entities_pos[1:4]]
    #     self.__debris_shapes = [
    #         self._make_shape(
    #             shape_type=shape,
    #             color_name=color,
    #             init_pos=pos,
    #             init_angle=angle,
    #         )
    #         for (pos, angle, shape, color) in zip(
    #             entities_pos[1:4],
    #             block_angles,
    #             debris_shapes,
    #             debris_colors,
    #         )
    #     ]
    #     self.add_entities(self.__debris_shapes)

    #     # Add robot last for draw order reasons.
    #     self.add_entities([robot])

    #     # Block lookup index.
    #     self.__ent_index = en.EntityIndex(self.__debris_shapes)

    #     self.stage_completed = [False] * self.num_debris
    #     self.actual_goal_stage = 0
    #     self.last_color_reward = 0
    
    # RANDOMIZE ONLY ROBOT POSITION, BLOCKS FIXED, WITH CONSTRAINTS
    # def on_reset(self) -> None:
    #     """
    #     Randomize only the robot position at each reset.
    #     - Robot must not be in the goal zone
    #     - Robot must be at least 0.4 units from all blocks and the goal zone
    #     - Robot must be at least 0.2 units from the arena walls
    #     - Blocks are placed at DEFAULT_BLOCK_POSES
    #     """
    #     arena_l, arena_r, arena_b, arena_t = self.ARENA_BOUNDS_LRBT
    #     wall_buffer = 0.2
    #     arena_x_min, arena_x_max = arena_l + wall_buffer, arena_r - wall_buffer
    #     arena_y_min, arena_y_max = arena_b + wall_buffer, arena_t - wall_buffer

    #     def is_in_goal_zone(pos, margin=0.4):
    #         goal_x, goal_y, goal_h, goal_w = DEFAULT_GOAL_XYHW
    #         x, y = pos
    #         in_x = (goal_x - margin) <= x <= (goal_x + goal_w + margin)
    #         in_y = (goal_y - goal_h - margin) <= y <= (goal_y + margin)
    #         return in_x and in_y

    #     # Get fixed block positions
    #     block_positions = [pose[0] for pose in DEFAULT_BLOCK_POSES]
    #     block_angles = [pose[1] for pose in DEFAULT_BLOCK_POSES]
    #     debris_shapes = [DEFAULT_BLOCK_SHAPE] * self.num_debris
    #     colors_set = [en.ShapeColor.RED, en.ShapeColor.BLUE, en.ShapeColor.YELLOW]
    #     self.rng.shuffle(colors_set)
    #     debris_colors = colors_set[: self.num_debris]

    #     # Sample robot position
    #     max_tries = 1000
    #     for _ in range(max_tries):
    #         x = self.rng.uniform(arena_x_min, arena_x_max)
    #         y = self.rng.uniform(arena_y_min, arena_y_max)
    #         pos = (x, y)
    #         if is_in_goal_zone(pos, margin=0.4):
    #             continue
    #         # Check distance to all blocks
    #         if any(np.linalg.norm(np.array(pos) - np.array(block_pos)) < 0.4 for block_pos in block_positions):
    #             continue
    #         robot_pos = pos
    #         break
    #     else:
    #         raise RuntimeError("Failed to sample valid robot position.")

    #     robot_angle = self.rng.uniform(-np.pi, np.pi)
    #     robot = self._make_robot(robot_pos, robot_angle)

    #     goal_color = DEFAULT_GOAL_COLOR
    #     if self.rand_colors:
    #         goal_color = self.rng.choice(en.SHAPE_COLORS)
    #     sensor = en.GoalRegion(
    #         *DEFAULT_GOAL_XYHW,
    #         goal_color,
    #         dashed=False,
    #     )
    #     self.add_entities([sensor])
    #     self.__sensor_ref = sensor

    #     self.starting_position = [y for (x, y) in block_positions]
    #     self.__debris_shapes = [
    #         self._make_shape(
    #             shape_type=shape,
    #             color_name=color,
    #             init_pos=pos,
    #             init_angle=angle,
    #         )
    #         for (pos, angle, shape, color) in zip(
    #             block_positions,
    #             block_angles,
    #             debris_shapes,
    #             debris_colors,
    #         )
    #     ]
    #     self.add_entities(self.__debris_shapes)

    #     # Add robot last for draw order reasons.
    #     self.add_entities([robot])

    #     # Block lookup index.
    #     self.__ent_index = en.EntityIndex(self.__debris_shapes)

    #     self.stage_completed = [False] * self.num_debris
    #     self.actual_goal_stage = 0
    #     self.last_color_reward = 0
    
    
    ### CURRICULUM
    # def on_reset(self) -> None:
    #     # print("ON RESET EXECUTED")
    #     subtask_len = 500_000
    #     min_subtask_1_index = 1_030_000
    #     min_subtask_2_index = 530_000
    #     min_subtask_3_index = 30_000
        
    #     arena_l, arena_r, arena_b, arena_t = self.ARENA_BOUNDS_LRBT
    #     wall_buffer = 0.2
    #     arena_x_min, arena_x_max = arena_l + wall_buffer, arena_r - wall_buffer
    #     arena_y_min, arena_y_max = arena_b + wall_buffer, arena_t - wall_buffer

    #     goal_color = DEFAULT_GOAL_COLOR
    #     if self.rand_colors:
    #         goal_color = self.rng.choice(en.SHAPE_COLORS)
    #     sensor = en.GoalRegion(
    #         *DEFAULT_GOAL_XYHW,
    #         goal_color,
    #         dashed=False,
    #     )
    #     self.add_entities([sensor])
    #     self.__sensor_ref = sensor

    #     y_coords = [pose[0][1] for pose in DEFAULT_BLOCK_POSES]
    #     x_coords = [pose[0][0] for pose in DEFAULT_BLOCK_POSES]
    #     angles = [pose[1] for pose in DEFAULT_BLOCK_POSES]
    #     self.starting_position = y_coords

    #     debris_shapes = [DEFAULT_BLOCK_SHAPE] * self.num_debris
    #     colors_set = [en.ShapeColor.RED, en.ShapeColor.BLUE, en.ShapeColor.YELLOW]
    #     self.rng.shuffle(colors_set)
    #     debris_colors = colors_set[: self.num_debris]

    #     # Determine subtask
    #     first_subtask = False
    #     second_subtask = False
    #     third_subtask = False
    #     if self.index_seed_steps >= min_subtask_1_index and self.index_seed_steps < min_subtask_1_index + subtask_len:
    #         print("SUBTASK 1 CHOOSEN")
    #         first_subtask = True
    #         second_subtask = False
    #         third_subtask = False
    #     elif self.index_seed_steps >= min_subtask_2_index and self.index_seed_steps < min_subtask_2_index + subtask_len:
    #         print("SUBTASK 2 CHOOSEN")
    #         second_subtask = True
    #         first_subtask = False
    #         third_subtask = False
    #     elif self.index_seed_steps >= min_subtask_3_index and self.index_seed_steps < min_subtask_3_index + subtask_len:
    #         print("SUBTASK 3 CHOOSEN")
    #         third_subtask = True
    #         first_subtask = False
    #         second_subtask = False

    #     def is_in_goal_zone(pos, margin=0.4):
    #         goal_x, goal_y, goal_h, goal_w = DEFAULT_GOAL_XYHW
    #         x, y = pos
    #         in_x = (goal_x - margin) <= x <= (goal_x + goal_w + margin)
    #         in_y = (goal_y - goal_h - margin) <= y <= (goal_y + margin)
    #         return in_x and in_y

    #     # Place blocks as in your subtask logic
    #     if first_subtask:
    #         if self.index_seed_steps <= min_subtask_1_index + subtask_len/4:
    #             init_red_block_y = 0.7
    #         elif self.index_seed_steps <= min_subtask_1_index + subtask_len/2:
    #             init_red_block_y = 0.5
    #         elif self.index_seed_steps <= min_subtask_1_index + 3*subtask_len/4:
    #             init_red_block_y = 0.3
    #         else:
    #             init_red_block_y = 0.0
                
    #         self.__debris_shapes = [
    #             self._make_shape(
    #                 shape_type=shape,
    #                 color_name=color,
    #                 init_pos=(x, init_red_block_y) if color == en.ShapeColor.RED else (x, y),
    #                 init_angle=0.0 if color == en.ShapeColor.RED else angle,
    #             )
    #             for (x, y, angle, shape, color) in zip(
    #                 x_coords, y_coords, angles, debris_shapes, debris_colors
    #             )
    #         ]
    #         # Find red block position
    #         for block in self.__debris_shapes:
    #             if block.color_name == en.ShapeColor.RED:
    #                 block_pos = np.array(block.init_pos)
    #                 break
    #     elif second_subtask:
    #         if self.index_seed_steps <= min_subtask_2_index + subtask_len/4:
    #             init_blue_block_y = 0.7
    #         elif self.index_seed_steps <= min_subtask_2_index + subtask_len/2:
    #             init_blue_block_y = 0.5
    #         elif self.index_seed_steps <= min_subtask_2_index + 3*subtask_len/4:
    #             init_blue_block_y = 0.3
    #         else:
    #             init_blue_block_y = 0.0
                
    #         self.__debris_shapes = [
    #             self._make_shape(
    #                 shape_type=shape,
    #                 color_name=color,
    #                 init_pos= ((x, 1.0) if color == en.ShapeColor.RED else (x,init_blue_block_y) if color == en.ShapeColor.BLUE else (x, y)),
    #                 init_angle=0.0 if color in [en.ShapeColor.RED, en.ShapeColor.BLUE] else angle,
    #             )
    #             for (x, y, angle, shape, color) in zip(
    #                 x_coords, y_coords, angles, debris_shapes, debris_colors
    #             )
    #         ]
    #         for block in self.__debris_shapes:
    #             if block.color_name == en.ShapeColor.BLUE:
    #                 block_pos = np.array(block.init_pos)
    #                 break
    #     elif third_subtask:
    #         if self.index_seed_steps <= min_subtask_3_index + subtask_len/4:
    #             init_yellow_block_y = 0.7
    #         elif self.index_seed_steps <= min_subtask_3_index + subtask_len/2:
    #             init_yellow_block_y = 0.5
    #         elif self.index_seed_steps <= min_subtask_3_index + 3*subtask_len/4:
    #             init_yellow_block_y = 0.3
    #         else:
    #             init_yellow_block_y = 0.0
    #         self.__debris_shapes = [
    #             self._make_shape(
    #                 shape_type=shape,
    #                 color_name=color,
    #                 init_pos= ((x, 1.0) if color in [en.ShapeColor.RED, en.ShapeColor.BLUE] else (x,init_yellow_block_y) if color == en.ShapeColor.YELLOW else (x, y)),
    #                 init_angle=0.0 if color in [en.ShapeColor.RED, en.ShapeColor.BLUE, en.ShapeColor.YELLOW] else angle,
    #             )
    #             for (x, y, angle, shape, color) in zip(
    #                 x_coords, y_coords, angles, debris_shapes, debris_colors
    #             )
    #         ]
    #         for block in self.__debris_shapes:
    #             if block.color_name == en.ShapeColor.YELLOW:
    #                 block_pos = np.array(block.init_pos)
    #                 break
    #     else:
    #         self.__debris_shapes = [
    #             self._make_shape(
    #                 shape_type=shape,
    #                 color_name=color,
    #                 init_pos=(x, y),
    #                 init_angle=angle
    #             )
    #             for (x, y, angle, shape, color) in zip(
    #                 x_coords, y_coords, angles, debris_shapes, debris_colors
    #             )
    #         ]
    #         block_pos = None  # No subtask, randomize robot as usual

    #     # ...existing code...

    #     # Randomize robot position near the block (if subtask), else default
    #     max_tries = 1000
    #     robot_pos = None
    #     robot_angle = 0.0

    #     if block_pos is not None:
    #         # Calculate proportional distance based on block's y position
    #         # Higher y position = closer to goal = robot should be closer to block
    #         block_y = block_pos[1]
    #         min_distance, max_distance =  0.35, 1.0
            
    #         for _ in range(max_tries):
    #             angle = random.uniform( np.pi, 2 * np.pi)  
    #             dist = random.uniform(min_distance, max_distance)
    #             offset = np.array([np.cos(angle), np.sin(angle)]) * dist
    #             robot_pos_candidate = block_pos + offset
                
    #             # print(f"Trying robot position: {robot_pos_candidate}, angle: {angle}, block position: {block_pos}, min_distance: {min_distance}, max_distance: {max_distance}")
                
    #             # Check if position is valid
    #             if not is_in_goal_zone(robot_pos_candidate):
    #                 if (arena_x_min <= robot_pos_candidate[0] <= arena_x_max and
    #                     arena_y_min <= robot_pos_candidate[1] <= arena_y_max):
                        
    #                     # Check distance to ALL blocks (not just the target)
    #                     too_close_to_blocks = False
    #                     for block in self.__debris_shapes:
    #                         block_xy = np.array(block.init_pos)
    #                         if np.linalg.norm(np.array(robot_pos_candidate) - block_xy) < 0.2:
    #                             too_close_to_blocks = True
    #                             break
                        
    #                     if not too_close_to_blocks:
    #                         robot_pos = tuple(robot_pos_candidate)
    #                         robot_angle = math.atan2(block_pos[1] - robot_pos[1], block_pos[0] - robot_pos[0]) - np.pi / 2
    #                         robot_angle = (robot_angle + 2 * np.pi) % (2 * np.pi)

    #                         break    
    #     # robot_angle = 0.0

    #     # Ensure we have a valid robot position
    #     if robot_pos is None:
    #         robot_pos = DEFAULT_ROBOT_POSE[0]

    #     robot = self._make_robot(robot_pos, robot_angle)
        
    #     self.add_entities(self.__debris_shapes)
    #     self.add_entities([robot])
    #     self.__ent_index = en.EntityIndex(self.__debris_shapes)
    #     self.stage_completed = [False] * self.num_debris
    #     self.actual_goal_stage = 0
    #     self.last_color_reward = 0
            
        

    def get_state(self) -> np.ndarray:
        robot_pos = self._robot.body.position
        robot_angle_cos = np.cos(self._robot.body.angle)
        robot_angle_sin = np.sin(self._robot.body.angle)
        goal_y = 1
        target_pos = []
        robot_target_dist = []
        target_goal_dist = []
        for target_shape in self.__debris_shapes:
            tpos = target_shape.shape_body.position
            color = {
            en.ShapeColor.RED:    [1.0, 0.0, 0.0],
            en.ShapeColor.BLUE:   [0.0, 1.0, 0.0],
            en.ShapeColor.YELLOW: [0.0, 0.0, 1.0],
            }[target_shape.color_name]
            target_pos.extend([tpos[0], tpos[1], *color])
            robot_target_dist.append(np.linalg.norm(robot_pos - tpos) / D_MAX)
            gpos = (tpos[0], goal_y)
            target_goal_dist.append(np.linalg.norm(tpos - gpos) / D_MAX)
        state = [
            *tuple(robot_pos),  # 2
            *target_pos,  # 2t
            robot_angle_cos,  # 1
            robot_angle_sin,  # 1
            *robot_target_dist,  # t
            *target_goal_dist,  # t
        ]  # total = 4 + 4t
        if self.action_dim == 3:
            state.append(self._robot.finger_width)
        state = np.array(state, dtype=np.float32)
        goal_one_hot = np.zeros(self.num_debris+1, dtype=np.float32)
        goal_one_hot[self.actual_goal_stage] = 1.0
        return np.concatenate([state, goal_one_hot], axis=0)

    def score_on_end_of_traj(self) -> float:
        """
        Score based on the correct sequential placement of blocks in the goal zone:
        - 1 point: if only the red block is in the goal zone
        - 2 points: if red and blue blocks are in the goal zone (in that order)
        - 3 points: if all three blocks are in the goal zone (in the correct order)
        - 0 points: any other configuration
        """
        # Get all entities completely contained in the goal zone
        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            contained=True, ent_index=self.__ent_index
        )
        
        # If no entities in goal zone, return 0
        if len(overlap_ents) == 0:
            return 0.0
        score = 0
        red_block, blue_block, yellow_block = self.stage_completed
        if red_block != 0 and blue_block == 0 and yellow_block == 0:
            score = 1.0
        elif red_block != 0 and blue_block != 0 and yellow_block == 0 and blue_block > red_block:
            score = 2.0
        elif red_block != 0 and blue_block != 0 and yellow_block != 0 and blue_block > red_block and yellow_block > blue_block:
            score = 3.0
        else:
            score = 0.0
            
        return score / 3.0
 
        

    # def _dense_reward(self) -> float:
    #     """Mean distance of all debris entitity positions to goal zone."""
    #     y = 1
    #     target_goal_dists = []
    #     for target_shape in self.__debris_shapes:
    #         target_pos = target_shape.shape_body.position
    #         goal_pos = (target_pos[0], y)  # Top of screen.
    #         dist = np.linalg.norm(target_pos - goal_pos)
    #         if target_pos[1] > 0.88:
    #             dist = 0
    #         target_goal_dists.append(dist)
    #     target_goal_dists = np.mean(target_goal_dists)
    #     return -1.0 * target_goal_dists

    # def _sparse_reward(self) -> float:
    #     """Fraction of debris entities inside goal zone."""
    #     # `score_on_end_of_traj` is supposed to be called at the end of a
    #     # trajectory but we use it here since it gives us exactly the reward
    #     # we're looking for.
    #     return self.score_on_end_of_traj()

    def _color_reward(self) -> float:
        """
        Reward function where the robot should move the red block to the goal area first,
        followed by the blue block, and finally the yellow block.
        """
        # Robot position
        robot_pos = np.array([self._robot.body.position.x, self._robot.body.position.y])

        # Goal position
        goal_x, goal_y, goal_h, goal_w = DEFAULT_GOAL_XYHW
        goal_y_min = goal_y - goal_h
        goal_y_max = goal_y
        goal_center_y = (goal_y_min + goal_y_max) / 2
        goal_lower_center = (goal_center_y + goal_y_min) / 2
        
        # Pinch center (robot's gripper center)
        left_finger_body = self._robot.finger_bodies[0]
        right_finger_body = self._robot.finger_bodies[1]
        lf_pos = np.array(left_finger_body.position)
        rf_pos = np.array(right_finger_body.position)
        pinch_center = (lf_pos + rf_pos) / 2.0
        
        # Helper function to check if a block is in the goal area
        def in_goal(pos_y: float) -> bool:
            return goal_lower_center <= pos_y <= goal_y_max

        # Helper function to calculate distances
        def calculate_distances(block, block_starting_y):
            block_x, block_y = block.shape_body.position
            block_pos = np.array([block_x, block_y])
            block_dist_to_goal = abs(block_y - goal_y_min) if (block_y - goal_y_min) <= 0 else 0
            block_dist_to_robot = np.linalg.norm(block_pos - pinch_center)
            block_dist_init = abs(block_starting_y - goal_y_min)
            return block_x, block_y, block_dist_to_goal, block_dist_to_robot, block_dist_init

        # Get blocks and their starting positions
        blocks = {
            "red": next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.RED),
            "blue": next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.BLUE),
            "yellow": next(block for block in self.__debris_shapes if block.color_name == en.ShapeColor.YELLOW),
        }
        starting_positions = {
            "red": self.starting_position[next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.RED)],
            "blue": self.starting_position[next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.BLUE)],
            "yellow": self.starting_position[next(i for i, block in enumerate(self.__debris_shapes) if block.color_name == en.ShapeColor.YELLOW)],
        }

        # Calculate distances for each block
        distances = {color: calculate_distances(blocks[color], starting_positions[color]) for color in blocks}
    
        # Reward calculation
        moving_to_block_reward = 0
        push_reward = 0
        grip_reward = 0
        
        # print(f"State: {self.stage_completed}, Actual goal stage: {self.actual_goal_stage}, Index: {self.index_seed_steps}, In Goal Red: {in_goal(distances['red'][1])}, In Goal Blue: {in_goal(distances['blue'][1])}, In Goal Yellow: {in_goal(distances['yellow'][1])}")
        if self.stage_completed[0] == 0:
            # print("Stage 0: Moving to red block")
            # Reward for moving the robot near the red block
            moving_to_block_reward += (1.0 / (1.0 + distances["red"][3]))
            push_reward += (distances["red"][4] - distances["red"][2]) / distances["red"][4]
            red_block_x, red_block_y = blocks["red"].shape_body.position
            grip_reward = 1.0 if self.is_block_gripped(red_block_x, red_block_y) else 0.0
            if in_goal(distances["red"][1]) and not (in_goal(distances["blue"][1]) or in_goal(distances["yellow"][1])):
                self.stage_completed[0] = self.index_seed_steps
                self.actual_goal_stage = 1
        elif self.stage_completed[1] == 0 and in_goal(distances["red"][1]):
            # print("Stage 1: Moving to blue block")
            # Reward for moving the robot near the blue block
            moving_to_block_reward += 1.2 + 1.0 / (1.0 + distances["blue"][3])
            push_reward += 1.2 + (distances["blue"][4] - distances["blue"][2]) / distances["blue"][4]
            blue_block_x, blue_block_y = blocks["blue"].shape_body.position
            grip_reward = 1.5 if self.is_block_gripped(blue_block_x, blue_block_y) else 0.0
            if in_goal(distances["blue"][1]) and in_goal(distances["red"][1]) and not in_goal(distances["yellow"][1]):
                self.stage_completed[1] = self.index_seed_steps
                self.actual_goal_stage = 2
        elif self.stage_completed[2] == 0 and in_goal(distances["blue"][1]) and in_goal(distances["red"][1]):
            # print("Stage 2: Moving to yellow block")
            # Reward for moving the robot near the yellow block
            moving_to_block_reward += 2.4 + 1.0 / (1.0 + distances["yellow"][3])
            push_reward += 2.4 + (distances["yellow"][4] - distances["yellow"][2]) / distances["yellow"][4]
            yellow_block_x, yellow_block_y = blocks["yellow"].shape_body.position
            grip_reward = 2 if self.is_block_gripped(yellow_block_x, yellow_block_y) else 0.0
            if in_goal(distances["yellow"][1]) and in_goal(distances["blue"][1]) and in_goal(distances["red"][1]):
                self.stage_completed[2] = self.index_seed_steps
                self.actual_goal_stage = 3
            else:
                self.last_color_reward = 0.3 * moving_to_block_reward + 0.7 * push_reward
         
        if self.stage_completed[0]!= 0 and self.stage_completed[1] != 0 and self.stage_completed[2]!=0:
            # print("All blocks in goal area, finalizing reward")
            # All blocks are in the goal area
            # To keep the final reward at the higher value possible without falling to zero
            reward = 4.0 + self.last_color_reward 
        else:
            reward = 0.3 * moving_to_block_reward + 0.7 * push_reward
 
        return reward
            
        
    def get_reward(self) -> float:
        if self.use_dense_reward:
            return self._dense_reward()
        if self.use_color_reward:
            return self._color_reward()
        return self._sparse_reward()

    def reset(self) -> np.ndarray:
        obs = super().reset()
        self.stage_completed = [0] * self.num_debris
        if self.use_state:
            return self.get_state()
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, rew, done, info = super().step(action)
        if self.use_state:
            obs = self.get_state()
        return obs, rew, done, info

    


