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

"""X-MAGICAL: Train a policy with a learned reward."""

import os
import subprocess

from absl import app
from absl import flags
from absl import logging
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
import yaml

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_path", None, "Path to pretraining experiment.")
flags.DEFINE_integer("seed", 12, "Seed to run.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_boolean("wandb", False, "Log on W&B.")
flags.DEFINE_string("algo", "inest", "The RL algorithm to use.")


def main(_):
  
  with open(os.path.join(FLAGS.pretrained_path, "metadata.yaml"), "r") as fp:
    kwargs = yaml.load(fp, Loader=yaml.FullLoader)
 
    
  if FLAGS.algo == "inest":
    reward_type = "inest"
  elif FLAGS.algo == "inest_knn":
    reward_type = "inest_knn"
  elif FLAGS.algo == "state_intrinsic":
    reward_type = "state_intrinsic"
  elif FLAGS.algo == "reds":
    reward_type = "reds"
  elif FLAGS.algo == "goal_classifier":
    reward_type = "goal_classifier"
  elif FLAGS.algo == "distance_to_goal":
    reward_type = "distance_to_goal"
  else:
    reward_type = "inest"
 
  # Map the embodiment to the x-MAGICAL env name.
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[kwargs["embodiment"]]

  # Generate a unique experiment name.
  # experiment_name = string_from_kwargs(
  #     env_name=env_name,
  #     reward="learned",
  #     reward_type=reward_type,
  #     mode=kwargs["mode"],
  #     algo=kwargs["algo"],
  #     uid=unique_id(),
  # )
  
  # To use a existing experiment name, uncomment below and comment above.
  # It must be followed by the RESUME entry in the subprocess call.
  experiment_name = "env_name=SweepToTop-Gripper-State-Allo-TestLayout-v0_reward=learned_reward_type=holdr_mode=same_algo=xirl_uid=ef3eede2-80d6-465a-bda9-1c4bcbc01209"
  logging.info("Experiment name: %s", experiment_name)

  # Execute each seed in parallel.
  print("Starting train_policy.py subprocess...")
  process = subprocess.Popen([  # pylint: disable=consider-using-with
    "python",
    "train_policy.py",
    "--experiment_name",
    experiment_name,
    "--env_name",
    f"{env_name}",
    "--config",
    f"configs/xmagical/rl/env_reward.py:{kwargs['embodiment']}",
    "--config.reward_wrapper.pretrained_path",
    f"{FLAGS.pretrained_path}",
    "--config.reward_wrapper.type",
    f"{reward_type}",
    "--seed",
    f"{FLAGS.seed}",
    "--device",
    f"{FLAGS.device}",
    "--wandb",
    f"{FLAGS.wandb}",
    "--resume",
    f"{True}"
  ])
  
  # Wait for the process to complete and get the return code
  return_code = process.wait()
  print(f"train_policy.py finished with return code: {return_code}")
  
  if return_code != 0:
    print(f"ERROR: train_policy.py failed with return code {return_code}")
    exit(return_code)
  else:
    print("train_policy.py completed successfully!")




if __name__ == "__main__":
  flags.mark_flag_as_required("pretrained_path")
  app.run(main)
