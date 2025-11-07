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

"""X-MAGICAL: Train a policy with the sparse environment reward."""

import subprocess

from absl import app
from absl import flags
from absl import logging
from configs.constants import EMBODIMENTS
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
CONFIG_PATH = "configs/xmagical/rl/env_reward.py"

flags.DEFINE_enum("embodiment", None, EMBODIMENTS, "Which embodiment to train.")
flags.DEFINE_integer("seed", 12 , "Seeds to run.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_boolean("wandb", False, "Log on W&B.")


def main(_):
  # Map the embodiment to the x-MAGICAL env name.
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[FLAGS.embodiment]

  # Generate a unique experiment name.
  # experiment_name = string_from_kwargs(
  #     env_name=env_name,
  #     reward="sparse_env",
  #     uid=unique_id(),
  # )
  
  experiment_name = "env_name=SweepToTop-Gripper-State-Allo-TestLayout-v0_reward=sparse_env_uid=dfce6666-ac5a-440c-8011-922d0c0a53f4"
  logging.info("Experiment name: %s", experiment_name)
  
  

  # Execute each seed in parallel.
  
  process = subprocess.Popen([  # pylint: disable=consider-using-with
            "python",
            "train_policy.py",
            "--experiment_name",
            experiment_name,
            "--env_name",
            f"{env_name}",
            "--config",
            f"{CONFIG_PATH}:{FLAGS.embodiment}",
            "--seed",
            f"{FLAGS.seed}",
            "--device",
            f"{FLAGS.device}",
             "--wandb",
            f"{FLAGS.wandb}",
            "--resume",
            f"{True}"
        ])
  
  return_code = process.wait()
  print(f"train_policy.py finished with return code: {return_code}")

  if return_code != 0:
    print(f"ERROR: train_policy.py failed with return code {return_code}")
    exit(return_code)
  else:
    print("train_policy.py completed successfully!")



if __name__ == "__main__":
  flags.mark_flag_as_required("embodiment")
  app.run(main)
