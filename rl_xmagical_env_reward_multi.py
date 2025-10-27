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
import os
import sys
from absl import app
from absl import flags
from absl import logging
from configs.constants import EMBODIMENTS
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id

import train_policy_multi as train_policy_multi
from absl import app as absl_app
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
CONFIG_PATH = "configs/xmagical/rl/env_reward.py"

flags.DEFINE_enum("embodiment", None, EMBODIMENTS, "Which embodiment to train.")
flags.DEFINE_list("seeds", [0, 5], "List specifying the range of seeds to run.")
flags.DEFINE_string("name_test", "xmagical", "Name of the test to run.")



def main(_):
  # Map the embodiment to the x-MAGICAL env name.
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[FLAGS.embodiment]

  import hashlib
  # Create a deterministic hash based on pretrained_path and current parameters
  hash_input = f"{env_name}_{FLAGS.embodiment}_{FLAGS.name_test}"
  deterministic_uid = hashlib.md5(hash_input.encode()).hexdigest()[:8]
  
  experiment_name = string_from_kwargs(
        env_name=env_name,
        reward="environmental",
        embodiment=FLAGS.embodiment,
        uid=deterministic_uid,  # Use deterministic UID instead of random
    )
  logging.info("Experiment name: %s", experiment_name)

  # Execute each seed in parallel.
  seed = int(FLAGS.seeds[0])
  
  sys.argv = [
        "train_policy_multi.py",
        "--experiment_name",
        experiment_name,
        "--env_name",
        f"{env_name}",
        "--config",
        f"{CONFIG_PATH}:{FLAGS.embodiment}",
        "--seed",
        f"{seed}",
        "--wandb",
        f"{True}",
  ]
  absl_app.run(train_policy_multi.main)
  



if __name__ == "__main__":
  flags.mark_flag_as_required("embodiment")
  app.run(main)
