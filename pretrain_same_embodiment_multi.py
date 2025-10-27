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

"""X-MAGICAL same-embodiment pretraining script."""

import os
import os.path as osp
import subprocess

from absl import app
from absl import flags
from absl import logging
from configs.constants import ALGORITHMS
from configs.constants import EMBODIMENTS
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
import yaml
import torch

import sys
import pretrain_multi as pretrain_multi
from absl import app as absl_app


# pylint: disable=logging-fstring-interpolation

# Mapping from pretraining algorithm to config file.
ALGO_TO_CONFIG = {
    "xirl": "configs/xmagical/pretraining/tcc.py",
    "lifs": "configs/xmagical/pretraining/lifs.py",
    "tcn": "configs/xmagical/pretraining/tcn.py",
    "goal_classifier": "configs/xmagical/pretraining/classifier.py",
    "raw_imagenet": "configs/xmagical/pretraining/imagenet.py",
    "holdr": "configs/xmagical/pretraining/holdr.py",
    "reds": "configs/xmagical/pretraining/reds.py",
    "dinov2": "configs/xmagical/pretraining/dinov2.py",
}
# We want to pretrain on the entire demonstrations.
MAX_DEMONSTRATIONS = -1
FLAGS = flags.FLAGS

flags.DEFINE_enum("algo", None, ALGORITHMS, "The pretraining algorithm to use.")
flags.DEFINE_enum(
    "embodiment", None, EMBODIMENTS,
    "Which embodiment to train. Will train all sequentially if not specified.")
flags.DEFINE_bool("unique_name", False,
                  "Whether to append a unique ID to the experiment name.")
flags.DEFINE_boolean("wandb", False, "Whether to log on W&B.")


def get_rank():
    """Get the current process rank for DDP."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def main(_):
    rank = get_rank()
    
    embodiments = EMBODIMENTS if FLAGS.embodiment is None else [FLAGS.embodiment]

    for embodiment in embodiments:
        # Generate experiment name deterministically based on input parameters
        # This ensures all DDP processes use the same experiment name
        import hashlib
        
        kwargs = {
            "dataset": "xmagical",
            "mode": "same",
            "algo": FLAGS.algo,
            "embodiment": embodiment,
        }
        
        if FLAGS.unique_name:
            # Create a deterministic hash based on current parameters
            hash_input = f"{FLAGS.algo}_{embodiment}_xmagical_same"
            deterministic_uid = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            kwargs["uid"] = deterministic_uid + "invrobo"
        
        experiment_name = string_from_kwargs(**kwargs)
        logging.info("Experiment name: %s", experiment_name)

        # Execute the pretraining with DDP using direct function call
        sys.argv = [
            "pretrain_multi.py",
            "--experiment_name", experiment_name,
            "--config", f"{ALGO_TO_CONFIG[FLAGS.algo]}",
            "--config.data.pretrain_action_class", f"({repr(embodiment)},)",
            "--config.data.downstream_action_class", f"({repr(embodiment)},)",
            "--config.data.max_vids_per_class", f"{MAX_DEMONSTRATIONS}",
            "--wandb", f"{FLAGS.wandb}",
        ]
        
        # Add raw_imagenet flag if needed
        if FLAGS.algo == "raw_imagenet":
            sys.argv.append("--raw_imagenet")
        
        absl_app.run(pretrain_multi.main)

        # Only rank 0 should handle post-processing
        if rank == 0:
            # Note: This assumes that the config.root_dir value has not been
            # changed to its default value of 'tmp/xirl/pretrain_runs/'.
            exp_path = osp.join("/home/liannello/xirl/experiment_results/Allocentric/pretraining", experiment_name)

            # Dump experiment metadata as yaml file first
            logging.info("Writing metadata.yaml to %s", exp_path)
            try:
                with open(osp.join(exp_path, "metadata.yaml"), "w") as fp:
                    yaml.dump(kwargs, fp)
                logging.info("Successfully wrote metadata.yaml")
            except Exception as e:
                logging.error("Failed to write metadata.yaml: %s", e)

            # The 'goal_classifier' baseline does not need to compute a goal embedding.
            if FLAGS.algo != "goal_classifier":
                logging.info("Computing goal embedding for experiment: %s", exp_path)
                try:
                    result = subprocess.run(
                        [
                            "python",
                            "compute_goal_embedding.py",
                            "--experiment_path",
                            exp_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logging.info("Successfully computed goal embedding")
                    logging.info("compute_goal_embedding.py output: %s", result.stdout)
                except subprocess.CalledProcessError as e:
                    logging.error("Failed to compute goal embedding: %s", e)
                    logging.error("Error output: %s", e.stderr)
                    logging.error("Return code: %s", e.returncode)
                except Exception as e:
                    logging.error("Unexpected error in compute_goal_embedding.py: %s", e)
            else:
                logging.info("Skipping goal embedding computation for goal_classifier")
        else:
            logging.info("Rank %d: Skipping post-processing (only rank 0 handles this)", rank)


if __name__ == "__main__":
    flags.mark_flag_as_required("algo")
    app.run(main)