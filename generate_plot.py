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

"""Example script for generating plots like Figures 3 and 4 of the paper."""

import dataclasses
import os
import pathlib
from typing import List

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

_PLOT_DIR = flags.DEFINE_string(
    "plot_dir",
    "/home/liannello/xirl/experiment_results/Egocentric/Results",
    "Directory wherein to store plots.")
_MATPLOTLIB_SIZE = flags.DEFINE_integer(
    "size",
    15,
    "Size multiplier for matplotlib figure.")
_STD_DEV_FRAC = flags.DEFINE_float(
    "std_dev_frac",
    0.5,
    "+/- standard deviation fraction.")
_MATPLOTLIB_DPI = flags.DEFINE_integer(
    "dpi",
    100,
    "Dots per inch for PNG plots.")


def update_plotting_params(size):
  plt.rcParams.update({
      "legend.fontsize": "large",
      "axes.titlesize": size,
      "axes.labelsize": size,
      "xtick.labelsize": size,
      "ytick.labelsize": size,
  })


def minimum_truncate_array_list(arrs):
  min_len = arrs[0].shape[0]
  for arr in arrs[1:]:
    if arr.shape[0] < min_len:
      min_len = arr.shape[0]
  return [arr[:min_len] for arr in arrs]


@dataclasses.dataclass
class Experiment:
  """Parses experiment data and computes statistics across seeds."""
  path: str
  name: str
  color: str
  linestyle: str

  def __post_init__(self):
    self.path = pathlib.Path(self.path)
    if not self.path.exists():
      raise ValueError(f"{self.path} does not exist.")
     # Directly check if "tb" exists inside the main directory
    tb_dir = self.path / "tb"
    if not tb_dir.exists():
        raise ValueError(f"{tb_dir} does not exist.")

    # Get all TensorBoard log files inside "tb"
    logfiles = list(tb_dir.iterdir())
    if not logfiles:
        raise ValueError(f"No log files found in {tb_dir}")

    data = []
    for logfile in logfiles:
      ea = event_accumulator.EventAccumulator(str(logfile))
      ea.Reload()
      df = pd.DataFrame(ea.Scalars("evaluation/average_eval_scores"))
      arr = df[["step", "value"]].to_numpy()
      data.append(arr)
    self.data = minimum_truncate_array_list(data)

  @property
  def mean(self):
    return np.mean(self.data, axis=0)

  @property
  def std_dev(self):
    return np.std(self.data, axis=0)


def cross_shortstick(savename):
  """Aggregates returns across experiment seeds and generates a figure."""
  # Note: Append baselines or other methods to this list.
  experiments = [
      Experiment(
          # Note: replace with an actual experiment path.
          path="/home/liannello/xirl/experiment_results/Egocentric/Results/env_name=SweepToTop-Gripper-State-Allo-TestLayout-v0_reward=sparse_env_uid=color_reward_2/2",
          # Note: You can customize the below attributes to your liking.
          name="XIRL",
          color="tab:red",
          linestyle="solid",
      ),
  ]

  _, ax = plt.subplots(1, constrained_layout=True)
  for experiment in experiments:
    return_mean = experiment.mean
    return_stddev = experiment.std_dev

    return_mean_x = return_mean[:, 0] / 1_000
    return_mean_y = return_mean[:, 1]

    ax.plot(
        return_mean_x,
        return_mean_y,
        lw=2,
        label=experiment.name,
        color=experiment.color,
        linestyle=experiment.linestyle,
    )
    ax.fill_between(
        return_mean_x,
        return_mean_y + _STD_DEV_FRAC.value * return_stddev[:, 1],
        return_mean_y - _STD_DEV_FRAC.value * return_stddev[:, 1],
        alpha=0.2,
        color=experiment.color,
    )

  ax.set_xlabel("Steps (thousands)")
  ax.set_ylabel("Success Rate")
  ax.set_title("Gripper")

  ax.legend(loc="lower right")
  ax.grid(linestyle="--", linewidth=0.5)

  plt.savefig(f"{_PLOT_DIR.value}/{savename}.pdf", format="pdf")
  plt.savefig(
      f"{_PLOT_DIR.value}/{savename}.png",
      format="png",
      dpi=_MATPLOTLIB_DPI.value,
  )
  plt.close()


def main(_):
  os.makedirs(_PLOT_DIR.value, exist_ok=True)
  update_plotting_params(_MATPLOTLIB_SIZE.value)
  cross_shortstick("Success_rate_EnvPolicy_2M_Fold2")


if __name__ == "__main__":
  app.run(main)
