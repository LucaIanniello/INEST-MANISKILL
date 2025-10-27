# XIRL

[![python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-383/)
[![arXiv](https://img.shields.io/badge/arXiv-2106.03911-b31b1b.svg)](https://arxiv.org/abs/2106.03911)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/google-research/google-research/blob/master/LICENSE)

- [Overview](#overview)
- [Setup](#setup)
- [Datasets](#datasets)
- [Code Navigation](#code-navigation)
- [Experiments: Reproducing Paper Results](#experiments-reproducing-paper-results)
- [Extending XIRL](#extending-xirl)
- [Acknowledgments](#acknowledgments)

<p align="center">
  <img src="./images/teaser.gif" width="75%"/>
</p>

## Overview

Code release for our CoRL 2021 conference paper:

<table><tr><td>
    <strong>
        <a href="https://x-irl.github.io/">
            XIRL: Cross-embodiment Inverse Reinforcement Learning
        </a><br/>
    </strong>
    Kevin Zakka<sup>1,3</sup>, Andy Zeng<sup>1</sup>, Pete Florence<sup>1</sup>, Jonathan Tompson<sup>1</sup>, Jeannette Bohg<sup>2</sup>, and Debidatta Dwibedi<sup>1</sup><br/>
    Conference on Robot Learning (CoRL) 2021
</td></tr></table>

<sup>1</sup><em>Robotics at Google,</em>
<sup>2</sup><em>Stanford University,</em>
<sup>3</sup><em>UC Berkeley</em>

---

This repository serves as a general-purpose library for (a) **self-supervised pretraining** on video data and **(b)** downstream **reinforcement learning** using the learned representations as reward functions. It also contains models, training scripts and config files for reproducing our results and as a reference for implementation details.

Our hope is that the code's modularity allows you to easily extend and build on top of our work. To aid in this effort, we're releasing two additional standalone libraries:

* [x-magical](https://github.com/kevinzakka/x-magical): our Gym-like benchmark extension of MAGICAL geared towards cross-embodiment imitation.
* [torchkit](https://github.com/kevinzakka/torchkit): a lightweight library containing useful PyTorch boilerplate utilities like logging and model checkpointing.

For the latest updates, see: [x-irl.github.io](https://x-irl.github.io)

## Setup

We use Python 3.8 and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for development. To create an environment and install dependencies, run the following steps:

```bash
# Clone and cd into xirl.
git clone https://github.com/LucaIanniello/INEST-IRL

# Create and activate environment.
conda create -n xirl_clone python=3.8
conda activate xirl_clone
# Install dependencies.
pip install -r requirements.txt
```

## Datasets

**X-MAGICAL**

Run the following bash script to download the demonstration dataset for the X-MAGICAL benchmark:

```bash
bash scripts/download_xmagical_dataset.sh
```

The dataset will be located in `/tmp/xirl/datasets/xmagical`. You are free to modify the save destination, just make sure you update `config.data.root` in the pretraining config file &mdash; see `base_configs/pretrain.py`.

## Code Navigation

At a high-level, our code relies on two important but generic python scripts: `pretrain.py` for pretraining and `train_policy.py` for reinforcement learning. We use [ml_collections](https://github.com/google/ml_collections) to parameterize these scripts with experiment-specific config files. **All experiments must use config files that inherit from the base config files in `base_configs/`**. Specifically, pretraining experiments must inherit from `base_configs/pretrain.py` and RL experiments must inherit from `base_configs/rl.py`.

The rest of the codebase is organized as follows:

* `configs/` contains all config files used in our CoRL submission. They inherit from `base_configs/`.
* `xirl/` is the core pretraining codebase.
* `sac/` is the core Soft-Actor-Critic implementation adapted from [pytorch_sac](https://github.com/denisyarats/pytorch_sac).
* `scripts/` contains miscellaneous bash scripts.

## Experiments: Reproducing Paper Results

**Core Scripts**

- [x] Same-embodiment setting (Section 5.1)
    - [x] Pretraining: `python pretrain_xmagical_same_embodiment.py --help`
    - [x] RL: `python rl_xmagical_learned_reward.py --help`
- [x] Cross-embodiment setting (Section 5.2)
    - [x] Pretraining: `python pretrain_xmagical_cross_embodiment.py --help`
    - [x] RL: `python rl_xmagical_learned_reward.py --help`
- [x] RL with environment reward
    - [x] `python rl_xmagical_env_reward.py --help`
- [x] Interactive reward visualization (Section 5.4)
    - [x] `python interact_reward.py --help`

**Misc. Scripts**

- [x] Visualize dataloader for frame sampler debugging
    - [x] `python debug_dataset.py --help`
- [x] Compute goal embedding with a pretrained model
    - [x] `python compute_goal_embedding.py --help`
- [x] Quick n' dirty multi-GPU RL training
    - [x] With environment reward: `bash scripts/launch_rl_multi_gpu.sh`

## Experiment Setup

Refer to `xirl_training.sh` for example experiment setups for pretraining and RL training.
The file to be edited are the following:
- 'xirl_training.sh': contains the commands to run the experiments.
- 'rl_xmagical_learned_reward.py': contains the RL script that executes the training with learned reward. In this file, you can set the algorithm to be used for training (distance_to_goal, goal_classifier, inest, inest_knn, state_intrinsic, reds) and the possibility to resume from an existing experiment. 
- 'train_policy.py': contains the RL training loop. Here you can modify the wandb metrics and parameters.
- 'rl.py': contains the base config for RL experiments. Here you can set the default algorithm for training and the path of the pretraining model to be used.

For the pretraining phase, the type of algorithm used for the inest pretraining is the xirl model. 