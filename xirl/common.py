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

"""Functionality common to pretraining and evaluation."""

from typing import Dict
from ml_collections import ConfigDict

import torch
from xirl import factory
from xirl.models import SelfSupervisedModel
import logging

DataLoadersDict = Dict[str, torch.utils.data.DataLoader]
ModelType = SelfSupervisedModel


def get_pretraining_dataloaders(
    config,
    debug = False,
):
  """Construct a train/valid pair of pretraining dataloaders.

  Args:
    config: ConfigDict object with config parameters.
    debug: When set to True, the following happens: 1. Data augmentation is
      disabled regardless of config values. 2. Sequential sampling of videos is
      turned on. 3. The number of dataloader workers is set to 0.

  Returns:
    A dict of train/valid pretraining dataloaders.
  """

  def _loader(split):
    dataset = factory.dataset_from_config(config, False, split, debug)
    batch_sampler = factory.video_sampler_from_config(
        config, dataset.dir_tree, downstream=False, sequential=debug)
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_sampler=batch_sampler,
        num_workers=2 if torch.cuda.is_available() and not debug else 0,
        pin_memory=torch.cuda.is_available() and not debug,
    )

  return {
      "train": _loader("train"),
      "valid": _loader("valid"),
  }


def get_downstream_dataloaders(
    config,
    debug = False,
):
  """Construct a train/valid pair of downstream dataloaders.

  Args:
    config: ConfigDict object with config parameters.
    debug: When set to True, the following happens: 1. Data augmentation is
      disabled regardless of config values. 2. Sequential sampling of videos is
      turned on. 3. The number of dataloader workers is set to 0.

  Returns:
    A dict of train/valid downstream dataloaders
  """

  def _loader(split):
    datasets = factory.dataset_from_config(config, True, split, debug)
    loaders = {}
    for action_class, dataset in datasets.items():
      batch_sampler = factory.video_sampler_from_config(
          config, dataset.dir_tree, downstream=True, sequential=debug)
      loaders[action_class] = torch.utils.data.DataLoader(
          dataset,
          collate_fn=dataset.collate_fn,
          batch_sampler=batch_sampler,
          num_workers=2 if torch.cuda.is_available() and not debug else 0,
          pin_memory=torch.cuda.is_available() and not debug,
      )
    return loaders

  return {
      "train": _loader("train"),
      "valid": _loader("valid"),
  }


def get_factories(
    config,
    device,
    debug = False,
):
  """Feed config to factories and return objects."""
  pretrain_loaders = get_pretraining_dataloaders(config, debug)
  downstream_loaders = get_downstream_dataloaders(config, debug)
  model = factory.model_from_config(config)

  # if torch.cuda.device_count() > 1:
  #     logging.info("Using %d GPUs with DataParallel", torch.cuda.device_count())
  #     device_ids = list(range(torch.cuda.device_count()))
  #     model = torch.nn.DataParallel(model, device_ids=device_ids)
  #     model = model.cuda()
  # else:
  #     model = model.to(device)

  # print("After DataParallel:", next(model.parameters()).device)
  model = model.to(device)
  if "reds_model" in config.model.model_type:
      clip_params = []
      other_params = []
      for name, param in model.named_parameters():
          if "clip_model" in name:
              clip_params.append(param)
          else:
              other_params.append(param)

      optimizer = torch.optim.Adam([
          {"params": clip_params, "lr": 1e-6},      # very low LR for CLIP
          {"params": other_params, "lr": 1e-4},     # higher LR for new layers
      ])
  else:
    optimizer = factory.optim_from_config(config, model)
    
  trainer = factory.trainer_from_config(config, model, optimizer, device)
  eval_manager = factory.evaluator_from_config(config)
  return (
      model,
      optimizer,
      pretrain_loaders,
      downstream_loaders,
      trainer,
      eval_manager,
  )


def get_model(config):
  """Construct a model from a config."""
  return factory.model_from_config(config)
