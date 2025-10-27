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

"""TCC config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
    config = _get_config()

    config.algorithm = "tcc"
    # Model settings
    config.model.model_type = "resnet50_linear"
    config.frame_sampler.strategy = "uniform"
    config.model.embedding_size = 32 # Increased from 32 for better feature representation
    config.model.normalize_embeddings = False  # Enable normalization for stable training
    config.model.learnable_temp = False  # Allow temperature learning
    
    # Training settings
    config.optim.train_max_iters = 6_000
    config.optim.lr = 5e-5  # Slightly higher learning rate
    config.optim.weight_decay = 1e-4  # Reduced weight decay
    
    # Data settings
    config.data.batch_size = 16  # Smaller batch size due to larger model
    config.frame_sampler.num_frames_per_sequence = 40  # More frames for temporal learning
    

    # config.data_augmentation.image_size = (224, 224)  # Standard ResNet input size
    
    # Loss settings
    config.loss.tcc.stochastic_matching = True
    config.loss.tcc.loss_type = "regression_mse"
    config.loss.tcc.similarity_type = "l2"
    config.loss.tcc.softmax_temperature = 0.05

    return config
