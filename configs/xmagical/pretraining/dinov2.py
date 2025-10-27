# coding=utf-8

# Copyright 2024 The Google Research Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DINOv2 config."""

from base_configs.pretrain import get_config as _get_config

def get_config():
    """DINOv2 config."""
    config = _get_config()
    
    # Algorithm configuration
    config.algorithm = "tcc"
    
    # Training configuration
    config.optim.train_max_iters = 3_000  # Can be adjusted based on your needs
    config.optim.learning_rate = 5e-5      # DINOv2 typically uses lower learning rates
    config.optim.weight_decay = 0.04       # DINOv2 recommended weight decay
    config.optim.warmup_steps = 1000       # Warmup steps for stable training
    
    # Frame sampling configuration
    config.frame_sampler.strategy = "uniform"
    config.frame_sampler.uniform_sampler.offset = 0
    config.frame_sampler.num_frames_per_sequence = 40
    
    # Model configuration - DINOv2 specific
    config.model.model_type = "dinov2"
    config.model.dinov2_variant = "dinov2_vitb14"  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    config.model.embedding_size = 32        # Larger embedding for DINOv2 features
    config.model.normalize_embeddings = False       # DINOv2 benefits from normalized embeddings
    config.model.learnable_temp = False        # Enable learnable temperature
    config.model.freeze_backbone = True            # Freeze DINOv2 backbone for transfer learning
    config.model.use_cls_token = True              # Use [CLS] token vs patch tokens
    config.model.trust_repo = True                 # Trust PyTorch Hub repository
    
    # DINOv2 advanced options (for advanced implementation)
    config.model.feature_fusion = "cls_only"       # Options: cls_only, patch_mean, cls_patch_concat, cls_patch_sum
    config.model.use_intermediate_layers = False   # Use multi-layer features
    config.model.num_last_blocks = 1              # Number of last blocks if using intermediate layers
    
    # # Data augmentation (DINOv2 is robust to augmentations)
    # config.data.image_size = 224                   # DINOv2 standard input size
    # config.data.crop_size = 224                    # Crop size for training
    # config.data.normalize = True                   # ImageNet normalization
    # config.data.horizontal_flip = True             # Enable horizontal flipping
    # config.data.color_jitter = 0.4                # Color jittering strength
    
    # Loss configuration - adapted for DINOv2 features
    config.loss.tcc.stochastic_matching = False
    config.loss.tcc.loss_type = "regression_mse"   # Can experiment with different loss types
    config.loss.tcc.similarity_type = "l2"     # Cosine similarity works well with normalized embeddings
    config.loss.tcc.softmax_temperature = 0.05     # Lower temperature for DINOv2 features
    
    # DINOv2 specific loss configurations (if implementing DINOv2 training)
    # config.loss.dinov2 = {}
    # config.loss.dinov2.teacher_temp = 0.04         # Teacher temperature
    # config.loss.dinov2.student_temp = 0.1          # Student temperature
    # config.loss.dinov2.center_momentum = 0.9       # Centering momentum
    
    # Batch size and distributed training
    # config.batch_size = 64                         # Adjust based on GPU memory
    # config.gradient_accumulation_steps = 1         # For effective larger batch sizes
    
    # Mixed precision training (recommended for DINOv2)
    config.mixed_precision = True
    config.gradient_clip_norm = 1.0                # Gradient clipping for stability
    
    # Evaluation configuration
    # config.eval.eval_freq = 1000                   # Evaluate every 1000 steps
    # config.eval.save_freq = 2000                   # Save checkpoint every 2000 steps
    
    return config
