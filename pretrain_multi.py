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

"""Launch script for pre-training representations."""

import os.path as osp

from absl import app
from absl import flags
from absl import logging
from base_configs import validate_config
from ml_collections import config_flags
import torch
from torchkit import CheckpointManager
from torchkit import experiment
from torchkit import Logger
from torchkit.utils.py_utils import Stopwatch
from utils import setup_experiment
from xirl import common
import wandb

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_boolean("resume", False, "Whether to resume training.")
flags.DEFINE_boolean("raw_imagenet", False, "")
flags.DEFINE_boolean("wandb", False, "Log on W&B.")

config_flags.DEFINE_config_file(
    "config",
    "base_configs/pretrain.py",
    "File path to the training hyperparameter configuration.",
)


@experiment.pdb_fallback
def main(_):
  import torch
  import torch.distributed as dist
  from torchkit import CheckpointManager
  from torchkit import experiment
  from torchkit import Logger
  from tqdm.auto import tqdm
  import wandb
  import os
  import sys
  import time
  
  pid = os.getpid()
  print(f"[DDP INIT] PID={pid} RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)

  # DDP initialization
  if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
      rank = int(os.environ["RANK"])
      world_size = int(os.environ["WORLD_SIZE"])
      local_rank = int(os.environ.get("LOCAL_RANK", 0))
      dist.init_process_group(backend="nccl", init_method="env://")
      print(f"[DDP INIT] PID={pid} RANK={rank} LOCAL_RANK={local_rank} initializing device (torch.cuda.device_count()={torch.cuda.device_count()})", flush=True)
      torch.cuda.set_device(local_rank)
      device = torch.device(f"cuda:{local_rank}")
      print(f"[DDP INIT] PID={pid} RANK={rank} Set device to cuda:{local_rank} successfully.", flush=True)
  else:
      rank = 0
      world_size = 1
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"[DDP TRAINING START] PID={pid} RANK={rank} DEVICE={device} Starting training loop.", flush=True)

  # Make sure we have a valid config that inherits all the keys defined in the
  # base config.
  validate_config(FLAGS.config, mode="pretrain")

  config = FLAGS.config
  exp_dir = osp.join(config.root_dir, FLAGS.experiment_name)
  
  if rank == 0:
    setup_experiment(exp_dir, config, FLAGS.resume)

    # No need to do any pretraining if we're loading the raw pretrained
    # ImageNet baseline.
    if FLAGS.raw_imagenet:
        return
    
    if FLAGS.wandb:
        wandb.init(project="Allocentric", group="PretrainXirlDinov2Allo_InvisibleRobot", name="PretrainXirlDinov2Allo_InvisibleRobot", mode="online")
        wandb.config.update(FLAGS)
        wandb.run.log_code(".")
        wandb.config.update(config.to_dict(), allow_val_change=True)

  # Synchronize all processes before continuing
  if world_size > 1:
    dist.barrier()

  # Set RNG seeds.
  if config.seed is not None:
    logging.info("Pretraining experiment seed: %d", config.seed)
    experiment.seed_rngs(config.seed)
    experiment.set_cudnn(config.cudnn_deterministic, config.cudnn_benchmark)
  else:
    logging.info("No RNG seed has been set for this pretraining experiment.")

  logging.info("Using device: %s", device)

  # Only rank 0 creates logger
  if rank == 0:
    logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume)
  else:
    logger = None

  # Load factories.
  (
      model,
      optimizer,
      pretrain_loaders,
      downstream_loaders,
      trainer,
      eval_manager,
  ) = common.get_factories(config, device)
  
  # Wrap model in DDP if using multiple GPUs
  if world_size > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
  
  # Only rank 0 handles checkpointing
  if rank == 0:
    checkpoint_dir = osp.join(exp_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
    )
    global_step = checkpoint_manager.restore_or_initialize()
  else:
    checkpoint_manager = None
    global_step = 0
  
  # Broadcast starting step to all processes
  if world_size > 1:
    start_tensor = torch.tensor([global_step], device=device)
    dist.broadcast(start_tensor, src=0)
    global_step = start_tensor.item()

  total_batches = max(1, len(pretrain_loaders["train"]))
  epoch = int(global_step / total_batches)
  complete = False
  stopwatch = Stopwatch()
  
  try:
    while not complete:
      for batch in pretrain_loaders["train"]:
        print(f"[TRAIN LOOP] PID={os.getpid()} RANK={rank} LOCAL_RANK={local_rank} Epoch={epoch} GlobalStep={global_step}", flush=True)
        train_loss = trainer.train_one_iter(batch)

        if not global_step % config.logging_frequency and rank == 0:
          for k, v in train_loss.items():
            logger.log_scalar(v, global_step, k, "pretrain")
          logger.flush()

        if not global_step % config.eval.eval_frequency and rank == 0:
          # Evaluate the model on the pretraining validation dataset.
          valid_loss = trainer.eval_num_iters(
              pretrain_loaders["valid"],
              config.eval.val_iters,
          )
          for k, v in valid_loss.items():
            logger.log_scalar(v, global_step, k, "pretrain")

          # Evaluate the model on the downstream datasets.
          for split, downstream_loader in downstream_loaders.items():
            eval_to_metric = eval_manager.evaluate(
                model,
                downstream_loader,
                device,
                config.eval.val_iters,
            )
            for eval_name, eval_out in eval_to_metric.items():
              eval_out.log(
                  logger,
                  global_step,
                  eval_name,
                  f"downstream/{split}",
              )
              
              if eval_name == "kendalls_tau":
                kendalls_tau = eval_out.scalar
                if FLAGS.wandb:
                  wandb.log({
                      "kendalls_tau": kendalls_tau,
                      "step": global_step,
                      "epoch": epoch,
                  }, step=global_step)
              
        # Save model checkpoint - only rank 0
        if not global_step % config.checkpointing_frequency and rank == 0:
          checkpoint_manager.save(global_step)

        # Synchronize processes
        if world_size > 1:
          dist.barrier()

        # Exit if complete.
        global_step += 1
        if global_step > config.optim.train_max_iters:
          complete = True
          break

        if rank == 0:  # Only rank 0 logs
          time_per_iter = stopwatch.elapsed()
          logging.info(
              "Iter[{}/{}] (Epoch {}), {:.6f}s/iter, Loss: {:.3f}".format(
                  global_step,
                  config.optim.train_max_iters,
                  epoch,
                  time_per_iter,
                  train_loss["train/total_loss"],
              ))
          if FLAGS.wandb:
            wandb.log({
                "train/total_loss": train_loss["train/total_loss"],
                "step": global_step,
                "epoch": epoch,
            }, step=global_step)
            if "reds" in FLAGS.experiment_name:
              wandb.log({
                  "train_reds/epic_loss": train_loss["train/epic_loss"],
                  "train_reds/supcon_loss": train_loss["train/supcon_loss"],
                  "step": global_step,
                  "epoch": epoch,
              }, step=global_step)
            if "holdr" in FLAGS.experiment_name:
              wandb.log({
                  "train_holdr/contrastive_loss": train_loss["train/contrastive_loss"],
                  "train_holdr/holdr_loss": train_loss["train/holdr_loss"],
                  "train_holdr/distance_frames_before_subtask_loss": train_loss["train/distance_frames_before_subtask_loss"],
                  "train_holdr/distance_subtask_means_loss": train_loss["train/distance_subtask_means_loss"],
                  "step": global_step,
                  "epoch": epoch,
              }, step=global_step)
            wandb.log({
              "evaluation loss": valid_loss["valid/total_loss"],
              "step": global_step,
              "epoch": epoch,
            }, step=global_step)
            if "reds" in FLAGS.experiment_name:
              wandb.log({
                  "valid_reds/epic_loss": valid_loss["valid/epic_loss"],
                  "valid_reds/supcon_loss": valid_loss["valid/supcon_loss"],
                  "step": global_step,
                  "epoch": epoch,
              }, step=global_step)
            if "holdr" in FLAGS.experiment_name:
              wandb.log({
                  "valid_holdr/contrastive_loss": valid_loss["valid/contrastive_loss"],
                  "valid_holdr/holdr_loss": valid_loss["valid/holdr_loss"],
                  "valid_holdr/distance_frames_before_subtask_loss": valid_loss["valid/distance_frames_before_subtask_loss"],
                  "valid_holdr/distance_subtask_means_loss": valid_loss["valid/distance_subtask_means_loss"],
                  "step": global_step,
                  "epoch": epoch,
              }, step=global_step)
          stopwatch.reset()
      epoch += 1

  except KeyboardInterrupt:
    print(f"[DDP EXIT] PID={pid} RANK={rank} Caught keyboard interrupt. Saving before quitting.", flush=True)

  finally:
    if rank == 0:
      print(f"[DDP EXIT] PID={pid} RANK={rank} Saving final checkpoint and closing logger.", flush=True)
      checkpoint_manager.save(global_step)
      logger.close()
      if "reds" in FLAGS.experiment_name:     
        # --- SAVE THE FULL MODEL (including CLIP) ---
        # Save the model's state_dict (recommended)
        torch.save(model.state_dict(), osp.join(exp_dir, "reds_model.pth"))
    
    if world_size > 1:
      print(f"[DDP EXIT] PID={pid} RANK={rank} Destroying process group.", flush=True)
      dist.destroy_process_group()

if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_name")
  app.run(main)
