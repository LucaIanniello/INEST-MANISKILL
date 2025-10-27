from base_configs.pretrain import get_config as _get_config


def get_config():
  """HOLDR config."""

  config = _get_config()

  config.algorithm = "holdr"
  config.optim.train_max_iters = 10_000
  config.frame_sampler.strategy = "uniform"
  config.frame_sampler.uniform_sampler.offset = 0
  config.data.batch_size = 32
  config.frame_sampler.num_frames_per_sequence = 40
  config.model.model_type = "resnet18_linear"
  ##TO BE CHANGED FOR HOLDR ARCHITECTURE
  # config.model.model_type = "resnet50_linear"
  config.model.embedding_size = 128
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False
  
  config.loss.holdr.temperature = 1.0
  config.loss.holdr.subtask_json_path = "/home/liannello/xirl/SubtaskDataset6Subtask/subgoal_frames.json"
  config.loss.holdr.distance_subtask_means_weight = 0.2
  config.loss.holdr.distance_frames_before_subtask_weight = 0.2
  config.loss.holdr.contrastive_weight = 0.4
  config.loss.holdr.hodlr_loss_weight = 0.6

  return config
