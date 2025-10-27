import abc
import math
from typing import List, Union

import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet
from torch.hub import load_state_dict_from_url

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Callable, Optional

from transformers import GPT2Model, GPT2Config
import clip
import math

def get_1d_sincos_pos_embed(embed_dim, length):
    position = torch.arange(length).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(length, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, D)

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    assert embed_dim % 2 == 0
    pos_y = torch.arange(grid_size).unsqueeze(1).repeat(1, grid_size).reshape(-1)
    pos_x = torch.arange(grid_size).repeat(grid_size)
    pos = torch.stack([pos_y, pos_x], dim=1)

    div_term = torch.exp(torch.arange(0, embed_dim // 2, 2) * (-math.log(10000.0) / (embed_dim // 2)))
    pe = torch.zeros(grid_size * grid_size, embed_dim)

    pe[:, 0::4] = torch.sin(pos[:, 0].unsqueeze(1) * div_term)
    pe[:, 1::4] = torch.cos(pos[:, 0].unsqueeze(1) * div_term)
    pe[:, 2::4] = torch.sin(pos[:, 1].unsqueeze(1) * div_term)
    pe[:, 3::4] = torch.cos(pos[:, 1].unsqueeze(1) * div_term)
    return pe


@dataclasses.dataclass
class SelfSupervisedOutput:
  """The output of a self-supervised model."""

  frames: Union[np.ndarray, torch.FloatTensor]
  feats: Union[np.ndarray, torch.FloatTensor]
  embs: Union[np.ndarray, torch.FloatTensor]

  def squeeze(self, dim):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      kwargs[k] = v.squeeze(dim)
    return self.__class__(**kwargs)

  def cpu(self):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      kwargs[k] = v.cpu()
    return self.__class__(**kwargs)

  def numpy(self):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      if k != "frames":
        kwargs[k] = v.cpu().detach().numpy()
    kwargs["frames"] = self.frames.permute(0, 2, 3, 1).cpu().detach().numpy()
    return self.__class__(**kwargs)

  @classmethod
  def merge(
      cls, output_list):
    kwargs = {}
    for k in dataclasses.asdict(output_list[0]).keys():
      kwargs[k] = torch.cat([getattr(o, k) for o in output_list], dim=1)
    return cls(**kwargs)


class SelfSupervisedModel(abc.ABC, nn.Module):
  """A self-supervised model trained on video data."""

  @abc.abstractmethod
  def __init__(
      self,
      num_ctx_frames,
      normalize_embeddings,
      learnable_temp,
  ):
    super().__init__()

    self.num_ctx_frames = num_ctx_frames
    self.normalize_embeddings = normalize_embeddings
    self.learnable_temp = learnable_temp

    # Log-parameterized multiplicative softmax temperature param.
    if learnable_temp:
      self.logit_scale = nn.Parameter(torch.ones([]))

  def forward(self, x):
    """Forward the video frames through the network.

    Args:
      x: The video frames of shape (B, T, C, H, W). If there are S video frames
        and we are using X context frames, then T = S * X.

    Returns:
      An instance of SelfSupervisedOutput.
    """
    batch_size, t, c, h, w = x.shape
    x_flat = x.view((batch_size * t, c, h, w))
    feats = self.backbone(x_flat)
    feats_flat = torch.flatten(feats, 1)
    embs = self.encoder(feats_flat)
    if self.normalize_embeddings:
      embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
    if self.learnable_temp:
      logit_scale = self.logit_scale.exp()
      embs = logit_scale * embs
    embs = embs.view((batch_size, t, -1))
    feats = feats.view((batch_size, t, -1))
    return SelfSupervisedOutput(frames=x, feats=feats, embs=embs)
    # return {
    #     "frames": x,
    #     "feats": feats,
    #     "embs": embs,
    # }

  @torch.no_grad()
  def infer(
      self,
      x,
      max_batch_size = 128,
  ):
    """Forward at inference with possible very large batch sizes."""
    # Figure out a max batch size that's a multiple of the number of context
    # frames. This is so we can support large videos with many frames.
    lcm = self.num_ctx_frames
    effective_bs = math.floor(max_batch_size / lcm) * lcm
    if x.shape[1] > effective_bs:
      out = []
      for i in range(math.ceil(x.shape[1] / effective_bs)):
        sub_frames = x[:, i * effective_bs:(i + 1) * effective_bs]
        
        out.append(self.forward(sub_frames).cpu())
        # partial_out = SelfSupervisedOutput(**self.forward(sub_frames)).cpu()
        # out.append(partial_out)
      out = SelfSupervisedOutput.merge(out)
    else:
      out = self.forward(x).cpu()
      # out = SelfSupervisedOutput(**self.forward(x)).cpu()
    return out.squeeze(0)


class MLP_REDS(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Optional[Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        activate_final: bool = False,
        dropout_rate: Optional[float] = None,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims) if input_dim is not None else list(hidden_dims)
        for i in range(len(hidden_dims)):
            if i == 0 and input_dim is not None:
                in_dim = input_dim
            else:
                in_dim = dims[i]
            out_dim = dims[i+1] if i+1 < len(dims) else hidden_dims[i]
            linear = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            # Activation and dropout
            if i + 1 < len(hidden_dims) or activate_final:
                if activations is not None:
                    layers.append(nn.ReLU() if activations == F.relu else activations())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ARPV1RewardModel(SelfSupervisedModel):
    def __init__(
        self,
        embedding_size,
        num_ctx_frames,
        normalize_embeddings,
        learnable_temp,
        action_dim=8,
        activation=None,
        activation_final=None,
        max_episode_steps=1000,
    ):
        super().__init__(
            num_ctx_frames=num_ctx_frames,
            normalize_embeddings=normalize_embeddings,
            learnable_temp=learnable_temp,
        )
        
        self.clip_model = load_clip_model()  # Load the CLIP model
        self.action_dim = action_dim
        self.activation = activation
        self.activation_final = activation_final
        self.max_episode_steps = max_episode_steps
        
        self.image_residual_weight = nn.Parameter(torch.ones(1) * 4.0)
        self.text_residual_weight = nn.Parameter(torch.ones(1) * 4.0)

        # Adapters (MLPs)
        clip_out_dim = self.clip_model.visual.output_dim  # Adjust as needed
        self.image_adapter = MLP_REDS(
            hidden_dims=[embedding_size * 2, clip_out_dim],
            activations=F.relu,
            activate_final=False,
            dropout_rate=None,
            input_dim=clip_out_dim,
        )
        self.text_adapter = MLP_REDS(
            hidden_dims=[embedding_size * 2, self.clip_model.text.output_dim],
            activations=F.relu,
            activate_final=False,
            dropout_rate=None,
            input_dim=self.clip_model.text.output_dim,
        )
        self.action_predictor = MLP_REDS(
            hidden_dims=[embedding_size * 2, action_dim],
            activations=F.relu,
            activate_final=False,
            dropout_rate=None,
            input_dim=clip_out_dim * 2 + self.clip_model.text.output_dim,
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.0)

    def get_clip_visual_feature(self, images, normalize=False):
        # images: (B, C, H, W), normalized to [0, 1]
        image_features = self.clip_model.visual(images)
        if normalize:
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-7)
        return image_features

    def get_clip_text_feature(self, tokens, normalize=False):
        text_features = self.clip_model.encode_text(tokens)
        if normalize:
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-7)
        return text_features

    def encode_image(self, images):
        image_features = self.get_clip_visual_feature(images)
        res = torch.sigmoid(self.image_residual_weight)
        adapted = self.image_adapter(image_features)
        image_features = res * image_features + (1 - res) * adapted
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-7)
        return image_features

    def encode_text(self, tokens):
        text_features = self.get_clip_text_feature(tokens)
        res = torch.sigmoid(self.text_residual_weight)
        adapted = self.text_adapter(text_features)
        text_features = res * text_features + (1 - res) * adapted
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-7)
        return text_features

    def predict_action(self, before_image_features, image_features, text_features):
        concat_feature = torch.cat([before_image_features, image_features, text_features], dim=-1)
        a_hat = self.action_predictor(concat_feature)
        return a_hat

    def video_score(self, video_features, text_features):
        # Cosine similarity
        return self.logit_scale * F.cosine_similarity(video_features, text_features, dim=-1)

    def text_score(self, video_features, text_features):
        return self.logit_scale * F.cosine_similarity(text_features, video_features, dim=-1)

    def forward(self, images, tokens):
        # images: (B, C, H, W), tokens: (B, T)
        image_feature = self.encode_image(images)
        text_feature = self.encode_text(tokens)
        concat_feature = torch.cat([image_feature, image_feature, text_feature], dim=-1)
        a_hat = self.action_predictor(concat_feature)
        # For compatibility with SelfSupervisedOutput
        return SelfSupervisedOutput(
            frames=images,
            feats=image_feature,
            embs=a_hat,
        )
        
class RPFRewardModel(ARPV1RewardModel):
    def __init__(
        self,
        embedding_size,
        num_ctx_frames,
        normalize_embeddings,
        learnable_temp,
        action_dim=8,
        activation=None,
        activation_final=None,
        max_episode_steps=1000,
        output_embd_dim=None,
        vision_embd_dim=None,
        num_images=1,
        num_layers = 4, 
        num_heads = 4, 
        max_seq_len = 100,
        **kwargs,
    ):
        super().__init__(
            embedding_size=embedding_size,
            num_ctx_frames=num_ctx_frames,
            normalize_embeddings=normalize_embeddings,
            learnable_temp=learnable_temp,
            action_dim=action_dim,
            activation=activation,
            activation_final=activation_final,
            max_episode_steps=max_episode_steps,
        )
        self.output_embd_dim = output_embd_dim or embedding_size
        self.vision_embd_dim = vision_embd_dim or embedding_size
        self.num_images = num_images

        # Override or add new modules specific to RPFRewardModel
        self.text_adapter = MLP_REDS(
            hidden_dims=[embedding_size * 2, embedding_size],
            activations=F.relu,
            activate_final=False,
            dropout_rate=None,
            input_dim=embedding_size,
        )
        self.text_residual_weight = nn.Parameter(torch.ones(1) * 4.0)
        self.image_input = MLP_REDS(
            hidden_dims=[embedding_size],
            activations=F.relu,
            activate_final=False,
            dropout_rate=None,
            input_dim=embedding_size * num_images,
        )
        
        # Create a GPT2 config with your desired parameters
        config = GPT2Config(
            n_embd=embedding_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_seq_len,
            is_decoder = True,
            add_cross_attention = False,
            use_cache = False, # Optional
            # ...other params as needed
        )

        # Instantiate the model
        self.temporal_decoder = GPT2Model(config)
       
        self.reward_predictor = MLP_REDS(
            hidden_dims=[embedding_size, 1],
            activations=F.relu,
            activate_final=False,
            dropout_rate=None,
            input_dim=embedding_size * 2,
        )

    def encode_text(self, text_features):
        res = torch.sigmoid(self.text_residual_weight)
        adapted = self.text_adapter(text_features)
        text_features = res * text_features + (1 - res) * adapted
        return text_features

    def encode_video(self, image_features, attn_mask=None):
        # image_features: (N, B, T, E)
        N, B, T, E = image_features.shape
        image_features = image_features.permute(1, 2, 0, 3).reshape(B, T, N * E)  # (B, T, N*E)
        image_feature_emb = self.image_input(image_features)  # (B, T, E)
        B, T, D = image_feature_emb.shape
        pos_embed = get_1d_sincos_pos_embed(D, T).to(image_feature_emb.device)
        image_feature_emb = image_feature_emb + pos_embed.unsqueeze(0)  # broadcast to (B, T, D)
        # Temporal decoding
        stacked_inputs = image_feature_emb.permute(1, 0, 2)  # (T, B, E)
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
        decoded_outputs = self.temporal_decoder(inputs_embeds=stacked_inputs, attention_mask=attn_mask)
        video_features = decoded_outputs.last_hidden_state  # shape: (B, T, E)
        # video_features = decoded_outputs.permute(1, 0, 2)  # (B, T, E)
        return video_features

    def predict_reward(self, video_features, text_feature):
        batch_size, seq_length, feat_dim = video_features.shape
        vid_t = video_features.reshape(batch_size, -1)
        reward_input = torch.cat([vid_t, text_feature], dim=-1)
        return self.reward_predictor(reward_input)

    def forward(self, video, phases, attn_mask=None):
        # video: (N, B, T, E), phases: (B, phase_dim)
        text_feature = self.encode_text(phases)
        video_feature = self.encode_video(video, attn_mask=attn_mask)
        reward = self.predict_reward(video_feature, text_feature)
        return SelfSupervisedOutput(
            frames=video,
            feats=video_feature,
            embs=reward,
        )

def load_clip_model(model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
    model, preprocess = clip.load(model_name, device=device)
    return model