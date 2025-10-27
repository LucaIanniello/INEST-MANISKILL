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

"""Self supervised models."""

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

import clip
from transformers import GPT2Model, GPT2Config
import pdb

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


class Resnet18LinearEncoderNet(SelfSupervisedModel):
  """A resnet18 backbone with a linear encoder head."""

  def __init__(self, embedding_size, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Visual backbone.
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    layers_ = list(resnet.children())[:-1]
    self.backbone = nn.Sequential(*layers_)

    # Encoder.
    self.encoder = nn.Linear(num_ftrs, embedding_size)


class GoalClassifier(SelfSupervisedModel):
  """A resnet18 backbone with a binary classification head."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Visual backbone.
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    layers_ = list(resnet.children())[:-1]
    self.backbone = nn.Sequential(*layers_)

    # Classification head.
    self.encoder = nn.Linear(num_ftrs, 1)


class Resnet18RawImageNetFeaturesNet(SelfSupervisedModel):
  """A resnet18 backbone with an identity encoder head."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Visual backbone.
    resnet = models.resnet18(pretrained=True)
    layers_ = list(resnet.children())[:-1]
    self.backbone = nn.Sequential(*layers_)

    # Identity encoder.
    self.encoder = nn.Identity()


class Upsampling(nn.Module):
  """Unet upsampling adapted from [1].

  References:
    [1]: https://github.com/milesial/Pytorch-UNet
  """

  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
        nn.BatchNorm2d(in_channels // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x1, x2):
    x1 = self.up(x1)
    diffy = x2.size()[2] - x1.size()[2]
    diffx = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1,
               [diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


@dataclasses.dataclass
class SelfSupervisedReconOutput(SelfSupervisedOutput):
  """Self-supervised output with a reconstruction tensor."""

  reconstruction: Union[np.ndarray, torch.FloatTensor]

  def numpy(self):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      if k != "frames" or k != "reconstruction":
        kwargs[k] = v.cpu().detach().numpy()
    kwargs["frames"] = self.frames.permute(0, 2, 3, 1).cpu().detach().numpy()
    kwargs["reconstruction"] = self.reconstruction.permute(
        0, 2, 3, 1).cpu().detach().numpy()
    return self.__class__(**kwargs)


class Resnet18LinearEncoderAutoEncoderNet(ResNet):
  """Resnet18LinearEncoder with an auxiliary autoencoding path."""

  def __init__(
      self,
      embedding_size,
      num_ctx_frames,
      normalize_embeddings,
      learnable_temp,
  ):
    super().__init__(BasicBlock, [2, 2, 2, 2])

    self.num_ctx_frames = num_ctx_frames
    self.normalize_embeddings = normalize_embeddings
    self.learnable_temp = learnable_temp

    # Load pretrained weights.
    state_dict = load_state_dict_from_url(
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        progress=True,
    )
    self.load_state_dict(state_dict)

    # Embedding head.
    self.fc = nn.Linear(self.fc.in_features, embedding_size)

    # Upsampling path.
    self.up1 = Upsampling(1024, 512 // 2)
    self.up2 = Upsampling(512, 256 // 2)
    self.up3 = Upsampling(256, 128 // 2)
    self.up4 = Upsampling(128, 64)
    self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    # Log-parameterized multiplicative softmax temperature param.
    if learnable_temp:
      self.logit_scale = nn.Parameter(torch.ones([]))

  def encode(self, x):
    # Compute embeddings.
    batch_size, t, c, h, w = x.shape
    x = x.view((batch_size * t, c, h, w))

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x1 = self.layer1(x)  # B, 64, 56, 56
    x2 = self.layer2(x1)  # B, 128, 28, 28
    x3 = self.layer3(x2)  # B, 256, 14, 14
    x4 = self.layer4(x3)  # B, 512, 7, 7

    # Compute embeddings.
    feats = self.avgpool(x4)  # B, 512, 1, 1
    flat_feats = torch.flatten(feats, 1)
    embs = self.fc(flat_feats)
    if self.normalize_embeddings:
      embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
    if self.learnable_temp:
      logit_scale = self.logit_scale.exp()
      embs = logit_scale * embs
    embs = embs.view((batch_size, t, -1))

    return embs, [x1, x2, x3, x4, feats]

  def decode_all_res(self, feature_maps):
    """Decode using all spatial resolutions, a la u-net."""
    x1, x2, x3, x4, feats = feature_maps
    x = self.up1(feats, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    recon = self.out_conv(x)
    return recon

  def decode_lowest_res(self, feature_maps):
    _, _, _, x, _ = feature_maps
    for up_conv in self.up_convs:
      x = F.relu(up_conv(x))
      x = F.interpolate(
          x,
          scale_factor=2,
          mode="bilinear",
          recompute_scale_factor=False,
          align_corners=True,
      )
    x = self.out_conv(x)
    return x

  def forward(self, x):
    embs, feature_maps = self.encode(x)
    recon = self.decode_all_res(feature_maps)
    feats = feature_maps[-1]
    feats = feats.view((embs.shape[0], embs.shape[1], *feats.shape[1:]))
    recon = recon.view((embs.shape[0], embs.shape[1], *recon.shape[1:]))
    return SelfSupervisedReconOutput(
        frames=x,
        feats=feats,
        embs=embs,
        reconstruction=recon,
    )

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
      out = SelfSupervisedReconOutput.merge(out)
    else:
      out = self.forward(x).cpu()
    return out.squeeze(0)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, zero_init_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        if zero_init_bn:
            nn.init.zeros_(self.bn3.weight)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet50(SelfSupervisedModel):
    def __init__(self, embedding_size, num_ctx_frames, normalize_embeddings, learnable_temp):
        super().__init__(
            num_ctx_frames=num_ctx_frames,
            normalize_embeddings=normalize_embeddings,
            learnable_temp=learnable_temp,
        )

        # Load pretrained ResNet50 with better initialization
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()  # Remove final FC layer
        self.backbone = resnet

        # Improved projection head
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, embedding_size)
        )

        # Better initialization
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        batch_size, t, c, h, w = x.shape
        x_flat = x.view(-1, c, h, w).contiguous()
        
        chunk_size = 16 
        feats_list = []
        
        for i in range(0, x_flat.size(0), chunk_size):
            end_idx = min(i + chunk_size, x_flat.size(0))
            x_chunk = x_flat[i:end_idx]
            
            with torch.cuda.amp.autocast():
                feats = self.backbone(x_chunk)
                feats = feats.view(feats.size(0), -1)
                feats = self.encoder(feats)
                feats_list.append(feats)
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        feats_flat = torch.cat(feats_list, dim=0)
        
        # Apply normalization and temperature scaling
        if self.normalize_embeddings:
            feats_flat = F.normalize(feats_flat, p=2, dim=-1)
        if self.learnable_temp:
            feats_flat = self.logit_scale.exp() * feats_flat

        # Reshape to batch format
        embs = feats_flat.view((batch_size, t, -1))
        feats_flat = feats_flat.view((batch_size, t, -1))

        return SelfSupervisedOutput(
            frames=x,
            feats=feats_flat,
            embs=embs
        )
        
# class DINOv2HFLinearEncoderNet(SelfSupervisedModel):
#     def __init__(self, embedding_size, model_name='dinov2_vitb14', *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         # Load official DINOv2 model (faster loading)
#         self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
#         self.backbone.eval()
        
#         # Freeze backbone
#         for param in self.backbone.parameters():
#             param.requires_grad = False
            
#         # Get backbone dimension
#         backbone_dims = {
#             'dinov2_vits14': 384,
#             'dinov2_vitb14': 768,
#             'dinov2_vitl14': 1024,
#             'dinov2_vitg14': 1536
#         }
#         backbone_dim = backbone_dims.get(model_name, 768)
            
#         self.encoder = nn.Linear(backbone_dim, embedding_size)
        
#     def forward(self, x: torch.Tensor) -> SelfSupervisedOutput:
#         batch_size, t, c, h, w = x.shape
#         x_flat = x.view((batch_size * t, c, h, w))
        
#         # Resize to 224x224 if needed
#         if x_flat.shape[-1] != 224:
#             x_flat = F.interpolate(x_flat, size=(224, 224), mode='bilinear', align_corners=False)
        
#         # Normalize to [0,1] range
#         if x_flat.max() > 1.0:
#             x_flat = x_flat / 255.0
            
#         with torch.no_grad():  # Since backbone is frozen
#             feats = self.backbone(x_flat)  # Direct features
        
#         # Apply encoder
#         embs = self.encoder(feats)
#         if self.normalize_embeddings:
#             embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
#         if self.learnable_temp:
#             logit_scale = self.logit_scale.exp()
#             embs = logit_scale * embs

#         # Reshape
#         embs = embs.view((batch_size, t, -1))
#         feats = feats.view((batch_size, t, -1))
        
#         return SelfSupervisedOutput(frames=x, feats=feats, embs=embs)
      
class DINOv2HFLinearEncoderNet(SelfSupervisedModel):
    def __init__(
        self,
        embedding_size: int,
        model_name: str = 'facebook/dinov2-large',
        freeze_backbone: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        from transformers import AutoImageProcessor, AutoModel
            
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.backbone_dim = self.backbone.config.hidden_size
        self.encoder = nn.Linear(self.backbone_dim, embedding_size)
        
        # Proper initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
           
    def forward(self, x: torch.Tensor) -> SelfSupervisedOutput:
        batch_size, t, c, h, w = x.shape
        x_flat = x.view(-1, c, h, w)

        # Process in chunks for memory efficiency
        chunk_size = 4  # Smaller chunks for transformer
        feats_list = []
        
        for i in range(0, x_flat.size(0), chunk_size):
            end_idx = min(i + chunk_size, x_flat.size(0))
            x_chunk = x_flat[i:end_idx]
            
            # Proper preprocessing
            inputs = self.processor(
                images=x_chunk, 
                return_tensors="pt",
                do_resize=True,
                do_center_crop=True,
                size={"height": 224, "width": 224}
            )
            pixel_values = inputs.pixel_values.to(x_chunk.device)

            # Extract features
            with torch.set_grad_enabled(not self.backbone.training):
                outputs = self.backbone(pixel_values=pixel_values)
                feats = outputs.last_hidden_state[:, 0]  # Use CLS token
                feats_list.append(feats)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        feats_flat = torch.cat(feats_list, dim=0)
        
        # Encode with proper normalization
        embs = self.encoder(feats_flat)
        if self.normalize_embeddings:
            embs = F.normalize(embs, p=2, dim=-1)
        if self.learnable_temp:
            embs = self.logit_scale.exp() * embs

        embs = embs.view((batch_size, t, -1))
        feats_flat = feats_flat.view((batch_size, t, -1))
        
        return SelfSupervisedOutput(frames=x, feats=feats_flat, embs=embs)
      
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
      
def load_clip_model(model_name="ViT-B/16", device="cuda" if torch.cuda.is_available() else "cpu"):
    model, preprocess = clip.load(model_name, device=device)
    return model        

@dataclasses.dataclass
class REDSInferOutput:
    embs: np.ndarray

    def numpy(self):
        # If embs is a torch tensor, convert to numpy
        embs = self.embs
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().detach().numpy()
        return REDSInferOutput(embs=embs)
    
class REDSRewardModel(nn.Module):
    def __init__(self, embedding_size, fusion="concat", gpt2_layers=3,  num_ctx_frames=None, normalize_embeddings=None, learnable_temp=None, device=None, **kwargs):
        super().__init__()
        self.clip_model = load_clip_model()
        for param in self.clip_model.parameters():
          param.requires_grad = False
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_img_dim = self.clip_model.visual.output_dim
        clip_txt_dim = self.clip_model.text_projection.shape[1]
        self.img_proj = nn.Linear(clip_img_dim, embedding_size)  # 512 -> 32
        self.text_residual_weight = nn.Parameter(torch.ones(1) * 4.0)
        self.img_residual_weight = nn.Parameter(torch.ones(1) * 4.0)

        self.img_proj = MLP_REDS([embedding_size * 2, embedding_size], input_dim=clip_img_dim)
        self.txt_proj = MLP_REDS([embedding_size * 2, embedding_size], input_dim=clip_txt_dim)

        if fusion == "concat":
            self.fusion_type = "concat"
            fusion_dim = embedding_size
        elif fusion == "sum":
            self.fusion_type = "sum"
            fusion_dim = embedding_size
        else:
            raise ValueError("Unknown fusion type")

        
        gpt2_config = GPT2Config(
            n_embd=fusion_dim,
            n_layer=gpt2_layers,
            n_head=8,
            n_positions=128,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.temporal_decoder = GPT2Model(gpt2_config)
        self.reward_predictor = MLP_REDS(
            hidden_dims=[embedding_size, 1],
            activations=F.relu,
            activate_final=False,
            dropout_rate=None,
            input_dim=embedding_size ,
        )

    def encode_text(self, texts):
      # texts: list of list of strings, shape [B, variable T]
      feats_txt_list = []
      for video in texts:
          tokens = clip.tokenize(video).to(self.device)  # (T, context_length)
          feats_txt = self.clip_model.encode_text(tokens).float()  # (T, D)          
          # Residual blending
          res = torch.sigmoid(self.text_residual_weight)
          adapted = self.txt_proj(feats_txt)
          feats_txt = res * feats_txt + (1 - res) * adapted
            
            # Ensure normalization (CRITICAL FIX)
          feats_txt = F.normalize(feats_txt, dim=-1)
          feats_txt_list.append(feats_txt)

      # feats_txt_list: list of (T, D)
      return feats_txt_list

    def encode_video(self, images):
      B, T, C, H, W = images.shape
      images_flat = images.view(B * T, C, H, W)
      # Resize to 224x224 for CLIP ViT-B/32
      images_flat = F.interpolate(images_flat, size=(224, 224), mode="bilinear", align_corners=False)
      # Ensure input dtype matches CLIP model weights
      if self.clip_model.visual.conv1.weight.dtype == torch.float16:
          images_flat = images_flat.half()
      else:
          images_flat = images_flat.float()
      feats_img = self.clip_model.visual(images_flat).float()
           
      adapted = self.img_proj(feats_img)                              
      res = torch.sigmoid(self.img_residual_weight)
      
      feats_img = res * feats_img + (1 - res) * adapted
      
      # Ensure normalization 
      feats_img = F.normalize(feats_img, dim=-1)
      feats_img = feats_img.view(B, T, -1)

      # Temporal modeling over image features
      # feats_img_t = feats_img.transpose(0, 1)  # (T, B, D)
      # temporal_out = self.temporal_decoder(inputs_embeds=feats_img_t).last_hidden_state  # (T, B, D)
      # temporal_out = temporal_out.transpose(0, 1)  # (B, T, D)
      
      return feats_img
    
    def debug_sequence_length_mismatch(self, video_features, text_features):
      mismatch_indices = []
      for idx, (vid_feat, txt_feat) in enumerate(zip(video_features, text_features)):
          print(f"Sample {idx}: video length {vid_feat.shape[0]}, text length {txt_feat.shape[0]}")
          if vid_feat.shape[0] != txt_feat.shape[0]:
              mismatch_indices.append(idx)
      if mismatch_indices:
          print(f"Mismatches found in samples: {mismatch_indices}")
      else:
          print("No mismatches found in video and text sequence lengths.")
      return mismatch_indices


    def predict_reward(self, video_features, text_features):
        rewards = []
        
        for vid_feat, txt_feat in zip(video_features, text_features):
          
            combined_features = vid_feat + txt_feat
            combined_features = combined_features.unsqueeze(0) # (1, T, D)
            combined_features_t = combined_features.transpose(0, 1)  # (T, 1, D)
            temporal_out = self.temporal_decoder(inputs_embeds=combined_features_t).last_hidden_state
            temporal_out = temporal_out.transpose(0, 1)  # (B, T, D)
            
            temporal_out = temporal_out.squeeze(0)  # (T, D)
            reward = self.reward_predictor(temporal_out)  # (min_len, 1)
            rewards.append(reward.squeeze(-1))

        return rewards


    def forward(self, images, texts, video_names=None):
      B, T = images.shape[:2]
      video_feature = self.encode_video(images)
      text_feature_list = self.encode_text(texts)
      # for idx, (v, t) in enumerate(zip(video_feature, text_feature)):
      #     if v.shape[0] != t.shape[0]:
      #         vid_name = video_names[idx] if video_names is not None else f"index {idx}"
      #         print(f"Frame/text mismatch for video: {vid_name} ({v.shape[0]} vs {t.shape[0]})")
      #         assert v.shape[0] == t.shape[0], f"Frame/text mismatch: {v.shape[0]} vs {t.shape[0]}"
      
      text_features = []
      for txt_feat in text_feature_list:
            if txt_feat.shape[0] < T:
                # Pad with last token if text is shorter
                padding = txt_feat[-1:].expand(T - txt_feat.shape[0], -1)
                txt_feat = torch.cat([txt_feat, padding], dim=0)
            elif txt_feat.shape[0] > T:
                # Truncate if text is longer
                txt_feat = txt_feat[:T]
            text_features.append(txt_feat)
        
      text_features = torch.stack(text_features, dim=0)  
      
            
      reward = self.predict_reward(video_feature, text_features)
      return reward, video_feature, text_features
    
  
    @torch.no_grad()
    def infer(self, images, texts=None, video_names=None):
        
        self.eval()
        if texts is None:
            T = images.shape[1]
            texts = [["move block"] * T]
        # Forward pass
        rewards, video_features, text_features = self.forward(images, texts, video_names)
        
        # Get the temporal embeddings (after combining and temporal modeling)
        B, T = images.shape[:2]
        
        # Apply the same process as in predict_reward to get final embeddings
        final_embeddings = []
        
        for vid_feat, txt_feat in zip(video_features, text_features):
            combined_features = vid_feat + txt_feat
            combined_features = combined_features.unsqueeze(0)  # Add batch dim
            combined_features_t = combined_features.transpose(0, 1)
            temporal_out = self.temporal_decoder(inputs_embeds=combined_features_t).last_hidden_state
            temporal_out = temporal_out.transpose(0, 1).squeeze(0)  # (T, D)
            final_embeddings.append(temporal_out)
        
        # Return embeddings for batch size 1
        if len(final_embeddings) == 1:
            embs = final_embeddings[0]  # (T, D)
        else:
            embs = torch.stack(final_embeddings, dim=0)  # (B, T, D)
        
        return REDSInferOutput(embs=embs.cpu().detach().numpy())
        