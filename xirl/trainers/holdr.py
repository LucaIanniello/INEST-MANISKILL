import torch
import torch.nn.functional as F
from xirl.trainers.base import Trainer
from typing import Dict, List, Union
import pdb
import json
import os
from collections import defaultdict

BatchType = Dict[str, Union[torch.Tensor, List[str]]]

import os
from collections import defaultdict

class HOLDRTrainer(Trainer):
    def __init__(self, model, optimizer, device, config):
        super().__init__(model, optimizer, device, config)
        self.temperature = config.loss.holdr.temperature if hasattr(config.loss, "holdr") else 1.0
        self.contrastive_weight = config.loss.holdr.contrastive_weight if hasattr(config.loss, "holdr") else 1.0
        self.distance_subtask_means_weight = config.loss.holdr.distance_subtask_means_weight if hasattr(config.loss, "holdr") else 1.0
        self.distance_frames_before_subtask_weight = config.loss.holdr.distance_frames_before_subtask_weight if hasattr(config.loss, "holdr") else 1.0
        self.hodlr_loss_weight = config.loss.holdr.hodlr_loss_weight if hasattr(config.loss, "holdr") else 1.0

        # Load subtask map from json file
        with open(config.loss.holdr.subtask_json_path, "r") as f:
            raw = json.load(f)
        # Map (video_id, frame_idx) => subtask_id
        self.subtask_map = {}
        for video_id, frame_paths in raw.items():
            # Each entry in frame_paths marks an end subtask frame
            for subtask_id, frame_path in enumerate(frame_paths):
                frame_idx = int(os.path.basename(frame_path).split(".")[0])  # Extract frame number
                # Store mapping by video ID and frame index for quick lookup
                self.subtask_map[(video_id, frame_idx)] = subtask_id

    def compute_loss(self, embs: torch.Tensor, batch: BatchType) -> torch.Tensor:
        B, T, D = embs.shape
        device = embs.device
        holdr_loss = 0.0
        distance_subtask_means_loss = 0.0
        distance_frames_before_subtask_loss = 0.0
        contrastive = 0.0
        n_pair_count = 0

        frame_idxs = batch["frame_idxs"].to(device)  # shape (B, T)
        video_names = batch["video_name"]  # list of video IDs, e.g., 'gripper/0'
        
        for i in range(B):
            emb = embs[i]  # (T, D)
            idxs = frame_idxs[i].long()  # (T,)
            video_id = video_names[i]  # e.g. 'gripper/0' or use just video_id if matches JSON keys
            
            # Reinitialize subtask embeddings per batch item to avoid mixing batches
            subtask_embeddings = defaultdict(list)
            
            # Compute pairwise embedding distances and temporal distances for HOLD-R loss
            emb_dists = torch.cdist(emb, emb, p=2) / self.temperature
            time_dists = torch.cdist(idxs.unsqueeze(1).float(), idxs.unsqueeze(1).float(), p=1)
            mask = torch.triu(torch.ones_like(time_dists), diagonal=1).bool()
            holdr_loss += F.mse_loss(emb_dists[mask], time_dists[mask])
            
            # Collect embeddings per subtask for current video
            for j, frame_idx in enumerate(idxs):
                key = (video_id, int(frame_idx.item()))
                if key in self.subtask_map:
                    subtask_id = self.subtask_map[key]
                    subtask_embeddings[subtask_id].append(emb[j])
            
            # Compute mean embeddings per subtask for this batch item
            subtask_means = {}
            for sub_id, emb_list in subtask_embeddings.items():
                if len(emb_list) > 0:
                    subtask_means[sub_id] = torch.stack(emb_list, dim=0).mean(dim=0)

            # Optional: Compute distance to subtask means loss
            # Consider lowering or zeroing this weight if it harms diversity
            for sub_id, emb_list in subtask_embeddings.items():
                if len(emb_list) < 2:
                    continue
                mean_emb = subtask_means[sub_id]
                for e in emb_list:
                    target = torch.tensor([1.0], device=e.device)  # similar label
                    distance_subtask_means_loss += F.cosine_embedding_loss(
                        e.unsqueeze(0), mean_emb.unsqueeze(0), target)
            
            # Distance of frames before subtask loss
            boundaries = sorted([frame for (vid, frame), sid in self.subtask_map.items() if vid == video_id])
            num_subtasks = len(subtask_means)

            for j, frame_idx in enumerate(idxs):
                frame_int = int(frame_idx.item())
                # Find which segment this frame belongs to
                next_subtask = None
                for seg, boundary in enumerate(boundaries):
                    if frame_int <= boundary:
                        next_subtask = seg
                        break
                if next_subtask is None or next_subtask >= num_subtasks:
                    continue
                for k in range(j):
                    prev_emb = emb[k]
                    target = torch.tensor([1.0], device=prev_emb.device)
                    distance_frames_before_subtask_loss += F.cosine_embedding_loss(
                        prev_emb.unsqueeze(0), subtask_means[next_subtask].unsqueeze(0), target)
            
            # N-pair contrastive loss
            all_subtask_ids = sorted(subtask_means.keys())
            for j, frame_idx in enumerate(idxs):
                frame_int = int(frame_idx.item())
                next_subtask = None
                for seg, boundary in enumerate(boundaries):
                    if frame_int <= boundary:
                        next_subtask = seg
                        break
                if next_subtask is None or next_subtask >= num_subtasks:
                    continue
                anchor = emb[j]
                pos_subtask_id = all_subtask_ids[next_subtask]
                if pos_subtask_id not in subtask_means:
                    continue
                positive = subtask_means[pos_subtask_id]
                neg_subtask_ids = [sid for sid in all_subtask_ids if sid != pos_subtask_id]
                if not neg_subtask_ids:
                    continue
                negatives = torch.stack([subtask_means[sid] for sid in neg_subtask_ids], dim=0)

                # Normalize embeddings
                anchor_norm = F.normalize(anchor, dim=0)
                positive_norm = F.normalize(positive, dim=0)
                negatives_norm = F.normalize(negatives, dim=1)

                pos_dot = torch.matmul(anchor_norm, positive_norm)
                neg_dots = torch.matmul(negatives_norm, anchor_norm)
                logits = torch.cat([pos_dot.unsqueeze(0), neg_dots], dim=0) / self.temperature
                log_prob = logits[0] - torch.logsumexp(logits, dim=0)
                contrastive += -log_prob
                n_pair_count += 1

        if n_pair_count > 0:
            contrastive /= n_pair_count
        holdr_loss /= B

        # total_loss = (self.hodlr_loss_weight * holdr_loss)
        total_loss = (self.hodlr_loss_weight * holdr_loss)

        return {
            "holdr_loss": holdr_loss,
            "contrastive": contrastive,
            "distance_subtask_means_loss": distance_subtask_means_loss,
            "distance_frames_before_subtask_loss": distance_frames_before_subtask_loss,
            "total_loss": total_loss
        }
