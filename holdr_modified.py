import torch
import torch.nn.functional as F
from xirl.trainers.base import Trainer
from typing import Dict, List, Union
import pdb
import json
import os
from collections import defaultdict

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class HOLDRTrainer(Trainer):
    """Trainer implementing HOLD-R loss.

    The model learns temporal structure by predicting the distance between frames
    in embedding space proportional to their true temporal distance.
    """

    def __init__(self, model, optimizer, device, config):
        super().__init__(model, optimizer, device, config)
        self.temperature = config.loss.holdr.temperature if hasattr(config.loss, "holdr") else 1.0
        self.contrastive_weight = config.loss.holdr.contrastive_weight if hasattr(config.loss, "holdr") else 1.0
        self.distance_subtask_means_weight = config.loss.holdr.distance_subtask_means_weight if hasattr(config.loss, "holdr") else 1.0
        self.distance_frames_before_subtask_weight = config.loss.holdr.distance_frames_before_subtask_weight if hasattr(config.loss, "holdr") else 1.0
        self.hodlr_loss_weight = config.loss.holdr.hodlr_loss_weight if hasattr(config.loss, "holdr") else 1.0
        with open(config.loss.holdr.subtask_json_path, "r") as f:
            raw = json.load(f)

        self.subtask_map = {}
        for vid_id, frame_paths in raw.items():
            for subtask_id, frame_path in enumerate(frame_paths):
                # Extract frame idx (e.g., '32.png' -> 32)
                frame_idx = int(os.path.basename(frame_path).split(".")[0])
                video_name = "/".join(frame_path.split("/")[-3:-1])  # e.g., "gripper/0"
                self.subtask_map[(video_name, frame_idx)] = subtask_id

    def compute_loss(self, embs: torch.Tensor, batch: BatchType) -> torch.Tensor:
        """
        Args:
            embs: torch.Tensor of shape (B, T, D), where B is batch size, 
                  T is number of frames per video, D is embedding dimension.
            batch: dict containing at least 'frame_idxs' with shape (B, T)
        """
       
        B, T, D = embs.shape  # embs: [batch_size, num_frames, embedding_dim]
        device = embs.device
        loss = 0.0
        holdr_loss = 0.0
        frame_idxs = batch["frame_idxs"].to(device)  # (B, T)
        
        distance_subtask_means_loss = 0.0
        distance_frames_before_subtask_loss = 0.0
        subtask_embeddings = defaultdict(list)
        
        for i in range(B):
            emb = embs[i]            
            idxs = frame_idxs[i].float()     
            # Compute pairwise embedding distances
            emb_dists = torch.cdist(emb, emb, p=2) / self.temperature
            
            # Compute ground-truth time distances considering the frame indices
            time_dists = torch.cdist(idxs.unsqueeze(1), idxs.unsqueeze(1), p=1)
            
            # Create a mask to consider only upper triangular part of the distance matrix
            # This is to ignore self-distances and lower triangular part
            # We use the upper triangular part because we want to predict distances on the future frames and not
            # considering also the past frames.
            mask = torch.triu(torch.ones_like(time_dists), diagonal=1).bool()

            # Mean squared error between predicted and ground-truth distances
            holdr_loss += F.mse_loss(emb_dists[mask], time_dists[mask])
            
            vid_path = batch["video_name"][i]
            video_name = "/".join(vid_path.split("/")[-2:])  # e.g., "gripper/0"
            
            for j,t in enumerate(idxs):
                key = (video_name, int(t.item()))
                if key in self.subtask_map:
                    subtask_id = self.subtask_map[key]
                    subtask_embeddings[subtask_id].append(emb[j])
        
        # Compute the mean embedding for each subtask
        subtask_means = {}
        for subtask_id, emb_list in subtask_embeddings.items():
            if len(emb_list) > 0:
                subtask_means[subtask_id] = torch.stack(emb_list, dim=0).mean(dim=0)
                
        for subtask_id, emb_list in subtask_embeddings.items():
            if len(emb_list) < 2:
                continue
            embs_tensor = torch.stack(emb_list, dim=0)
            mean_emb = embs_tensor.mean(dim=0)
            for e in emb_list:
                target = torch.tensor([1.0], device=e.device)  # label: similar
                distance_subtask_means_loss += F.cosine_embedding_loss(
                    e.unsqueeze(0),
                    mean_emb.unsqueeze(0),
                    target
                )
        
        
        # --- N-pair contrastive loss ---
        contrastive = 0.0
        n_pair_count = 0
        all_subtask_ids = sorted(subtask_means.keys())
        num_subtasks = len(all_subtask_ids)

        for i in range(B):
            emb = embs[i]
            idxs = frame_idxs[i].long()
            vid_path = batch["video_name"][i]
            video_name = "/".join(vid_path.split("/")[-2:])
            
            # AGGREGATING LOSS
            for j, t in enumerate(idxs):
                t_int = int(t.item())
                key = (video_name, t_int)

                if key in self.subtask_map:
                    subtask_id = self.subtask_map[key]
                    for k in range(j): 
                        prev_emb = emb[k]
                        target = torch.tensor([1.0], device=prev_emb.device)
                        distance_frames_before_subtask_loss += F.cosine_embedding_loss(
                            prev_emb.unsqueeze(0),
                            subtask_means[subtask_id].unsqueeze(0),
                            target
                        ) 
                        
            # N-PAIR CONTRASTIVE LOSS
            # Find all subtask boundary frames for this video
            boundaries = sorted([
                frame for (v, frame), sid in self.subtask_map.items() if v == video_name
            ])
            
            for j, t in enumerate(idxs):
                t_int = int(t.item())
                # Find which segment this frame belongs to
                next_subtask = None
                for seg, boundary in enumerate(boundaries):
                    if t_int <= boundary:
                        next_subtask = seg
                        break
                if next_subtask is None or next_subtask >= num_subtasks:
                    continue  # No next subtask available
                anchor = emb[j]
                # Positive: mean of the next subtask
                pos_subtask_id = all_subtask_ids[next_subtask]
                if pos_subtask_id not in subtask_means:
                    continue
                positive = subtask_means[pos_subtask_id]
                # Negatives: means of all other subtasks except the positive
                neg_subtask_ids = [sid for sid in all_subtask_ids if sid != pos_subtask_id]
                
                # print(f"Positive subtask ID: {pos_subtask_id}, Negatives: {neg_subtask_ids}, Video: {video_name}, frame: {t_int}")
                if not neg_subtask_ids:
                    continue
                negatives = torch.stack([subtask_means[sid] for sid in neg_subtask_ids], dim=0)
                
                
                # Normalize for cosine similarity (optional, but recommended)
                anchor_norm = F.normalize(anchor, dim=0)
                positive_norm = F.normalize(positive, dim=0)
                negatives_norm = F.normalize(negatives, dim=1)

                # Compute dot products
                pos_dot = torch.matmul(anchor_norm, positive_norm)  # scalar
                neg_dots = torch.matmul(negatives_norm, anchor_norm)  # (num_neg,)

                logits = torch.cat([pos_dot.unsqueeze(0), neg_dots], dim=0) / self.temperature
                log_prob = logits[0] - torch.logsumexp(logits, dim=0)
                contrastive += -log_prob
                n_pair_count += 1

        if n_pair_count > 0:
            contrastive /= n_pair_count
            
            
        holdr_loss /= B
        total_loss = self.hodlr_loss_weight * holdr_loss + self.contrastive_weight * contrastive + self.distance_subtask_means_weight * distance_subtask_means_loss + self.distance_frames_before_subtask_weight * distance_frames_before_subtask_loss
        # return holdr_loss
        return {
            "holdr_loss": holdr_loss,
            "contrastive": contrastive,
            "distance_subtask_means_loss": distance_subtask_means_loss,
            "distance_frames_before_subtask_loss": distance_frames_before_subtask_loss,
            "total_loss": total_loss
        }
