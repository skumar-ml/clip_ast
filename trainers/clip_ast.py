"""CLIP-AST: Adaptive Selective Fine-tuning using DASSL framework."""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import time

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn import functional as F
from tqdm import tqdm
import open_clip

# Import DASSL components
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.data import DataManager
from dassl.utils import MetricMeter, AverageMeter
import torch.nn as nn

from trainers.losses import cross_entropy_loss, scl_losses
from utils import set_trainable, get_device


class CLIPModel(nn.Module):
    """CLIP model wrapper for DASSL integration."""
    
    def __init__(self, model_name="ViT-B-16", pretrained="openai", device=None):
        super().__init__()
        
        # Create CLIP model
        self.clip_model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Get feature dimensions
        self.visual_fdim = self.clip_model.visual.output_dim
        self.text_fdim = self.clip_model.text.output_dim
        
        # Freeze by default - will be unfrozen selectively during training
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def encode_image(self, images):
        return self.clip_model.encode_image(images)
    
    def encode_text(self, text):
        return self.clip_model.encode_text(text)
    
    def forward(self, images, text_tokens=None):
        # Encode images
        image_features = self.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # If text tokens provided, compute similarity
        if text_tokens is not None:
            text_features = self.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.T
            return logits
        
        return image_features


@TRAINER_REGISTRY.register()
class CLIPAST(TrainerX):
    """CLIP-AST trainer that combines Stage 1 and Stage 2 into a single process using DASSL."""
    
    def __init__(self, cfg):
        # Store config
        self.cfg = cfg
        
        # Extract CLIP-AST specific parameters
        self.random_selection = getattr(cfg, 'RANDOM_SELECTION', False)
        self.random_seed = getattr(cfg, 'RANDOM_SEED', 42)
        self.stage1_epochs = getattr(cfg, 'STAGE1_EPOCHS', 1)
        self.stage2_epochs = getattr(cfg, 'STAGE2_EPOCHS', 30)
        self.k = getattr(cfg, 'K', 6)
        self.stage2_lr = getattr(cfg, 'STAGE2_LR', 1e-4)
        self.eval_freq = getattr(cfg, 'EVAL_FREQ', 5)
        
        # SCL loss weights
        lmbd = getattr(cfg, 'LMBD', [0.5, 0.5, 1.0])
        self.lmbd_img, self.lmbd_txt, self.lmbd_kl = lmbd
        
        # Device specification
        self.device_name = getattr(cfg, 'DEVICE', 'cuda:0')
        
        # Training state
        self.current_stage = 1
        self.frozen_model = None
        self.text_tokens = None
        
        # Initialize DASSL trainer
        super().__init__(cfg)
        
        # Setup text tokens after initialization
        self._setup_text_tokens()
        
    def build_data_loader(self):
        """Override to handle dataset setup and text tokens."""
        # Use DASSL's data loading
        super().build_data_loader()
        
        # Setup text tokens after data loading
        if hasattr(self, 'dm') and hasattr(self.dm.dataset, 'classnames'):
            self._setup_text_tokens()
    
    def build_model(self):
        """Build CLIP model."""
        print(f"Building CLIP model on device: {self.device_name}")
        
        # Set device
        self.device = torch.device(self.device_name)
        
        # Create CLIP model
        self.model = CLIPModel(
            model_name="ViT-B-16", 
            pretrained="openai", 
            device=self.device
        )
        self.model.to(self.device)
        
        print(f"# params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup Stage 1 training
        self._setup_stage1()
        
        # Create optimizer
        self.optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-5,  # Stage 1 learning rate
            weight_decay=1e-4
        )
        self.sched = None  # Will be set up during training
        
        # Register model with DASSL
        self.register_model("model", self.model, self.optim, self.sched)
    
    def _setup_text_tokens(self):
        """Setup text tokens for class names."""
        if hasattr(self, 'dm') and hasattr(self.dm.dataset, 'classnames'):
            classnames = self.dm.dataset.classnames
        elif hasattr(self, 'lab2cname'):
            # Convert label to classname dict to list
            classnames = [self.lab2cname[i] for i in range(len(self.lab2cname))]
        else:
            # Fallback - will be set later
            return
            
        if hasattr(self, 'model') and self.model is not None:
            self.text_tokens = self.model.tokenizer(classnames).to(self.device)
    
    def _setup_stage1(self):
        """Setup Stage 1: Fine-tune transformer blocks only."""
        print("[CLIP-AST] Setting up Stage 1: Transformer fine-tuning")
        
        if self.random_selection:
            print(f"[CLIP-AST] Using random parameter selection (seed: {self.random_seed})")
            self._setup_random_selection()
        else:
            def _trainable(name):
                return ("visual.transformer" in name) or ("transformer" in name and "visual" not in name)
            
            set_trainable(self.model.clip_model, _trainable)
            n_train = sum(p.requires_grad for p in self.model.parameters())
            print(f"[CLIP-AST] Stage 1: {n_train} trainable parameters (visual + text transformer blocks)")
    
    def _setup_random_selection(self):
        """Setup random parameter selection for comparison."""
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # Get all transformer parameters grouped by block
        grouped_params = self._group_transformer_params()
        
        # Randomly select k parameters per block
        trainable_names = set()
        for block_name, param_list in grouped_params.items():
            if len(param_list) <= self.k:
                selected = param_list
            else:
                selected = random.sample(param_list, self.k)
            
            trainable_names.update(selected)
            print(f"[CLIP-AST] Block {block_name}: Selected {len(selected)}/{len(param_list)} parameters")
        
        # Apply parameter selection
        set_trainable(self.model.clip_model, lambda n: n in trainable_names)
        n_train = sum(p.requires_grad for p in self.model.parameters())
        print(f"[CLIP-AST] Random selection: {n_train} trainable parameters")
        
        # Create frozen model copy for SCL
        self.frozen_model = copy.deepcopy(self.model).eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False
    
    def _group_transformer_params(self):
        """Group transformer parameters by block."""
        grouped = defaultdict(list)
        
        for name, param in self.model.clip_model.named_parameters():
            if not param.requires_grad:
                continue
                
            parts = name.split(".")
            if parts[0] == "visual" and "transformer" in name:
                if "resblocks" in parts:
                    block_idx = parts[3]
                    block_name = f"visual_block_{block_idx}"
                    grouped[block_name].append(name)
            elif parts[0] == "transformer" and "visual" not in name:
                if "resblocks" in parts:
                    block_idx = parts[2]
                    block_name = f"text_block_{block_idx}"
                    grouped[block_name].append(name)
        
        return grouped
    
    def _transition_to_stage2(self):
        """Transition from Stage 1 to Stage 2."""
        print("[CLIP-AST] Transitioning to Stage 2: Computing parameter importance...")
        
        # Extract AdamW statistics
        state = self.optim.state_dict()["state"]
        name_to_param = {name: p for name, p in self.model.clip_model.named_parameters()}
        scores = {}
        eps = 1e-8
        
        # Create mapping from optimizer parameters to names
        optimizer_params = list(filter(lambda p: p.requires_grad, self.model.clip_model.parameters()))
        param_to_name = {id(p): name for name, p in name_to_param.items() if p.requires_grad}
        
        # Compute importance scores
        for i, param in enumerate(optimizer_params):
            if i in state and "exp_avg_sq" in state[i]:
                v = state[i]["exp_avg_sq"]
                score = torch.mean(1.0 / torch.sqrt(v + eps)).item()
                param_id = id(param)
                if param_id in param_to_name:
                    param_name = param_to_name[param_id]
                    scores[param_name] = score
        
        # Group parameters by transformer block
        def group_key(name):
            parts = name.split(".")
            if parts[0] == "visual":
                block = f"visual_{parts[3]}"
                sub = ".".join(parts[4:])
            elif parts[0] == "transformer":
                block = f"text_{parts[2]}"
                sub = ".".join(parts[3:])
            else:
                block, sub = None, None
            return block, sub
        
        grouped = defaultdict(list)
        for name, score in scores.items():
            block, sub = group_key(name)
            if block is not None and sub is not None:
                grouped[block].append((name, score))
        
        # Select top-K parameters per block
        trainable_names = set()
        for block, param_list in grouped.items():
            param_list.sort(key=lambda x: x[1], reverse=True)
            for name, _ in param_list[:self.k]:
                trainable_names.add(name)
        
        # Apply new parameter selection
        set_trainable(self.model.clip_model, lambda n: n in trainable_names)
        n_train = sum(p.requires_grad for p in self.model.parameters())
        print(f"[CLIP-AST] Stage 2: Selected {n_train} trainable parameters (top-{self.k} per block)")
        
        # Create frozen model copy for SCL
        self.frozen_model = copy.deepcopy(self.model).eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        
        # Create new optimizer with Stage 2 learning rate
        self.optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.stage2_lr,
            weight_decay=1e-4
        )
        
        # Setup learning rate scheduler for Stage 2
        total_steps = self.stage2_epochs * len(self.train_loader_x)
        warmup_steps = int(0.1 * total_steps)
        
        warmup_scheduler = LinearLR(self.optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optim, T_max=total_steps - warmup_steps)
        
        self.sched = SequentialLR(self.optim, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        
        # Update registered model
        self.register_model("model", self.model, self.optim, self.sched)
        
        # Update current stage
        self.current_stage = 2
    
    def forward_backward(self, batch):
        """Forward and backward pass following DASSL pattern."""
        input, label = self.parse_batch_train(batch)
        
        if self.text_tokens is None:
            self._setup_text_tokens()
        
        if self.current_stage == 1:
            return self._forward_backward_stage1(input, label)
        else:
            return self._forward_backward_stage2(input, label)
    
    def _forward_backward_stage1(self, input, label):
        """Stage 1 forward/backward pass."""
        # Encode images and text
        img_feat = self.model.encode_image(input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        text_feat = self.model.encode_text(self.text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        # Cross-entropy loss only
        logit_scale = self.model.clip_model.logit_scale.exp()
        loss = cross_entropy_loss(img_feat, text_feat, label, logit_scale)
        
        # Backward pass
        self.model_backward_and_update(loss)
        
        # Compute accuracy
        logits = logit_scale * img_feat @ text_feat.T
        acc = compute_accuracy(logits, label)[0].item()
        
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }
        
        # Check if we need to transition to Stage 2
        if self.epoch + 1 == self.stage1_epochs and not self.random_selection:
            self._transition_to_stage2()
        
        return loss_summary
    
    def _forward_backward_stage2(self, input, label):
        """Stage 2 forward/backward pass with SCL."""
        
        # Forward pass - trainable model
        img_feat = self.model.encode_image(input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        text_feat = self.model.encode_text(self.text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        # Forward pass - frozen model for SCL
        with torch.no_grad():
            img_feat_frozen = self.frozen_model.encode_image(input)
            img_feat_frozen = img_feat_frozen / img_feat_frozen.norm(dim=-1, keepdim=True)
            text_feat_frozen = self.frozen_model.encode_text(self.text_tokens)
            text_feat_frozen = text_feat_frozen / text_feat_frozen.norm(dim=-1, keepdim=True)
        
        # Cross-entropy loss
        logit_scale = self.model.clip_model.logit_scale.exp()
        ce_loss = cross_entropy_loss(img_feat, text_feat, label, logit_scale)
        
        # Self-consistency losses
        img_l1, txt_l1, kl = scl_losses(
            img_feat, text_feat,
            img_feat_frozen, text_feat_frozen,
            logit_scale
        )
        
        scl_loss = self.lmbd_img * img_l1 + self.lmbd_txt * txt_l1 + self.lmbd_kl * kl
        total_loss = ce_loss + scl_loss
        
        # Backward pass
        self.model_backward_and_update(total_loss)
        
        # Update learning rate scheduler
        if self.sched is not None:
            self.sched.step()
        
        # Compute accuracy
        logits = logit_scale * img_feat @ text_feat.T
        acc = compute_accuracy(logits, label)[0].item()
        
        loss_summary = {
            "loss": total_loss.item(),
            "loss_ce": ce_loss.item(),
            "loss_scl": scl_loss.item(),
            "loss_img_l1": img_l1.item(),
            "loss_txt_l1": txt_l1.item(),
            "loss_kl": kl.item(),
            "acc": acc,
        }
        
        return loss_summary
    
    def parse_batch_train(self, batch):
        """Parse training batch following DASSL pattern."""
        if isinstance(batch, dict):
            # DASSL format
            input = batch["img"]
            label = batch["label"]
        else:
            # Tuple format (img, label)
            input, label = batch
        
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def before_train(self):
        """Setup before training starts."""
        super().before_train()
        
        # Adjust max epochs to account for both stages
        if not self.random_selection:
            self.max_epoch = self.stage1_epochs + self.stage2_epochs
        else:
            self.max_epoch = self.stage2_epochs
            # For random selection, go directly to stage 2
            self.current_stage = 2
            self._setup_random_selection()
    
    def after_epoch(self):
        """Called after each epoch."""
        super().after_epoch()
        
        # Custom evaluation logic if needed
        if (self.epoch + 1) % self.eval_freq == 0:
            print(f"[CLIP-AST] Epoch {self.epoch + 1}: Stage {self.current_stage}") 