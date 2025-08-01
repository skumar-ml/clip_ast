"""CLIP-AST: Adaptive Selective Fine-tuning using DASSL framework."""

from __future__ import annotations

import copy
import random
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

# CLIP imports
from clip import clip
from clip.model import CLIP

# Import DASSL components
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.nn import functional as F
from trainers.losses import scl_losses
from utils import set_trainable, build_text_features

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class CustomCLIP(nn.Module):
    def __init__(self, model: CLIP, trainable_names: set, text_tokens: torch.Tensor):
        super().__init__()
        # Create model
        self.model = model
        self.trainable_names = trainable_names
        self.text_tokens = text_tokens

    def forward(self, x):
        return self.model.forward(x, self.text_tokens)[0] # Gets logits_per_image from CLIP model

@TRAINER_REGISTRY.register()
class CLIPAST(TrainerX):
    """CLIP-AST trainer that combines Stage 1 and Stage 2 into a single process using DASSL."""            
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames        

        # Build CLIP model
        print(f"Building CLIP model (backbone: {self.cfg.MODEL.BACKBONE.NAME}) on device: {self.device}")
        self.model = load_clip_to_cpu(self.cfg)
        self.model.to(self.device)                        

        # Text tokens
        self.text_tokens = build_text_features(clip.tokenize, classnames, self.dm.dataset.prompt_template, self.device)
        
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Create frozen model copy for SCL BEFORE parameter selection
        self.frozen_model = copy.deepcopy(self.model).eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        
        # Find the sub-blocks that should be trainable for Stage 2
        trainable_names = self._find_stage2_subblocks()

        # Apply parameter selection to self.model
        set_trainable(self.model, lambda n: n in trainable_names)
        n_train = sum(p.requires_grad for p in self.model.parameters())
        print(f"[CLIP-AST] Stage 2: {n_train} trainable parameters selected")

        # Instantiate CustomCLIP model
        self.model = CustomCLIP(copy.deepcopy(self.model), trainable_names, self.text_tokens)
        
        # Create Stage 2 optimizer and scheduler
        self.optim = build_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        
        # Register model with DASSL for Stage 2
        self.register_model("model", self.model, self.optim, self.sched)
            
    def _find_stage2_subblocks(self):
        """Find the sub-blocks that should be trainable for Stage 2.
        
        Returns:
            set: Set of parameter names that should be trainable
        """
        print("[CLIP-AST] Finding Stage 2 sub-blocks...")
        
        if self.cfg.RANDOM_SELECTION:
            print(f"[CLIP-AST] Using random parameter selection (seed: {self.cfg.RANDOM_SEED})")
            return self._random_parameter_selection()
        else:
            print("[CLIP-AST] Using CLIP-AST importance-based selection")
            return self._clipast_parameter_selection()
    
    def _random_parameter_selection(self):
        """Randomly select k parameters per transformer block.
        
        Returns:
            set: Set of parameter names that should be trainable
        """
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
        
        return trainable_names
    
    def _clipast_parameter_selection(self):
        """Use CLIP-AST method: Stage 1 training + importance scoring.
        
        Returns:
            set: Set of parameter names that should be trainable
        """
        # Create a deepcopy of the model for Stage 1 training
        stage1_model = copy.deepcopy(self.model)        
        
        # Setup Stage 1 parameter selection on the copy
        def _trainable(name):
            return ("visual.transformer" in name) or ("transformer" in name and "visual" not in name)
        
        set_trainable(stage1_model, _trainable)
        n_train = sum(p.requires_grad for p in stage1_model.parameters())
        print(f"[CLIP-AST] Stage 1: {n_train} trainable parameters (visual + text transformer blocks)")
        
        # Run Stage 1 training on the copy
        stage1_optim = self._run_stage1_training_on_copy(stage1_model)
        
        # Compute importance scores and select top-K parameters
        trainable_names = self._compute_importance_and_select_params(stage1_model, stage1_optim)
        
        return trainable_names
    
    def _run_stage1_training_on_copy(self, stage1_model : CLIP):
        """Run Stage 1 training on a model copy.
        
        Args:
            stage1_model: Deepcopy of the model for Stage 1 training
            
        Returns:
            torch.optim.Optimizer: The optimizer used for Stage 1 (for importance computation)
        """
        print("[CLIP-AST] Running Stage 1 training...")
        
        # Hardcoded Stage 1 optimizer - user doesn't need to specify
        stage1_optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, stage1_model.parameters()),
            lr=1e-5,  # Stage 1 learning rate
            eps=self.cfg.OPTIM.EPS,
            weight_decay=1e-4
        )
        
        # Stage 1 training loop
        stage1_model.train()
        for epoch in range(self.cfg.STAGE1_EPOCHS):
            print(f"[CLIP-AST] Stage 1 Epoch {epoch + 1}/{self.cfg.STAGE1_EPOCHS}")
            
            total_loss = 0            
            num_batches = 0
            
            for batch in tqdm(self.train_loader_x, desc=f"Stage 1 Epoch {epoch + 1}"):
                input, label = self.parse_batch_train(batch)
                
                # Forward pass
                logits, _ = stage1_model.forward(input, self.text_tokens)
                loss = F.cross_entropy(logits, label)
                
                # Backward pass
                stage1_optim.zero_grad()
                loss.backward()

                # Check which grads are inf or nan
                for name, param in stage1_model.named_parameters():
                    if param.grad is not None:                        
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"[CLIP-AST] WARNING: Inf or NaN gradient in {name}")                

                stage1_optim.step()
                
                total_loss += loss.item()                
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"[CLIP-AST] Stage 1 Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        return stage1_optim
    
    def _group_transformer_params(self):
        """Group transformer parameters by block.
        
        Returns:
            dict: Dictionary mapping block names to lists of parameter names
        """
        grouped = defaultdict(list)
        
        for name, param in self.model.named_parameters():
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
    
    def _compute_importance_and_select_params(self, stage1_model, stage1_optim):
        """Compute parameter importance and select top-K per block.
        
        Args:
            stage1_model: The model used for Stage 1 training
            stage1_optim: The optimizer used for Stage 1 training
            
        Returns:
            set: Set of parameter names that should be trainable
        """
        print("[CLIP-AST] Computing parameter importance and selecting top-K parameters...")
        
        # Extract AdamW statistics
        state = stage1_optim.state_dict()["state"]
        name_to_param = {name: p for name, p in stage1_model.named_parameters()}
        scores = {}
        eps = self.cfg.OPTIM.EPS
        
        # Create mapping from optimizer parameters to names
        optimizer_params = list(filter(lambda p: p.requires_grad, stage1_model.parameters()))
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
            selected_params = param_list[:self.cfg.K]
            for name, score in selected_params:
                trainable_names.add(name)
            print(f"[CLIP-AST] Block {block}: Selected {len(selected_params)}/{len(param_list)} parameters")
        
        return trainable_names    
    
    def forward_backward(self, batch):
        """Forward and backward pass - now only handles Stage 2 with SCL."""
        input, label = self.parse_batch_train(batch)
        
        # Forward pass - trainable model
        img_feat = self.model.model.encode_image(input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        text_feat = self.model.model.encode_text(self.text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        # Forward pass - frozen model for SCL
        with torch.no_grad():
            img_feat_frozen = self.frozen_model.encode_image(input)
            img_feat_frozen = img_feat_frozen / img_feat_frozen.norm(dim=-1, keepdim=True)
            text_feat_frozen = self.frozen_model.encode_text(self.text_tokens)
            text_feat_frozen = text_feat_frozen / text_feat_frozen.norm(dim=-1, keepdim=True)
        
        # Cross-entropy loss
        logit_scale = self.model.model.logit_scale.exp()
        logits = logit_scale * img_feat @ text_feat.t()
        ce_loss = F.cross_entropy(logits, label)
        
        # Self-consistency losses
        img_l1, txt_l1, kl = scl_losses(
            img_feat, text_feat,
            img_feat_frozen, text_feat_frozen,
            logit_scale
        )
        
        scl_loss = self.cfg.LMBD * img_l1 + self.cfg.LMBD * txt_l1 + self.cfg.LMBD * kl
        total_loss = ce_loss + scl_loss
        
        # Backward pass
        self.model_backward_and_update(total_loss)
        
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
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
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