"""Unified Training: Combined Stage 1 and Stage 2 in a single process."""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import open_clip

from datasets import FewShotDataset
from losses import cross_entropy_loss, scl_losses
from utils import build_text_features, set_trainable, get_device


class Trainer:
    """CLIP-AST trainer that combines Stage 1 and Stage 2 into a single process."""
    
    def __init__(self, dataset: FewShotDataset, args):
        self.device = get_device(args.device)
        print(f"[CLIP-AST] Using device: {self.device}")
        
        # Model setup
        self.model, self.process_train, self.process_val = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai", device=self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")

        # Dataset setup
        self.dataset = dataset
        fewshot_dataset, eval_dataset = dataset.get_fewshot_datasets()
        self.loader = torch.utils.data.DataLoader(
            fewshot_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        self.eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        
        # Text features
        classnames = dataset.get_classnames()
        self.text_features_frozen = build_text_features(
            self.model, self.tokenizer, classnames, self.device
        )
        
        # Training parameters
        self.stage1_epochs = args.stage1_epochs
        self.stage2_epochs = args.stage2_epochs
        self.total_epochs = self.stage1_epochs + self.stage2_epochs
        self.k = args.k  # Top-K parameters per block for Stage 2
        self.eval_freq = args.eval_freq
        self.logit_scale = self.model.logit_scale.exp().detach()
        self.lmbd_img, self.lmbd_txt, self.lmbd_kl = args.lmbd
        
        # Output path
        self.out = Path(args.out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize for Stage 1 (transformer blocks only)
        self._setup_stage1()
        
    def _setup_stage1(self):
        """Setup for Stage 1: Fine-tune transformer blocks only."""
        print("[CLIP-AST] Setting up Stage 1: Transformer fine-tuning")
        
        def _trainable(name):
            return ("visual.transformer" in name) or ("transformer" in name and "visual" not in name)
        
        set_trainable(self.model, _trainable)
        n_train = sum(p.requires_grad for p in self.model.parameters())
        print(f"[CLIP-AST] Stage 1: {n_train} trainable parameters (transformer blocks)")
        
        # Optimizer for Stage 1
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-5,  # Stage 1 learning rate
            weight_decay=1e-4
        )
        
    def _transition_to_stage2(self):
        """Transition from Stage 1 to Stage 2: Compute importance and select parameters."""
        print("[CLIP-AST] Transitioning to Stage 2: Computing parameter importance...")
        
        # Extract AdamW statistics from current optimizer
        state = self.optimizer.state_dict()["state"]
        name_to_param = {name: p for name, p in self.model.named_parameters()}
        scores: dict[str, float] = {}
        eps = 1e-8
        
        # Create mapping from optimizer state indices to parameter names
        optimizer_params = self.optimizer.param_groups[0]['params']
        param_to_name = {id(p): name for name, p in name_to_param.items()}
        
        # Compute importance scores
        for pid, s in state.items():
            if "exp_avg_sq" not in s:
                continue
            v = s["exp_avg_sq"]  # AdamW second moment
            score = torch.mean(1.0 / torch.sqrt(v + eps)).item()
            
            # Map optimizer state index to parameter name
            if pid < len(optimizer_params):
                param = optimizer_params[pid]
                param_id = id(param)
                if param_id in param_to_name:
                    param_name = param_to_name[param_id]
                    scores[param_name] = score
        
        # Group parameters by transformer block
        def group_key(n):
            parts = n.split(".")
            if parts[0] == "visual":
                block = f"visual_{parts[3]}"  # resblocks.<idx>
                sub = ".".join(parts[4:])
            elif parts[0] == "transformer":  # text encoder
                block = f"text_{parts[2]}"
                sub = ".".join(parts[3:])
            else:
                block, sub = None, None
            return block, sub

        grouped: defaultdict[str, List[Tuple[str, float]]] = defaultdict(list)
        for name, sc in scores.items():
            blk, sub = group_key(name)
            if blk is not None and sub is not None:
                grouped[blk].append((name, sc))

        # Select top-K parameters per block
        trainable_names = set()
        for blk, lst in grouped.items():
            lst.sort(key=lambda x: x[1], reverse=True)  # high importance first
            for name, _ in lst[:self.k]:
                trainable_names.add(name)

        # Apply new parameter selection
        set_trainable(self.model, lambda n: n in trainable_names)
        n_train = sum(p.requires_grad for p in self.model.parameters())
        print(f"[CLIP-AST] Stage 2: Selected {n_train} trainable parameters (top-{self.k} per block)")
        
        # Create frozen model copy for SCL
        self.frozen_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        
        # Create new optimizer with selected parameters and Stage 2 learning rate
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-6,  # Stage 2 learning rate (lower)
            weight_decay=1e-4
        )
        
        # Learning rate scheduler for Stage 2
        total_steps_stage2 = self.stage2_epochs * len(self.loader)
        warmup_steps_stage2 = int(0.1 * total_steps_stage2)  # 10% warmup
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps_stage2
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps_stage2 - warmup_steps_stage2
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps_stage2]
        )
        
    def _train_stage1_epoch(self, epoch):
        """Train one epoch of Stage 1."""
        stage_1_model = copy.deepcopy(self.model)
        stage_1_model.train()
        
        for imgs, labels in tqdm(self.loader, desc=f"Stage 1 - Epoch {epoch}"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            # Forward pass
            img_feat = self.model.encode_image(imgs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            
            # Cross-entropy loss only
            loss = cross_entropy_loss(
                img_feat, self.text_features_frozen, labels, self.logit_scale
            )            
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()                              
        
    def _train_stage2_epoch(self, epoch):
        """Train one epoch of Stage 2."""
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_scl = 0
        correct = 0
        total = 0
        num_batches = 0
        
        for imgs, labels in tqdm(self.loader, desc=f"Stage 2 - Epoch {epoch}"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            # Forward pass - trainable model
            img_feat = self.model.encode_image(imgs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # Forward pass - frozen model for SCL
            with torch.no_grad():
                img_feat_frozen = self.frozen_model.encode_image(imgs)
                img_feat_frozen = img_feat_frozen / img_feat_frozen.norm(dim=-1, keepdim=True)

            # Cross-entropy loss
            ce_loss = cross_entropy_loss(
                img_feat, self.text_features_frozen, labels, self.logit_scale
            )

            # Self-consistency losses
            img_l1, txt_l1, kl = scl_losses(
                img_feat, self.text_features_frozen,
                img_feat_frozen, self.text_features_frozen,
                self.logit_scale
            )
            
            scl_loss = self.lmbd_img * img_l1 + self.lmbd_txt * txt_l1 + self.lmbd_kl * kl
            total_loss_batch = ce_loss + scl_loss

            # Calculate training accuracy
            with torch.no_grad():
                logits = self.logit_scale * img_feat @ self.text_features_frozen.T
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += total_loss_batch.item()
            total_ce += ce_loss.item()
            total_scl += scl_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_ce = total_ce / num_batches
        avg_scl = total_scl / num_batches
        accuracy = 100.0 * correct / total
        current_lr = self.scheduler.get_last_lr()[0]
        print(f"[CLIP-AST] Stage 2 - Epoch {epoch}: Total = {avg_loss:.4f}, CE = {avg_ce:.4f}, SCL = {avg_scl:.4f}, Train Accuracy = {accuracy:.2f}%, LR = {current_lr:.2e}")

    def _evaluate(self, epoch):
        """Evaluate the model on the evaluation dataset."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.eval_loader, desc="Evaluating"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                # Forward pass
                img_feat = self.model.encode_image(imgs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                
                # Compute logits
                logits = self.logit_scale * img_feat @ self.text_features_frozen.T
                predictions = logits.argmax(dim=-1)
                
                # Update accuracy
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total        
        print(f"[CLIP-AST] Epoch {epoch}: Evaluation Accuracy = {accuracy:.2f}%")
        
        self.model.train()  # Switch back to training mode

    def run(self):
        """Run the unified training process."""
        print(f"[CLIP-AST] Starting training: {self.stage1_epochs} Stage 1 epochs + {self.stage2_epochs} Stage 2 epochs")
        
        # Stage 1: Transformer training to find parameter importances
        for epoch in range(self.stage1_epochs):
            self._train_stage1_epoch(epoch)
        
        # Transition to Stage 2
        self._transition_to_stage2()
        
        # Stage 2: Adaptive selective fine-tuning
        for epoch in range(self.stage2_epochs):
            self._train_stage2_epoch(epoch)
            
            # Evaluation
            if epoch+1 % self.eval_freq == 0:
                self._evaluate(epoch)

        # Save final model
        torch.save({"model": self.model.state_dict()}, self.out)
        print(f"[CLIP-AST] Training complete! Saved model to {self.out}") 