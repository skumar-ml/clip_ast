"""Loss functions for CLIP-AST training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, labels):
    """Standard cross-entropy loss for CLIP."""    
    return F.cross_entropy(logits, labels)


def scl_losses(image_feat, text_feat, frozen_img, frozen_txt, logit_scale):
    """Self-consistency losses from PromptSRC (L1 + KL)."""
    img_l1 = F.l1_loss(image_feat, frozen_img.detach())
    txt_l1 = F.l1_loss(text_feat, frozen_txt.detach())

    logits_ft = logit_scale * image_feat @ text_feat.t()
    logits_fr = logit_scale * frozen_img @ frozen_txt.t()
    kl = F.kl_div(
        F.log_softmax(logits_ft, dim=-1),
        F.softmax(logits_fr, dim=-1),
        reduction="batchmean"
    )
    return img_l1, txt_l1, kl 