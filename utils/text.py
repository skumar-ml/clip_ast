"""Text processing and model utilities."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

_PROMPT_TMPL = "a photo of a {}."

def build_text_features(model, tokenizer, classnames: List[str], device: torch.device):
    """Tokenise prompts and get *frozen* text features."""
    prompts = [_PROMPT_TMPL.format(name.replace("_", " ")) for name in classnames]
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_feat = model.encode_text(tokens)  # (C, D)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat  # (C, D)


def set_trainable(model: nn.Module, predicate):
    """Set parameters as trainable based on a predicate function."""
    for name, p in model.named_parameters():
        p.requires_grad = bool(predicate(name))


def get_device(device_str: str) -> torch.device:
    """Parse device string and return torch.device object."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(device_str)
            # Validate the device is available
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device '{device_str}' requested but CUDA is not available")
            return device
        except Exception as e:
            raise ValueError(f"Invalid device '{device_str}': {e}") 