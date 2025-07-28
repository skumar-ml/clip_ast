"""Text processing and model utilities."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

def build_text_features(tokenizer, classnames: List[str], prompt_template: str, device: torch.device):
    """Tokenise prompts and get *frozen* text features."""
    prompts = [prompt_template.format(name.replace("_", " ")) for name in classnames]
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
    return tokens  # (C, D)


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