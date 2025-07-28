#!/usr/bin/env python3
"""
CLIP Zero-Shot Evaluation
========================

This script evaluates the zero-shot accuracy of CLIP on a given dataset
without any fine-tuning, providing a baseline for comparison.

Usage:
    python zero_shot_eval.py --root ~/data --dataset caltech101 --batch_size 64 --device auto
"""

from __future__ import annotations

import argparse
from typing import List

import torch
from tqdm import tqdm
import clip

from datasets import ImageFolderFewShot
from utils import build_text_features, get_device


def evaluate_zero_shot(model, tokenizer, dataset, device: torch.device, batch_size: int = 64) -> float:
    """Evaluate zero-shot accuracy of CLIP on the given dataset.
    
    Args:
        model: CLIP model (pretrained, no fine-tuning)
        tokenizer: CLIP tokenizer
        dataset: Dataset instance (FewShotDataset subclass)
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        
    Returns:
        Zero-shot accuracy as percentage
    """
    print("[Zero-Shot] Setting up evaluation...")
    
    # Get classnames and build text features
    classnames = dataset.get_classnames()
    print(f"[Zero-Shot] Found {len(classnames)} classes: {classnames[:5]}...")
    
    text_features = build_text_features(model, tokenizer, classnames, device)
    print(f"[Zero-Shot] Built text features with shape: {text_features.shape}")
    
    # Create full test dataset (no few-shot split)
    test_dataset = dataset.build_full_dataset(train=False)  # Use eval transforms
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    print(f"[Zero-Shot] Created test dataset with {len(test_dataset)} images")
    
    # Get logit scale from model
    logit_scale = model.logit_scale.exp().detach()
    print(f"[Zero-Shot] Using logit scale: {logit_scale.item():.3f}")
    
    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    
    print("[Zero-Shot] Starting evaluation...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating Zero-Shot"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Encode images
            img_features = model.encode_image(imgs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            # Compute logits
            logits = logit_scale * img_features @ text_features.T
            predictions = logits.argmax(dim=-1)
            
            # Update accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy


def create_dataset(args):
    """Factory function to create the appropriate dataset."""
    if args.dataset == "caltech101":
        return ImageFolderFewShot(root=args.root, shots=16)  # shots doesn't matter for zero-shot
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        "CLIP Zero-Shot Evaluation",
        description="Evaluate zero-shot accuracy of CLIP on various datasets"
    )
    
    parser.add_argument("--root", type=str, required=True,
                       help="Dataset root path.")
    parser.add_argument("--dataset", type=str, default="caltech101",
                       choices=["caltech101"], help="Dataset to evaluate on.")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use ('cpu', 'cuda', 'cuda:0', etc. or 'auto' for automatic detection).")
    parser.add_argument("--model", type=str, default="ViT-B-16",
                       help="CLIP model architecture to use.")
    parser.add_argument("--pretrained", type=str, default="openai",
                       help="CLIP pretrained weights to use.")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("CLIP Zero-Shot Evaluation")
    print("=" * 60)
    
    # Setup device
    device = get_device(args.device)
    print(f"[Zero-Shot] Using device: {device}")
    
    # Load CLIP model
    print(f"[Zero-Shot] Loading CLIP model: {args.model}...")
    model, _ = clip.load(args.model, device=device)
    
    # Create tokenizer wrapper for compatibility
    def tokenizer_wrapper(texts):
        return clip.tokenize(texts)
    tokenizer = tokenizer_wrapper
    print("[Zero-Shot] Model loaded successfully!")
    
    # Create dataset
    print(f"[Zero-Shot] Loading dataset: {args.dataset} from {args.root}")
    dataset = create_dataset(args)
    
    # Evaluate zero-shot accuracy
    accuracy = evaluate_zero_shot(model, tokenizer, dataset, device, args.batch_size)
    
    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} ({args.pretrained})")
    print(f"Total images: {len(dataset.build_full_dataset(train=False))}")
    print(f"Number of classes: {len(dataset.get_classnames())}")
    print(f"Zero-shot accuracy: {accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main() 