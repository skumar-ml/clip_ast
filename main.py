#!/usr/bin/env python3
"""
CLIP-AST Few-Shot CLI Entry Point
=====================================

This script provides command-line interface for CLIP-AST few-shot learning.

Usage:
    # Unified Training: Combined Stage 1 + Stage 2
    python main.py train --root ~/data --shots 16 --stage1_epochs 1 --stage2_epochs 30 --k 6 --out ckpts/ast_caltech.pt
"""

from __future__ import annotations

import argparse

from datasets import ImageFolderFewShot
from stages import Trainer


def create_dataset(args):
    """Factory function to create the appropriate dataset."""
    # For now we only support Caltech-101, but this makes it easy to add new datasets
    if hasattr(args, 'dataset') and args.dataset != 'caltech101':
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    return ImageFolderFewShot(root=args.root, shots=args.shots)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        "CLIP-AST Few-Shot Learning",
        description="Adaptive Selective Fine-tuning for Few-shot Learning with CLIP"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- Unified Training ---
    p_train = sub.add_parser("train", help="CLIP-AST unified training.")
    p_train.add_argument("--root", type=str, required=True, 
                        help="Dataset root path.")
    p_train.add_argument("--dataset", type=str, default="caltech101",
                        choices=["caltech101"], help="Dataset to use.")
    p_train.add_argument("--shots", type=int, default=16,
                        help="Number of shots per class.")
    p_train.add_argument("--stage1_epochs", type=int, default=1,
                        help="Number of Stage 1 epochs (transformer fine-tuning).")
    p_train.add_argument("--stage2_epochs", type=int, default=30,
                        help="Number of Stage 2 epochs (adaptive selective fine-tuning).")
    p_train.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    p_train.add_argument("--k", type=int, default=6,
                        help="Top-K sub-layers per block to fine-tune in Stage 2.")
    p_train.add_argument("--lmbd", type=float, nargs=3, default=(0.5, 0.5, 1.0),
                        metavar=("IMG", "TXT", "KL"), 
                        help="Weights for SCL losses (image L1, text L1, KL divergence).")
    p_train.add_argument("--out", type=str, default="ast_model.pt",
                        help="Output model path.")
    p_train.add_argument("--device", type=str, default="auto",
                        help="Device to use ('cpu', 'cuda', 'cuda:0', etc. or 'auto' for automatic detection).")
    p_train.add_argument("--eval_freq", type=int, default=5,
                        help="Frequency of evaluation (in epochs).")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create dataset
    dataset = create_dataset(args)
    
    # Run training
    if args.command == "train":
        trainer = Trainer(dataset, args)
        trainer.run()
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
