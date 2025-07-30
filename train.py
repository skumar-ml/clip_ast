"""Training script for CLIP-AST using DASSL framework."""

import argparse
import torch
from pathlib import Path

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DataManager

from trainers.clip_ast import CLIPAST
import datasets.caltech101


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    """Reset config based on arguments."""
    if args.root:
        cfg.DATASET.ROOT = args.root
    
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    if args.resume:
        cfg.RESUME = args.resume
    
    if args.seed:
        cfg.SEED = args.seed    
    
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    
    # CLIP-AST specific parameters
    if hasattr(args, 'stage1_epochs'):
        cfg.STAGE1_EPOCHS = args.stage1_epochs
    
    if hasattr(args, 'k'):
        cfg.K = args.k

    if hasattr(args, 'random_selection'):
        cfg.RANDOM_SELECTION = args.random_selection


def extend_cfg(cfg):
    """Add CLIP-AST specific configuration."""
    cfg.RANDOM_SELECTION = False
    cfg.RANDOM_SEED = 42
    cfg.STAGE1_EPOCHS = 1    
    cfg.K = 6
    cfg.LMBD = 0.0
    cfg.DEVICE = "cuda:0"  # Default device
    
    # Add OPTIM.EPS to support AdamW epsilon parameter
    cfg.OPTIM.EPS = 1e-5

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

def setup_cfg(args):
    """Setup configuration."""
    cfg = get_cfg_default()
    extend_cfg(cfg)
    
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    
    # 2. From the method config file  
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # 3. From input arguments
    reset_cfg(cfg, args)
    
    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup"
    )
    parser.add_argument("--trainer", type=str, default="CLIPAST", help="name of trainer")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode"
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line"
    )
    
    # CLIP-AST specific arguments
    parser.add_argument("--stage1-epochs", type=int, default=1, help="Stage 1 epochs")    
    parser.add_argument("--k", type=int, default=6, help="Top-K parameters per block")
    parser.add_argument("--random-selection", action="store_true", help="Use random parameter selection") 
    
    args = parser.parse_args()
    
    cfg = setup_cfg(args)
    
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    
    trainer = build_trainer(cfg)
    
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    
    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    main() 