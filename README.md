# CLIP-AST: Open-Source Implementation

An unofficial, open-source implementation of [CLIP-AST (Adaptive Selective Transformer)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Adaptive_Parameter_Selection_for_Tuning_Vision-Language_Models_CVPR_2025_paper.html) for few-shot learning, built on the DASSL framework.

## ⚠️ Reproduction Note

I attempted to reproduce the few-shot results from the original CLIP-AST paper on Caltech101 but was unable to achieve their reported performance. The approach appears to be highly sensitive to hyperparameters, particularly:

- SCL (Self-Consistency Loss) coefficients
- Learning rate and training schedule

From my experiments, it seems that these two settings are vital in controlling the degree of overfitting to the train set. From what I can tell, the parameters need to be set for each dataset and shot. Unfortunately, the original paper does not specify these critical hyperparameter settings, making reproduction challenging.

If you find errors in my implementation or are able to reproduce the results, please let me know (GitHub Issues or PRs)!

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install DASSL framework
cd Dassl.pytorch
pip install -e .
cd ..
```

For specific instructions on how to set up the dataset and DASSL, please refer to the docs in the [PromptSRC repo](https://github.com/muzairkhattak/PromptSRC/tree/main).

### Running Experiments

For a single experiment:
```bash
python train.py \
    --root ~/data \
    --config-file configs/trainers/clip_ast_few_shot.yaml \
    --output-dir output/clip_ast \
    --device cuda:0 \
    --stage1-epochs 1 \
    --stage2-epochs 30 \
    --k 6
```

### Hyperparameter Sweeps

Use the provided bash script to run comprehensive hyperparameter sweeps:

```bash
# Run sweep for Caltech101 (sweeps learning rate and SCL coefficients)
bash scripts/few_shot_sweeps.sh caltech101

# The script will automatically test:
# - Shots: 1, 2, 4, 8, 16
# - Learning rates: 5e-8, 1e-7, 5e-7, 1e-6, 5e-6  
# - SCL coefficients: 0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 100.0
```

**Note**: Modify the `DATA` path in `scripts/few_shot_sweeps.sh` to point to your data directory.

## Key Configuration Parameters

- `STAGE1_EPOCHS`: Transformer fine-tuning epochs (default: 1)
- `STAGE2_EPOCHS`: Adaptive selective fine-tuning epochs (default: 30)
- `K`: Top-K parameters per transformer block (default: 6)
- `OPTIM.LR`: Learning rate for Stage 2
- `LMBD`: SCL coefficient weight

## How It Works

1. **Stage 1**: Fine-tune all transformer blocks, collect AdamW statistics
2. **Stage 2**: Select top-K important parameters per block, fine-tune with self-consistency losses

## License

This open-source implementation is provided as-is for research purposes. 