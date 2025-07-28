# CLIP-AST: Adaptive Selective Fine-tuning for Few-shot Learning

This repository contains a clean, modular implementation of CLIP-AST (Adaptive Selective Transformer) for few-shot learning using the DASSL framework. The implementation combines both training stages into a single, seamless process with excellent research integration capabilities.

## 🚀 Quick Start

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

## Features

The DASSL-integrated implementation provides:

- **Better experiment management** with DASSL's configuration system
- **Standardized training loop** following domain adaptation best practices
- **Built-in logging and checkpointing** via DASSL's infrastructure
- **Easy integration** with other domain adaptation methods
- **Professional framework** following established research patterns

### DASSL Trainer Features

- **Registered trainer**: Use `CLIPAST` as trainer name in DASSL configs
- **Config-driven**: All parameters configurable via YAML files
- **Two-stage training**: Automatic transition from Stage 1 to Stage 2
- **SCL losses**: Self-consistency losses for improved adaptation
- **Random vs Importance-based selection**: Both parameter selection modes supported

## Installation

Install dependencies:
```bash
# Install base requirements
pip install -r requirements.txt

# Install DASSL framework
cd Dassl.pytorch
pip install -e .
cd ..
```

## Project Structure

```
clip_ast/
├── trainers/
│   └── clip_ast.py             # DASSL-based CLIP-AST trainer
├── configs/
│   └── trainers/
│       └── clip_ast_few_shot.yaml  # DASSL config for CLIP-AST
├── train.py                    # Main training script
├── datasets/                   # Dataset implementations
├── utils/                      # Utility functions
└── README.md                   # This file
```

## Configuration

### DASSL Config Parameters

The DASSL-based trainer supports the following configuration parameters:

```yaml
# Training stages
STAGE1_EPOCHS: 1          # Stage 1 epochs (transformer fine-tuning)
STAGE2_EPOCHS: 30         # Stage 2 epochs (adaptive selective fine-tuning)
K: 6                      # Top-K parameters per transformer block
STAGE2_LR: 0.0001        # Learning rate for Stage 2

# Parameter selection mode  
RANDOM_SELECTION: False   # Use importance-based (False) or random (True) selection
RANDOM_SEED: 42          # Seed for random selection

# Self-consistency loss weights
LMBD: [0.5, 0.5, 1.0]    # [img_l1, txt_l1, kl] weights

# Device selection
DEVICE: "cuda:0"         # Device to use (cuda:0, cuda:1, cpu, etc.)

# Evaluation
EVAL_FREQ: 5             # Evaluation frequency in epochs
```

### Using Custom Datasets with DASSL

The DASSL-based trainer works with DASSL's DataManager:

```python
from trainers.clip_ast import CLIPAST
from dassl.config import get_cfg_default

# Setup config
cfg = get_cfg_default()
cfg.merge_from_file("configs/trainers/clip_ast_few_shot.yaml")

# The trainer automatically uses DASSL's DataManager
trainer = CLIPAST(cfg)
trainer.train()
```

## Training Parameters

### Command Line Parameters

- `--stage1-epochs` / `STAGE1_EPOCHS`: Number of epochs for transformer fine-tuning (default: 1)
- `--stage2-epochs` / `STAGE2_EPOCHS`: Number of epochs for adaptive selective fine-tuning (default: 30)  
- `--k` / `K`: Top-K sublayers per transformer block to fine-tune in Stage 2 (default: 6)
- `--stage2-lr` / `STAGE2_LR`: Learning rate for Stage 2 (default: 1e-4)
- `--eval-freq` / `EVAL_FREQ`: Evaluation frequency in epochs (default: 5)

### DASSL-specific Parameters

- `--config-file`: Path to DASSL trainer config file
- `--dataset-config-file`: Path to DASSL dataset config file
- `--output-dir`: Output directory for logs and checkpoints
- `--device`: Device to use (e.g., cuda:0, cuda:1, cpu) (default: cuda:0)
- `--eval-only`: Run evaluation only
- `--model-dir`: Directory containing saved model for evaluation

## Advanced Usage

### Custom Parameter Selection

You can experiment with different parameter selection strategies:

```bash
# Importance-based selection (default)
python train.py --config-file configs/trainers/clip_ast_few_shot.yaml --device cuda:0

# Random selection for comparison
python train.py --config-file configs/trainers/clip_ast_few_shot.yaml --device cuda:0 --random-selection

# Different K values on different GPU
python train.py --config-file configs/trainers/clip_ast_few_shot.yaml --device cuda:1 --k 12
```

### Custom Loss Weights

Adjust the self-consistency loss weights:

```bash
python train.py \
    --config-file configs/trainers/clip_ast_few_shot.yaml \
    LMBD [1.0, 1.0, 0.5]  # Increase image/text consistency, reduce KL
```

### Device Selection

The implementation supports explicit device selection for better control:

```bash
# Use specific CUDA device (default)
--device cuda:0

# Use different GPU
--device cuda:1

# Force CPU usage
--device cpu
```

**Note**: DataParallel has been removed. For multi-GPU training, specify the desired GPU device explicitly. This provides better control and avoids the complexity of automatic parallelization.

## How CLIP-AST Works

### Two-Stage Training Process

1. **Stage 1 (Epochs 1-N)**: 
   - Fine-tunes transformer blocks only
   - Uses cross-entropy loss
   - Collects AdamW second-moment statistics

2. **Automatic Transition**:
   - Extracts importance scores from AdamW statistics  
   - Selects top-K sublayers per transformer block
   - Creates frozen model copy for self-consistency

3. **Stage 2 (Epochs N+1-M)**:
   - Fine-tunes only selected important parameters
   - Uses cross-entropy + self-consistency losses (SCL)
   - Regularizes against frozen model predictions

### Adaptive Parameter Selection

The method uses AdamW's second-moment estimates (`exp_avg_sq`) to compute parameter importance:
- **Importance score**: `v' = mean(1/√(v + ε))` where `v` is the second moment
- **Selection**: Top-K parameters per transformer block based on importance
- **Rationale**: Parameters with larger accumulated gradients are more important

## Benefits of DASSL Integration

1. **Standardized Framework**: Uses established patterns from domain adaptation research
2. **Better Logging**: Automatic tensorboard logging and metric tracking
3. **Config Management**: YAML-based configuration with easy parameter sweeps
4. **Checkpoint Management**: Automatic model saving and resuming
5. **Evaluation Tools**: Built-in evaluation and testing infrastructure
6. **Research Integration**: Easy to compare with other DA/DG methods

## Evaluation

### Running Evaluation Only

```bash
python train.py \
    --config-file configs/trainers/clip_ast_few_shot.yaml \
    --eval-only \
    --model-dir output/clip_ast \
    --load-epoch 30
```

### Resuming Training

```bash
python train.py \
    --config-file configs/trainers/clip_ast_few_shot.yaml \
    --resume output/clip_ast \
    --output-dir output/clip_ast_resumed
```

## Troubleshooting

### Import Issues

If you get import errors with DASSL:

```bash
# Make sure DASSL is properly installed
cd Dassl.pytorch
pip install -e .

# Check if the trainer is registered
python -c "from trainers.clip_ast import CLIPAST; print('Success!')"
```

### Config Issues

If config parameters aren't being recognized:

1. Check that your config file has the correct parameter names
2. Use the `extend_cfg()` function to add custom parameters
3. Verify parameter types match expected values

### Memory Issues

For large models or datasets:

```bash
# Reduce batch size
python train.py \
    --config-file configs/trainers/clip_ast_few_shot.yaml \
    DATALOADER.TRAIN_X.BATCH_SIZE 8

# Use gradient accumulation (if available)
python train.py \
    --config-file configs/trainers/clip_ast_few_shot.yaml \
    OPTIM.GRAD_ACCUMULATION_STEPS 4
```

## Extending the Framework

### Adding Custom Datasets

1. Create a DASSL-compatible dataset in the `datasets/` directory
2. Register it with DASSL's dataset registry
3. Create a corresponding config file

### Adding Custom Loss Functions

1. Implement loss functions in `trainers/losses.py`
2. Modify the `_forward_backward_stage2()` method in `CLIPAST`
3. Add configuration parameters for loss weights

## Citation

If you use this code, please cite the original CLIP-AST paper and consider citing DASSL for the framework integration.

## License

This project is licensed under the same terms as the original implementation. 