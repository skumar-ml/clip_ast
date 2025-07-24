# CLIP-AST: Adaptive Selective Fine-tuning for Few-shot Learning

This repository contains a clean, modular implementation of CLIP-AST (Adaptive Selective Transformer) for few-shot learning that combines both training stages into a single, seamless process.

## 🚀 Quick Start

```bash
python main.py train \
    --root ~/data \
    --shots 16 \
    --stage1_epochs 1 \
    --stage2_epochs 30 \
    --k 6 \
    --device auto \
    --out ckpts/ast_caltech.pt
```

## Project Structure

```
clip_ast/
├── __init__.py                 # Package initialization
├── main.py                     # CLI entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── datasets/                   # Dataset implementations
│   ├── __init__.py
│   ├── base.py                 # Abstract dataset base class
│   └── imagefolder.py          # ImageFolder implementation
├── stages/                     # Training implementation
│   ├── __init__.py
│   └── trainer.py             # CLIP-AST trainer
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── text.py                # Text processing utilities
└── losses.py                   # Loss functions (CE + SCL)
```

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Parameters

- `--stage1_epochs`: Number of epochs for transformer fine-tuning (default: 1)
- `--stage2_epochs`: Number of epochs for adaptive selective fine-tuning (default: 30)  
- `--k`: Top-K sublayers per transformer block to fine-tune in Stage 2 (default: 6)
- `--lmbd`: Weights for SCL losses [img_l1, txt_l1, kl] (default: 0.5 0.5 1.0)
- `--eval_freq`: Evaluation frequency in epochs (default: 5)

### Device Selection

```bash
# Automatic device detection (default)
--device auto

# Force CPU usage
--device cpu

# Use specific CUDA device
--device cuda:0
--device cuda:1

# Use any available CUDA device
--device cuda
```

## Key Features

### 🎯 Easy Dataset Extension
The abstract `FewShotDataset` class in `datasets/base.py` makes it straightforward to add new datasets:

```python
from datasets.base import FewShotDataset

class MyDataset(FewShotDataset):
    def build_full_dataset(self, train: bool = True):
        # Build your complete dataset with transforms
        pass
    
    def get_classnames(self) -> List[str]:
        # Return class names for your dataset
        pass
```

### 🔧 Simple Few-shot API
Get both training and test datasets with one simple method:

```python
from datasets import ImageFolderFewShot
from torch.utils.data import DataLoader

# Create dataset
dataset = ImageFolderFewShot(root="~/data", shots=16)

# Get both few-shot training and test datasets
fewshot_dataset, test_dataset = dataset.get_fewshot_datasets()

# Create DataLoaders
train_loader = DataLoader(fewshot_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 📊 Complete Evaluation Setup
- **Training set**: Contains exactly `shots` images per class with training transforms
- **Test set**: Contains all remaining images with evaluation transforms  
- **No overlap**: Training and test sets are completely separate
- **Deterministic**: Same `seed` produces same splits

### 🔧 Modular Architecture
- **Losses**: All loss functions (CE, SCL) in `losses.py`
- **Training**: CLIP-AST trainer combining both stages in `stages/trainer.py`
- **Utils**: Common utilities like text processing in `utils/`
- **Datasets**: Clean dataset abstraction with easy extensibility

## How CLIP-AST Works

### Unified Training Process

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

## Adding New Datasets

1. Create a new file in `datasets/` (e.g., `datasets/imagenet.py`)
2. Inherit from `FewShotDataset` and implement required methods
3. Add your dataset to `datasets/__init__.py`
4. Update the dataset choices in `main.py`

Example:
```python
# datasets/imagenet.py
from .base import FewShotDataset

class ImageNetFewShot(FewShotDataset):
    def build_full_dataset(self, train: bool = True):
        # Your ImageNet loading logic with transforms
        pass
    
    def get_classnames(self) -> List[str]:
        # Return ImageNet class names
        pass
```

## Dependencies

- PyTorch >= 1.12.0
- torchvision >= 0.13.0  
- open_clip_torch >= 2.0.0

## License

This project provides a clean, efficient implementation of CLIP-AST with improved modularity and ease of use. 