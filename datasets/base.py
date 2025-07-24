from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

_IMG_SIZE = 224
_NORMALIZE = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711))

_transform_train = T.Compose([
    T.RandomResizedCrop(_IMG_SIZE, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    _NORMALIZE,
])

_transform_eval = T.Compose([
    T.Resize(_IMG_SIZE + 32),
    T.CenterCrop(_IMG_SIZE),
    T.ToTensor(),
    _NORMALIZE,
])


class FewShotDataset(ABC):
    """Abstract base class for few-shot datasets."""
    
    def __init__(self, root: str | Path, shots: int, seed: int = 42):
        self.root = Path(root)
        self.shots = shots
        self.seed = seed
        
    @abstractmethod
    def build_full_dataset(self, train: bool = True):
        """Build the complete dataset with transforms."""
        pass
    
    @abstractmethod 
    def get_classnames(self) -> List[str]:
        """Return list of class names for this dataset."""
        pass
    
    def get_fewshot_datasets(self) -> Tuple[Subset, Subset]:
        """Create both few-shot training and test datasets.
        
        Returns:
            Tuple[Subset, Subset]: (fewshot_dataset, test_dataset)
                - fewshot_dataset: Contains 'shots' images per class with training transforms
                - test_dataset: Contains all remaining images with eval transforms
        """
        # Build datasets with appropriate transforms
        train_dataset = self.build_full_dataset(train=True)   # Training transforms
        test_dataset = self.build_full_dataset(train=False)   # Eval transforms
        
        # Map class -> indices (use test_dataset since transforms don't affect targets)
        cls_to_indices: defaultdict[int, List[int]] = defaultdict(list)
        for idx, (_, target) in enumerate(test_dataset):
            cls_to_indices[target].append(idx)

        fewshot_indices: List[int] = []
        test_indices: List[int] = []
        rng = random.Random(self.seed)
        
        for cls, idxs in cls_to_indices.items():
            rng.shuffle(idxs)
            fewshot_indices.extend(idxs[:self.shots])
            test_indices.extend(idxs[self.shots:])

        fewshot_subset = Subset(train_dataset, fewshot_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        return fewshot_subset, test_subset 