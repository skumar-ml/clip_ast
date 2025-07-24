from __future__ import annotations

from pathlib import Path
from typing import List

from torchvision.datasets import ImageFolder

from .base import FewShotDataset, _transform_train, _transform_eval


class ImageFolderFewShot(FewShotDataset):
    """Few-shot dataset implementation using ImageFolder."""
    
    def build_full_dataset(self, train: bool = True):
        """Build the complete ImageFolder dataset."""
        transform = _transform_train if train else _transform_eval
        
        return ImageFolder(
            root=str(self.root), 
            transform=transform
        )
    
    def get_classnames(self) -> List[str]:
        """Return class names from ImageFolder."""
        # Create a temporary dataset to get class names
        temp_ds = ImageFolder(root=str(self.root))
        return temp_ds.classes 