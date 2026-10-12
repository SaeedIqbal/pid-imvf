import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

class MultiViewDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule handling strict inductive splits (10% train, 10% val, 80% test).
    Ensures stratified splitting based on class labels to maintain distribution across splits,
    preventing transductive data leakage.
    """
    def __init__(
        self,
        dataset_class: type,
        root_dir: str = "/home/phd/datasets",
        batch_size: int = 256,
        num_workers: int = 4,
        transform=None
    ):
        super().__init__()
        self.dataset_class = dataset_class
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download data if necessary. Handled by torchvision datasets or assumed present.
        """
        pass

    def setup(self, stage: str = None):
        """
        Assign train/val/test datasets for use in dataloaders.
        """
        if self.full_dataset is None:
            self.full_dataset = self.dataset_class(root_dir=self.root_dir, transform=self.transform)
            
        labels = self.full_dataset.labels.numpy()
        all_indices = np.arange(len(labels))
        
        # Strict stratified 10% train, 10% val, 80% test
        # First split: 10% train, 90% temp
        train_idx, temp_idx = train_test_split(
            all_indices, 
            test_size=0.90, 
            stratify=labels, 
            random_state=42
        )
        
        # Second split: 1/9 of temp (which is 10% of total) for val, 8/9 (80% of total) for test
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=8/9, 
            stratify=labels[temp_idx], 
            random_state=42
        )
        
        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, val_idx)
        self.test_dataset = Subset(self.full_dataset, test_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )