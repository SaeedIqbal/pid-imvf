import os
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from data.transforms import DeterministicNoisyTransform

class NoisyMNISTMultiViewDataset(Dataset):
    """
    NoisyMNIST multi-view generator.
    Creates 2 views: Original and Deterministic Noisy (fixed seed).
    """
    def __init__(self, root_dir: str = "/home/phd/datasets/noisy_mnist", train: bool = True, transform=None, noise_std: float = 0.15):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.noise_std = noise_std
        
        self.base_dataset = torchvision.datasets.MNIST(
            root=root_dir, 
            train=train, 
            download=True,
            transform=transforms.ToTensor()
        )
        self.num_samples = len(self.base_dataset)
        
        # Initialize deterministic noisy transform
        self.noisy_transform = DeterministicNoisyTransform(noise_std=self.noise_std, seed=42)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        img, label = self.base_dataset[idx]
        
        # View 1: Original
        view1 = img
        # View 2: Deterministic noisy version
        view2 = self.noisy_transform(img)
        
        views = [view1, view2]
        
        if self.transform is not None:
            views = [self.transform(v) for v in views]
            
        return views, label