import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
import helpers  # local helper functions
from monai.data import list_data_collate

# Dataset

class Prostate3DDataset(Dataset):
    """
    MONAI-friendly dataset that loads NIfTI volumes on-the-fly.
    """
    def __init__(self, data_dicts, transform=None):
        """
        Args:
            data_dicts: list of dicts with keys "image" and "label"
            transform: MONAI transform pipeline
        """
        self.data_dicts = data_dicts
        self.transform = transform

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data = self.data_dicts[idx].copy()  # {'image': path, 'label': path}
        if self.transform:
            data = self.transform(data)
        return data

# Data loader

def get_dataloaders(
    mr_folder,
    label_folder,
    batch_size=2,
    num_workers=1,
    train_spatial_size=(96, 96, 48),
    val_spatial_size=(256, 256, 128),
    num_classes=6,
    seed=42,
):
    """
    Create train, validation, and test dataloaders using MONAI transforms.
    """
    # Get all file paths as dictionaries
    data_dicts = helpers.get_paired_paths(mr_folder, label_folder)

    # Split data: 80% train, 10% val, 10% test
    np.random.seed(seed)
    indices = np.random.permutation(len(data_dicts))
    train_end = int(0.8 * len(data_dicts))
    val_end = int(0.9 * len(data_dicts))

    train_dicts = [data_dicts[i] for i in indices[:train_end]]
    val_dicts = [data_dicts[i] for i in indices[train_end:val_end]]
    test_dicts = [data_dicts[i] for i in indices[val_end:]]

    print(f"Split: {len(train_dicts)} train, {len(val_dicts)} val, {len(test_dicts)} test")

    # MONAI transforms
    base_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]), 
            Resized(
            keys=["image", "label"], 
            spatial_size=train_spatial_size, # (96, 96, 48)
            mode=["trilinear", "nearest"] # trilinear for image, nearest for label
        ),
        ])

    # Combine base steps with advanced training transforms
    train_transform_advanced = helpers.get_train_transforms_monai(
        spatial_size=train_spatial_size, 
        num_classes=num_classes # Pass num_classes
    )
    train_transform = Compose([base_transforms, train_transform_advanced]) 

    # Combine base steps with validation transforms
    val_test_transform_advanced = helpers.get_val_transforms_monai(
        spatial_size=val_spatial_size,
        num_classes=num_classes # Pass num_classes
    )
    val_transform = Compose([base_transforms, val_test_transform_advanced])

    # Datasets
    train_ds = Prostate3DDataset(train_dicts, transform=train_transform)
    val_ds = Prostate3DDataset(val_dicts, transform=val_transform)
    test_ds = Prostate3DDataset(test_dicts, transform=val_transform)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn = list_data_collate
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate
    )

    return train_loader, val_loader, test_loader