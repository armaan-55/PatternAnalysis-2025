import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from helpers import ImageProcessor, ImageProcessor

TARGET_SHAPE = (128, 128, 128)

# Dataset
class Prostate3DDataset(Dataset):
    """
    Custom dataset that loads NIfTI volumes on-the-fly and uses 
    helpers.ImageProcessor for all transforms and augmentation.
    """
    def __init__(self, data_dicts, is_training, target_shape):
        """
        Args:
            data_dicts: list of dicts with keys "image" and "label" (paths)
            is_training: bool, True for training data (enables augmentation)
            target_shape: tuple, the desired output spatial size.
        """
        self.data_dicts = data_dicts
        self.is_training = is_training
        
        # Initialize the ImageProcessor helper for all transforms
        self.processor = ImageProcessor(target_shape=target_shape)

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        # Get file paths
        image_path = self.data_dicts[idx]["image"]
        label_path = self.data_dicts[idx]["label"]
        
        # Process image and label using the helper class
        image_tensor, label_tensor = self.processor.process_pair(
            mri_path=image_path,
            label_path=label_path,
            is_augmenting=self.is_training
        )
        
        return image_tensor, label_tensor

# Data loader

def get_dataloaders(
    mr_folder,
    label_folder,
    batch_size=4,
    num_workers=4,
    train_spatial_size=TARGET_SHAPE,
    val_spatial_size=TARGET_SHAPE,
    num_classes=6,
    seed=42,
):
    """
    Create train, validation, and test dataloaders using custom helpers.py logic.
    """
    
    # Ensure each image has a corresponding label
    mr_files = sorted(os.listdir(mr_folder))
    label_files = sorted(os.listdir(label_folder))
    
    if len(mr_files) != len(label_files):
         raise ValueError("Mismatched number of images and labels.")
         
    # Create the list of data dictionaries from the sorted file lists
    data_dicts = []
    for mr_file, label_file in zip(mr_files, label_files):
        data_dicts.append({
            "image": os.path.join(mr_folder, mr_file),
            "label": os.path.join(label_folder, label_file)
        })

    # Randomly split data into 80% train, 10% val, 10% test
    np.random.seed(seed)
    indices = np.random.permutation(len(data_dicts))
    train_end = int(0.8 * len(data_dicts))
    val_end = int(0.9 * len(data_dicts))

    train_dicts = [data_dicts[i] for i in indices[:train_end]]
    val_dicts = [data_dicts[i] for i in indices[train_end:val_end]]
    test_dicts = [data_dicts[i] for i in indices[val_end:]]

    print(f"Split: {len(train_dicts)} train, {len(val_dicts)} val, {len(test_dicts)} test")

    train_ds = Prostate3DDataset(
        train_dicts, 
        is_training=True, 
        target_shape=train_spatial_size
    )
    val_ds = Prostate3DDataset(
        val_dicts, 
        is_training=False, 
        target_shape=val_spatial_size
    )
    test_ds = Prostate3DDataset(
        test_dicts, 
        is_training=False, 
        target_shape=val_spatial_size
    )

    # Data loaders for each set
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader