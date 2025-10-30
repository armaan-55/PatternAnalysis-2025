"""
dataset.py - Data loader classes and functions.

A script to setup the relevant dataloaders to load images in the proper format for the 
3D segmentation task.

Author: Armaan Aulakh
Date: October 30 2025
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from helpers import ImageProcessor 


class Prostate3DDataset(Dataset):
    """
    Custom PyTorch dataset that loads NIfTI volumes on-the-fly and uses 
    helpers.ImageProcessor for all transforms and augmentation.

    Args:
        data_dicts (list): A list of dictionaries where each dict contains an image and label path
                           for a single volume.
        
        is_training(bool): Flag indicating whether the dataset is used for training or not.
                           If true, augmentation is applied as transforms should only occur on the
                           training set.

        target_shape(tuple): The spatial size (D, H, W) that images should be cropped to.

        num_classes(int): Number of segmentation classes (always 6 for our case)
    """
    def __init__(self, data_dicts, is_training, target_shape, num_classes=6):
        self.data_dicts = data_dicts
        self.is_training = is_training
        
        # Initialize the ImageProcessor helper from helpers.py for all transforms
        self.processor = ImageProcessor(target_shape=target_shape, num_classes=num_classes)

    def __len__(self):
        # Return length of data_dicts
        return len(self.data_dicts)

    def __getitem__(self, idx):
        # Returns tensors containing image and label data for the file path pairs.
        image_path = self.data_dicts[idx]["image"]
        label_path = self.data_dicts[idx]["label"]
        
        image_tensor, label_tensor = self.processor.process_pair(
            mri_path=image_path,
            label_path=label_path,
            is_augmenting=self.is_training
        )
        
        return image_tensor, label_tensor

def get_dataloaders(
    mr_folder,
    label_folder,
    batch_size=4,
    num_workers=4,
    train_spatial_size=(192, 192, 96),
    val_spatial_size=(192, 192, 96),
    num_classes=6,
    seed=42,
):
    """
    Creates PyTorch DataLoaders for training, validation, and testing.

    The function performs a subjectID level split for an 80/10/10 train/validation/test split.
    This method was researched to be an improvement over random split to prevent longitudinal data leakage
    in a case such as HipMRI (https://www.researchgate.net/publication/374549533_How_You_Split_Matters_Data_Leakage_and_Subject_Characteristics_Studies_in_Longitudinal_Brain_MRI_Analysis)

    Args:
        mr_folder(str): Path to the directory containing the MRI volumes.
        label_folder(str): Path to the directory containing the label volumes.
        batch_size(int): Batch size for the training data loader.
        num_workers(int): Number of cores alotted to data loading for performance.
        train_spatial_size(tuple): Target spatial size for training data.
        val_spatial_size(tuple): Target spatial size for validation data.
        num_classes(int): Number of classes for problem (6).
        seed(int): Seed for reproducibility of results.
    
    Returns:
        tuple: Tuple containing three DataLoaders (train_loader, val_loader, test_loader)
    
    Raises:
        ValueError: If the number of images doesn't match the number of labels.
    """
    
    # Gather all files and match pairs
    mr_files = sorted([f for f in os.listdir(mr_folder) 
                       if f.endswith('.nii') or f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(label_folder) 
                          if f.endswith('.nii') or f.endswith('.nii.gz')])
    
    if len(mr_files) != len(label_files):
        raise ValueError(f"Mismatched count: {len(mr_files)} images vs {len(label_files)} labels")
    
    
    # Build map of Subject ID -> List of their (Image, Label) file pairs
    subject_to_files = {}
    
    for mr_file, label_file in zip(mr_files, label_files):
        
        # Extract Subject ID using string indexing methods.
        try:
            # Find the position of the week identifier
            week_pos = mr_file.index('_Week')
            subject_id = mr_file[:week_pos]
        except ValueError:
            # Fallback for files without a '_Week' identifier, assumes subject ID is the first part
            subject_id = mr_file.split('_')[0] 

        if subject_id not in subject_to_files:
            subject_to_files[subject_id] = []
            
        subject_to_files[subject_id].append({
            "image": os.path.join(mr_folder, mr_file),
            "label": os.path.join(label_folder, label_file)
        })

    unique_subjects = list(subject_to_files.keys())
    print(f"Found {len(unique_subjects)} unique subjects.")

    
    # Split data by Subject ID for better model generalisation.
    np.random.seed(seed)
    indices = np.random.permutation(len(unique_subjects))
    train_end = int(0.8 * len(unique_subjects)) # Split subjects: 80% train
    val_end = int(0.9 * len(unique_subjects))   # 10% val, 10% test

    train_subjects = [unique_subjects[i] for i in indices[:train_end]]
    val_subjects = [unique_subjects[i] for i in indices[train_end:val_end]]
    test_subjects = [unique_subjects[i] for i in indices[val_end:]]
    
    # Reconstruct file lists (data_dicts) ensuring no leakage
    train_dicts = [item for subject in train_subjects for item in subject_to_files[subject]]
    val_dicts = [item for subject in val_subjects for item in subject_to_files[subject]]
    test_dicts = [item for subject in test_subjects for item in subject_to_files[subject]]

    print(f"Split (Subjects): {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test")
    print(f"Split (Files): {len(train_dicts)} train, {len(val_dicts)} val, {len(test_dicts)} test files.")
    
    # Create datasets
    train_ds = Prostate3DDataset(
        train_dicts, 
        is_training=True, 
        target_shape=train_spatial_size,
        num_classes=num_classes
    )
    val_ds = Prostate3DDataset(
        val_dicts, 
        is_training=False, 
        target_shape=val_spatial_size,
        num_classes=num_classes
    )
    test_ds = Prostate3DDataset(
        test_dicts, 
        is_training=False, 
        target_shape=val_spatial_size,
        num_classes=num_classes
    )

    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader