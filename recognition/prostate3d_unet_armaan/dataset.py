import os
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

# Ensure helpers.py is imported
sys.path.insert(0, os.path.dirname(__file__))
import helpers

# 3D Data loading functions
def load_data_3D(image_paths, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, orient=False, early_stop=False, num_classes=None,
                 target_shape=None):
    """
    Load 3D medical image data from a list of NIfTI file paths.
    If target_shape is provided, all volumes will be resized to that shape.
    """
    affines = []
    interp = 'linear' if dtype != np.uint8 else 'nearest'
    num = len(image_paths)

    # Load first image to determine shape if target_shape not given
    niftiImage = nib.load(image_paths[0])
    if orient:
        niftiImage = helpers.applyOrientation(niftiImage, interpolation=interp)
    first_case = niftiImage.get_fdata()
    if first_case.ndim == 4:
        first_case = first_case[:, :, :, 0]

    if target_shape is None:
        target_shape = first_case.shape

    if categorical:
        shape = (num, num_classes, *target_shape)
    else:
        shape = (num, *target_shape)

    data = np.zeros(shape, dtype=dtype)

    for i, path in enumerate(tqdm(image_paths)):
        img = nib.load(path)
        if orient:
            img = helpers.applyOrientation(img, interpolation=interp)
        arr = img.get_fdata()
        if arr.ndim == 4:
            arr = arr[:, :, :, 0]

        # Resize to target_shape
        if arr.shape != target_shape:
            zoom_factors = [t / s for t, s in zip(target_shape, arr.shape)]
            order = 1 if dtype != np.uint8 else 0
            arr = zoom(arr, zoom_factors, order=order)

        arr = arr.astype(dtype)
        if normImage:
            arr = (arr - arr.mean()) / arr.std()

        if categorical:
            arr = helpers.to_channels(arr, num_classes=num_classes, dtype=dtype)
            arr = np.moveaxis(arr, -1, 0)

        data[i] = arr
        affines.append(img.affine)

        if early_stop and i > 20:
            break

    return (data, affines) if getAffines else data

def load_dataset_3D(image_paths, label_paths, normImage=True, dtype=np.float32,
                    early_stop=False, num_classes=6, target_shape=None):
    images = load_data_3D(image_paths, normImage=normImage, dtype=dtype, 
                          early_stop=early_stop, target_shape=target_shape)
    labels = load_data_3D(label_paths, normImage=False, dtype=np.uint8,
                          early_stop=early_stop, categorical=True, 
                          num_classes=num_classes, target_shape=target_shape)
    return images, labels

# PyTorch Dataset Wrapper
class Prostate3DDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None, num_classes=6, 
                 early_stop=False, target_shape=None):
        self.images, self.labels = load_dataset_3D(image_paths, label_paths,
                                                   normImage=True,
                                                   early_stop=early_stop,
                                                   num_classes=num_classes,
                                                   target_shape=target_shape)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# Pairing logic for MR and labels
def get_paired_paths(mr_folder, label_folder):
    """
    Pair MR images and labels by patient ID (first part of filename).
    Returns lists of paths.
    """
    mr_files = [f for f in os.listdir(mr_folder) if f.endswith(".nii.gz")]
    label_files = [f for f in os.listdir(label_folder) if f.endswith(".nii.gz")]

    paired_mr = []
    paired_label = []

    for label_file in label_files:
        patient_id = label_file.split('_')[0]
        mr_match = [f for f in mr_files if f.startswith(patient_id)]
        if mr_match:
            paired_mr.append(os.path.join(mr_folder, mr_match[0]))
            paired_label.append(os.path.join(label_folder, label_file))

    if len(paired_mr) == 0:
        raise ValueError("No matching files found between MR and label folders!")

    print(f"Found {len(paired_mr)} matching pairs")
    return paired_mr, paired_label

# Quick test
if __name__ == "__main__":
    base_dir = "../Prostate3D_local"
    mr_folder = os.path.join(base_dir, "semantic_MRs")
    label_folder = os.path.join(base_dir, "semantic_labels_only")

    image_paths, label_paths = get_paired_paths(mr_folder, label_folder)
    
    target_shape = None
    
    dataset = Prostate3DDataset(image_paths, label_paths, early_stop=False, 
                                target_shape=target_shape)
    
    print(f"\nNumber of samples: {len(dataset)}")
    x, y = dataset[0]
    print("Input shape:", x.shape)
    print("Label shape:", y.shape)