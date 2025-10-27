import numpy as np
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    Resized,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    ToTensord,
    AsDiscreted
)
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
import os

def to_channels(label_volume, num_classes=None, dtype=np.uint8):
    if num_classes is None:
        num_classes = int(label_volume.max() + 1)
    shape = label_volume.shape + (num_classes,)
    out = np.zeros(shape, dtype=dtype)
    for c in range(num_classes):
        out[..., c] = (label_volume == c).astype(dtype)
    return out

def get_paired_paths(mr_folder, label_folder):
    """
    Pair MR images and labels by patient ID.
    Returns list of dictionaries for MONAI compatibility.
    """
    mr_files = sorted([f for f in os.listdir(mr_folder) if f.endswith(".nii.gz")])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith(".nii.gz")])

    data_dicts = []

    for label_file in label_files:
        patient_id = label_file.split('_')[0]
        mr_match = [f for f in mr_files if f.startswith(patient_id)]
        if mr_match:
            data_dicts.append({
                'image': os.path.join(mr_folder, mr_match[0]),
                'label': os.path.join(label_folder, label_file)
            })

    if len(data_dicts) == 0:
        raise ValueError("No matching files found between MR and label folders!")

    print(f"Found {len(data_dicts)} matching pairs")
    return data_dicts

# Handle MONAI transforms
def get_train_transforms_monai(spatial_size=(96, 96, 48), num_classes=6):
    """
    Training transforms using MONAI.
    Patch-based training with augmentation.
    """
    return Compose([
        # Spatial augmentation
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
        
        # Intensity augmentation (only on image)
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
        
        # Normalize
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        
        # Smart cropping - KEY TRANSFORM!
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            num_classes=num_classes,
            num_samples=1,
        ),
        
        AsDiscreted(keys="label", to_onehot=num_classes),
        # Convert to tensors with channel dimension
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms_monai(spatial_size=(256, 256, 128), num_classes=6):
    """
    Validation/test transforms - no augmentation.
    """
    return Compose([
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(
            keys=["image", "label"], 
            spatial_size=spatial_size, 
            mode=["trilinear", "nearest"]
        ),
        AsDiscreted(keys="label", to_onehot=num_classes),
        ToTensord(keys=["image", "label"]),
    ])

def get_dice_loss(include_background=False, softmax=True):
    """
    Get MONAI's Dice Loss.
    
    Args:
        include_background: If False, ignore background class (class 0)
        to_onehot_y: If True, convert target to one-hot (use False if already one-hot)
        softmax: If True, apply softmax to predictions (use False if model outputs softmax)
    
    Returns:
        DiceLoss instance
    """
    return DiceLoss(
        include_background=include_background,
        softmax=softmax,
        squared_pred=False,  # Use standard Dice formula
        reduction="mean",
    )


def get_dice_ce_loss(include_background=False, softmax=True, lambda_dice=1.0, lambda_ce=1.0):
    """
    Get combined Dice + Cross Entropy Loss (often works better!).
    
    This is what many top medical segmentation models use.
    Dice helps with class imbalance, CE helps with harder examples.
    
    Args:
        lambda_dice: Weight for Dice loss
        lambda_ce: Weight for Cross Entropy loss
    """
    return DiceCELoss(
        include_background=include_background,
        softmax=softmax,
        lambda_dice=lambda_dice,
        lambda_ce=lambda_ce,
    )


def get_dice_metric(include_background=False, reduction="mean", get_not_nans=False):
    return DiceMetric(
        include_background=include_background,
        reduction=reduction,
        get_not_nans=get_not_nans,
    )