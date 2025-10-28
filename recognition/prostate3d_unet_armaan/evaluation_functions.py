import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# A small constant added to the numerator and denominator to prevent division by zero
SMOOTH = 1e-5

class DiceLoss(nn.Module):
    """
    A pure PyTorch implementation of the Dice Loss for multi-class segmentation.
    Excludes the background (class 0) from the loss calculation.
    """
    def __init__(self, num_classes, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, D, H, W) logits -> apply softmax
        # target: (B, D, H, W) sparse indices (long)

        pred = F.softmax(pred, dim=1)

        # Convert target to one-hot encoding (B, D, H, W) -> (B, D, H, W, C) -> (B, C, D, H, W)
        target_oh = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        # Reshape for easy calculation across all classes/batches (B*C, D*H*W)
        pred_flat = pred.reshape(-1, pred.shape[-3] * pred.shape[-2] * pred.shape[-1])
        target_flat = target_oh.reshape(-1, target_oh.shape[-3] * target_oh.shape[-2] * target_oh.shape[-1])

        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        # Calculate Dice score (DSC) per class/sample pair
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Reshape back to (Batch, Class)
        dice_per_class_flat = dice.reshape(target.shape[0], self.num_classes)

        # Exclude background (class 0)
        dice_per_class = dice_per_class_flat[:, 1:]

        # Mean Dice Loss across all non-background classes and all batch samples
        mean_dice = dice_per_class.mean()
        loss = 1.0 - mean_dice

        return loss

def dice_coefficient(pred, target, num_classes, smooth=1e-5):
    """
    Calculates the mean Dice coefficient and per-class Dice scores
    (excluding background, class 0).
    """
    pred = F.softmax(pred, dim=1)

    # Convert target to one-hot encoding
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    # Reshape for easy calculation (B*C, D*H*W)
    pred_flat = pred.reshape(-1, pred.shape[-3] * pred.shape[-2] * pred.shape[-1])
    target_flat = target_oh.reshape(-1, target_oh.shape[-3] * target_oh.shape[-2] * target_oh.shape[-1])

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    # Calculate Dice score (DSC) per class/sample pair
    dice = (2. * intersection + smooth) / (union + smooth)

    # Reshape back to (Batch, Class)
    dice_per_class_flat = dice.reshape(target.shape[0], num_classes)

    # Exclude background (class 0) for metric calculation
    dice_per_class_non_bg = dice_per_class_flat[:, 1:]

    # Calculate the mean Dice across all non-background classes and all batch samples
    mean_dice_non_bg = dice_per_class_non_bg.mean()

    return mean_dice_non_bg, dice_per_class_non_bg