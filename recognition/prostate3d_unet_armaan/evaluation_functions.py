"""
evaluation_functions.py - Contains calculations for per class dice losses and dice coefficient
to use as a metric in train.py.

Author: Armaan Aulakh
Date: October 30 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# A small constant added to the numerator and denominator to prevent division by zero
SMOOTH = 1e-5

class DiceLoss(nn.Module):
    """
    A pure PyTorch implementation of the Dice Loss for multi-class segmentation.
    
    Excludes the background (class 0) from the loss calculation for training to prevent inflation of model performance.
    Loss function is defined as L = 1 - DSC, where DSC is Dice Similarity Coeff.

    Attributes:
        num_classes(int): Total number of classes (6).
        smooth(float): A small constant to prevent division by zero.
    """
    def __init__(self, num_classes, smooth=1e-5):
        """
        Initializes the DiceLoss instance.

        Args:
            num_classes(int): Total number of classes (6).
            smooth(float, optional): Value for numerical stability, with default defined as 
                                     environmental variable.
        """
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Calculates the Dice Loss (1 - Mean DSC) over non-background classes.

        The DSC is calculated for each class and sample, and the mean is taken 
        over all foreground classes (1 to 5) and all batch samples.

        Args:
            pred (torch.Tensor): Model logits (B, C, D, H, W). Softmax is applied internally.
            target (torch.Tensor): Ground truth sparse index map (B, D, H, W) of type long.

        Returns:
            torch.Tensor: A scalar tensor representing the mean Dice Loss for the batch.
        """
        pred = F.softmax(pred, dim=1)

        # Convert target to one-hot encoding
        target_oh = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        dice_scores = []
        for c in range(self.num_classes):
            pred_c = pred[:, c, ...].reshape(pred.shape[0], -1)  # (B, D*H*W)
            target_c = target_oh[:, c, ...].reshape(target.shape[0], -1)  # (B, D*H*W)
            
            intersection = (pred_c * target_c).sum(dim=1)  # (B,)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)  # (B,)
            
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)  # (B,)
            dice_scores.append(dice_c)
        
        # Stack to (B, C)
        dice_per_class = torch.stack(dice_scores, dim=1)
        
        # Exclude background (class 0)
        dice_per_class_non_bg = dice_per_class[:, 1:]
        
        # Mean Dice Loss across all non-background classes and all batch samples
        mean_dice = dice_per_class_non_bg.mean()
        loss = 1.0 - mean_dice
        
        return loss

def dice_coefficient(pred, target, num_classes, smooth=1e-5):
    """
    Calculates the per-class Dice Similarity Coefficient (DSC) as a metric.

    This function computes the DSC for all classes (0 to 5), but only returns 
    the mean DSC and the per-class DSC array for non-BG classes (1 to 5).

    Args:
        pred (torch.Tensor): Model logits (B, C, D, H, W). Softmax is applied internally.
        target (torch.Tensor): Ground truth sparse index map (B, D, H, W) of type long.
        num_classes (int): The total number of classes (C).
        smooth (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.

    Returns:
        tuple: (mean_dice_non_bg, dice_per_class_non_bg)
            - mean_dice_non_bg (torch.Tensor): The mean DSC across all non-background classes and samples.
            - dice_per_class_non_bg (torch.Tensor): DSC for each sample and each non-background class (B, C-1).
    """
    pred = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    
    # Calculate per-class Dice for each batch sample
    dice_scores = []
    for c in range(num_classes):
        pred_c = pred[:, c, ...].reshape(pred.shape[0], -1)
        target_c = target_oh[:, c, ...].reshape(target.shape[0], -1)
        
        intersection = (pred_c * target_c).sum(dim=1)
        union = pred_c.sum(dim=1) + target_c.sum(dim=1)
        
        dice_c = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_c)
    
    # Stack to (B, C)
    dice_per_class = torch.stack(dice_scores, dim=1)
    
    # Exclude background (class 0)
    dice_per_class_non_bg = dice_per_class[:, 1:]
    mean_dice_non_bg = dice_per_class_non_bg.mean()
    
    return mean_dice_non_bg, dice_per_class_non_bg