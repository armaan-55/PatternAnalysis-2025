import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# A small constant added to the numerator and denominator to prevent division by zero
SMOOTH = 1e-5

def dice_coefficient(pred, target, num_classes, smooth=SMOOTH):
    """
    Calculates the generalized Dice Similarity Coefficient (DSC) for multi-class segmentation.
    
    Args:
        pred (torch.Tensor): Predicted logits (raw scores) from the model (N, C, D, H, W).
        target (torch.Tensor): Ground truth labels (N, D, H, W). Expects sparse labels (0 to C-1).
        num_classes (int): Number of segmentation classes (C).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: The mean Dice Coefficient across all classes (excluding background if class_weights are used).
    """
    
    # 1. Convert logits to probabilities using Softmax
    # Shape changes from (N, C, D, H, W) to (N, C, D, H, W) where C is probability
    pred = F.softmax(pred, dim=1) 
    
    # 2. Convert sparse target labels to one-hot encoding for comparison
    # target shape (N, D, H, W) -> one_hot_target shape (N, C, D, H, W)
    one_hot_target = F.one_hot(target.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    
    # Flatten spatial dimensions (D*H*W) for easier calculation (N, C, V) where V = D*H*W
    pred = pred.view(-1, num_classes, pred.size(2) * pred.size(3) * pred.size(4))
    one_hot_target = one_hot_target.view(-1, num_classes, one_hot_target.size(2) * one_hot_target.size(3) * one_hot_target.size(4))

    # Calculate Intersection and Union for all classes simultaneously
    # Intersection: sum(p_c * t_c) over all spatial elements for each class c
    intersection = torch.sum(pred * one_hot_target, dim=2)  # Shape (N, C)
    
    # Union (Denominator): sum(p_c + t_c) over all spatial elements for each class c
    union = torch.sum(pred + one_hot_target, dim=2) # Shape (N, C)

    # Calculate Dice Score per class (N, C)
    # The original formula is 2 * I / U
    dice_per_class = (2. * intersection + smooth) / (union + smooth) 
    
    # Typically, we ignore the background (class 0) for the final metric reporting.
    # We take the mean across classes 1 to C-1.
    # Note: If you want to include the background, use dice_per_class.mean()
    
    # Average the dice scores across the batch and the non-background classes
    mean_dice = dice_per_class[:, 1:].mean()

    return mean_dice

class DiceLoss(nn.Module):
    """
    The generalized multi-class Dice Loss (1 - Dice Coefficient).
    Designed to be used with the CrossEntropy loss for robust training (e.g., BCE/CE + Dice).
    """
    def __init__(self, num_classes, smooth=SMOOTH, weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        # Optional: class weights can be used to mitigate class imbalance
        self.weights = weights if weights is not None else torch.ones(num_classes)

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted logits (raw scores) from the model (N, C, D, H, W).
            target (torch.Tensor): Ground truth labels (N, D, H, W). Expects sparse labels (0 to C-1).
        
        Returns:
            torch.Tensor: The weighted Dice Loss value.
        """
        
        # 1. Softmax to get probabilities
        # Shape: (N, C, D, H, W)
        pred = F.softmax(pred, dim=1)
        
        # 2. One-hot encode the target
        # Target (N, D, H, W) -> One-Hot (N, C, D, H, W)
        one_hot_target = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Ensure weights are on the correct device
        if self.weights.device != pred.device:
            self.weights = self.weights.to(pred.device)

        # Flatten spatial dimensions
        pred_flat = pred.view(-1, self.num_classes)        # (N*V, C)
        target_flat = one_hot_target.view(-1, self.num_classes) # (N*V, C)
        
        # Apply class weights:
        # We compute the weighted sum over all spatial dimensions (V = D*H*W)
        
        # Intersection: (p_c * t_c) over V dimensions -> (N, C)
        intersection = torch.sum(pred * one_hot_target, dim=[2, 3, 4]) 

        # Sum of elements: (p_c + t_c) over V dimensions -> (N, C)
        union = torch.sum(pred + one_hot_target, dim=[2, 3, 4]) 

        # Dice Score per class (N, C)
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Weighted mean Dice across all N batches
        # We weight the dice score for each class and then average across the batch.
        # This implementation sums the weighted dice scores across all N*C entries and normalizes by N*C.
        weighted_dice = (dice_per_class * self.weights.view(1, -1)).sum(dim=1) / self.weights.sum() 
        
        # Total Dice Loss is 1 - the weighted Dice score, averaged over the batch
        dice_loss = 1. - weighted_dice.mean()

        return dice_loss