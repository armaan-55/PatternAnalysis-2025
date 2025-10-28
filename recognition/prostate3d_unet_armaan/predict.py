"""
predict.py - Evaluate trained 3D U-Net on the Prostate 3D test set.
Computes Dice metrics and saves visualization comparisons using pure PyTorch functions.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from dataset import get_dataloaders
from modules import UNet3D
from evaluation_functions import dice_coefficient # Import the custom Dice Coefficient metric

# Config

class Config:
    BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open"
    MR_FOLDER = os.path.join(BASE_DIR, "semantic_MRs")
    LABEL_FOLDER = os.path.join(BASE_DIR, "semantic_labels_only")
    
    CHECKPOINT_PATH = "checkpoints/best_model.pth"
    VIS_DIR = "logs/predictions"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 6
    BATCH_SIZE = 1
    NUM_WORKERS = 1
    SPATIAL_SIZE = (256, 256, 128)
    
    EXCLUDE_BACKGROUND = True


# Visualization function
def visualize_prediction(image, label, pred, save_path, idx, num_classes):
    """
    Save visualization of 3D prediction vs ground truth for one sample.
    Assumes image is (B, 1, D, H, W) and label/pred are (B, D, H, W) sparse index maps.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Image: (B, 1, D, H, W) -> (D, H, W). Take the single channel and remove batch dim.
    image = image[0, 0].cpu().numpy() 
    
    # Label/Pred: (B, D, H, W) index map. Remove batch dim.
    true_label = label[0].cpu().numpy()
    pred_label = pred[0].cpu().numpy()
    
    # Choose a central slice for visualization
    mid_slice = image.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Transpose for common viewing orientation (e.g., Axial slice)
    slice_func = lambda arr: arr[:, :, mid_slice].transpose() 
    
    axes[0].imshow(slice_func(image), cmap='gray') 
    axes[0].set_title("Input MRI")
    
    # Use vmin/vmax to ensure consistent color mapping for all classes
    axes[1].imshow(slice_func(true_label), cmap='tab10', vmin=0, vmax=num_classes - 1)
    axes[1].set_title("Ground Truth")
    
    axes[2].imshow(slice_func(pred_label), cmap='tab10', vmin=0, vmax=num_classes - 1)
    axes[2].set_title("Prediction")
    
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_{idx:03d}.png"), dpi=150)
    plt.close(fig)

# Model evaluation
def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and compute Dice scores using evaluation_functions.py.
    """
    
    model.eval()
    all_dice_scores = [] # Store class-wise dice scores for all samples
    
    with torch.no_grad():
        # Assumes DataLoader returns a tuple: (image_tensor, label_tensor)
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device) # Labels are sparse index maps (B, D, H, W)
            
            outputs = model(images) # Outputs are logits (B, C, D, H, W)
            
            # 1. Compute Dice Score
            # dice_coefficient returns (mean_dice_non_background, dice_per_class_non_background)
            _, dice_per_class = dice_coefficient(
                pred=outputs, 
                target=labels, 
                num_classes=Config.NUM_CLASSES
            )
            all_dice_scores.append(dice_per_class.cpu().numpy())
            
            # 2. Prepare prediction for visualization
            # Convert logits to class index map (B, D, H, W)
            pred_index_map = torch.argmax(outputs, dim=1) 
            
            # Save visualization for first few samples
            if idx < 5:
                # The label tensor here is the sparse index map from the dataloader
                visualize_prediction(images, labels, pred_index_map, Config.VIS_DIR, idx, Config.NUM_CLASSES)
    
    # Aggregate scores
    # all_dice_scores is a list of (1, C-1) arrays (one per sample)
    all_dice_scores = np.concatenate(all_dice_scores, axis=0)
    
    # Calculate the mean score across all samples for each class
    mean_dice_per_class = all_dice_scores.mean(axis=0) 
    mean_dice = mean_dice_per_class.mean()
    
    # Output class labels start from 1 because background (class 0) is excluded
    class_labels = range(1, Config.NUM_CLASSES)
    
    print("\n--- Dice Scores (Excluding Background) ---")
    for i, score in zip(class_labels, mean_dice_per_class): 
        print(f"Class {i}: {score:.4f}")
    print(f"Overall Mean Dice (no background): {mean_dice:.4f}")
    
    return mean_dice, mean_dice_per_class

# Main function

def main():
    print("Loading data and model...")
    
    # Load test set (from dataset.py)
    _, _, test_loader = get_dataloaders(
        mr_folder=Config.MR_FOLDER,
        label_folder=Config.LABEL_FOLDER,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        val_spatial_size=Config.SPATIAL_SIZE,
        num_classes=Config.NUM_CLASSES,
    )
    
    # Initialize model (from modules.py)
    # Note: Using the same dropout_p=0.2 from your train.py configuration
    model = UNet3D(in_channels=1, out_channels=Config.NUM_CLASSES, dropout_p=0.2) 
    
    # Load checkpoint
    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    
    print(f"Loaded checkpoint from {Config.CHECKPOINT_PATH}")
    
    # Evaluate
    mean_dice, dice_per_class = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Save summary
    os.makedirs(Config.VIS_DIR, exist_ok=True)
    with open(os.path.join(Config.VIS_DIR, "dice_scores.txt"), "w") as f:
        f.write(f"Overall Mean Dice (no background): {mean_dice:.4f}\n")
        
        # Save class-wise scores
        class_labels = range(1, Config.NUM_CLASSES)
        for i, d in zip(class_labels, dice_per_class):
            f.write(f"Class {i}: {d:.4f}\n")
    
    print(f"\nResults saved to {Config.VIS_DIR}/dice_scores.txt")

if __name__ == "__main__":
    main()