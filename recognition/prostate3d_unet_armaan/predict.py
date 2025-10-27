"""
predict.py - Evaluate trained 3D U-Net on the Prostate 3D test set.
Computes Dice metrics and saves visualization comparisons.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.data import DataLoader

# Local imports
from dataset import get_dataloaders
from modules import UNet3D
import helpers # Import helpers to access get_dice_metric

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
    NUM_WORKERS = 2
    SPATIAL_SIZE = (256, 256, 128)
    
    # Use the same configuration as training
    INCLUDE_BACKGROUND = False


# Visualization function
def visualize_prediction(image, label, pred, save_path, idx, num_classes):
    """
    Save visualization of 3D prediction vs ground truth for one sample.
    Assumes label and pred are one-hot (C, D, H, W) and converts them to index maps.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Image: (B, 1, D, H, W) -> (D, H, W). Take the single channel and remove batch dim.
    image = image[0, 0].cpu().numpy() 
    
    # Label/Pred: (B, C, D, H, W) -> (D, H, W) index map. Take the single channel for batch dim.
    true_label = torch.argmax(label[0], dim=0).cpu().numpy() 
    pred_label = torch.argmax(pred[0], dim=0).cpu().numpy()
    
    # Choose a central slice for visualization
    mid_slice = image.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image[:, :, mid_slice].transpose(), cmap='gray') # Transpose for better display order (H, W)
    axes[0].set_title("Input MRI")
    
    # Use vmin/vmax to ensure consistent color mapping for all classes
    # cmap='tab10' is good for discrete classes up to 10
    axes[1].imshow(true_label[:, :, mid_slice].transpose(), cmap='tab10', vmin=0, vmax=num_classes - 1)
    axes[1].set_title("Ground Truth")
    
    axes[2].imshow(pred_label[:, :, mid_slice].transpose(), cmap='tab10', vmin=0, vmax=num_classes - 1)
    axes[2].set_title("Prediction")
    
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_{idx:03d}.png"), dpi=150)
    plt.close(fig)

# Model evaluation
def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and compute Dice scores using helpers.py.
    """
    # Use the consistent DiceMetric setup from helpers.py
    dice_metric = helpers.get_dice_metric(
        include_background=Config.INCLUDE_BACKGROUND,
        reduction="none" # Keep reduction="none" to get class-wise scores
    )
    
    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = batch['image'].to(device)
            labels = batch['label'].to(device) # Labels are now assumed one-hot (B, C, D, H, W)
            
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1) # Predictions are (B, C, D, H, W) probabilities
            
            # Save visualization for first few samples
            if idx < 5:
                visualize_prediction(images, labels, outputs, Config.VIS_DIR, idx, Config.NUM_CLASSES)
            
            # Compute Dice per sample
            dice_metric(y_pred=outputs, y=labels)
    
    # Aggregate scores
    # If include_background=False (Config default), this returns C-1 scores.
    # We use reduction="none" to get the full array.
    dice_per_class = dice_metric.aggregate(reduction="none").cpu().numpy()
    dice_mean = dice_per_class.mean()
    dice_metric.reset()
    
    # Output class labels start from 1 because background (class 0) is excluded
    class_labels = range(1, Config.NUM_CLASSES)
    
    print("\n--- Dice Scores (Excluding Background) ---")
    for i, score in zip(class_labels, dice_per_class[0]): # dice_per_class is (1, C-1)
        print(f"Class {i}: {score:.4f}")
    print(f"Mean Dice (no background): {dice_mean:.4f}")
    
    return dice_mean, dice_per_class[0]

# Main function

def main():
    print("Loading data and model...")
    
    # Load test set
    _, _, test_loader = get_dataloaders(
        mr_folder=Config.MR_FOLDER,
        label_folder=Config.LABEL_FOLDER,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        val_spatial_size=Config.SPATIAL_SIZE,
        num_classes=Config.NUM_CLASSES,
    )
    
    # Initialize model
    # Note: Using the same dropout_p=0.2 from your train.py configuration
    model = UNet3D(in_channels=1, out_channels=Config.NUM_CLASSES, dropout_p=0.2) 
    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    
    print(f"Loaded checkpoint from {Config.CHECKPOINT_PATH}")
    
    # Evaluate
    mean_dice, dice_per_class = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Save summary
    os.makedirs(Config.VIS_DIR, exist_ok=True)
    with open(os.path.join(Config.VIS_DIR, "dice_scores.txt"), "w") as f:
        f.write(f"Mean Dice (no background): {mean_dice:.4f}\n")
        
        # Save class-wise scores
        class_labels = range(1, Config.NUM_CLASSES)
        for i, d in zip(class_labels, dice_per_class):
            f.write(f"Class {i}: {d:.4f}\n")
    
    print(f"\nResults saved to {Config.VIS_DIR}/dice_scores.txt")

if __name__ == "__main__":
    main()