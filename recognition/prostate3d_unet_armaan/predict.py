"""
predict.py - 3D U-Net Evaluation Methods

A script to evaluate the final trained 3D U-Net model on the Prostate 3D test set.
It computes class-wise Dice metrics (including background) and saves 
visualization comparisons of the ground truth versus the model's prediction 
for initial samples.

Author: Armaan Aulakh
Date: October 30 2025
"""

# Package imports
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from dataset import get_dataloaders
from modules import UNet3D

# Config
class Config:
    """
    Class handling environmental configuration to run predict.py - dynamically sets path names
    and controls backend infrastructure based on training device.
    """

    # Select device based on which backend is available for torch
    if torch.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    # Dynamically select path to images based on backend architecture
    if torch.mps.is_available():
        BASE_DIR = "recognition/Prostate3D_local"
    else:
        BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open"
    
    # Set environmental variables
    MR_FOLDER = os.path.join(BASE_DIR, "semantic_MRs")
    LABEL_FOLDER = os.path.join(BASE_DIR, "semantic_labels_only")
    
    CHECKPOINT_PATH = "checkpoints/best_model.pth"
    VIS_DIR = "logs/predictions"
    
    NUM_CLASSES = 6
    BATCH_SIZE = 1
    NUM_WORKERS = 1
    SPATIAL_SIZE = (192, 192, 96)
    
    EXCLUDE_BACKGROUND = False


# Visualization function
def visualize_prediction(image, label, pred, save_path, idx, num_classes) -> None:
    """
    Generates and saves a three-panel comparison plot (Input, Ground Truth, Prediction).

    The plot visualizes a central axial slice of a 3D volume, ensuring a consistent 
    color map for all segmentation classes (0 to 5).

    Args:
        image (torch.Tensor): The input MRI volume (B, 1, D, H, W).
        label (torch.Tensor): The ground truth sparse index map (B, D, H, W).
        pred (torch.Tensor): The predicted class index map (B, D, H, W).
        save_path (str): Directory where the output PNG file will be saved.
        idx (int): The sample index, used for naming the output file.
        num_classes (int): The total number of classes (6).
    
    Returns:
        None: The function saves the figure to disk and closes the plot.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Image: (B, 1, D, H, W) -> (D, H, W). Take the single channel and remove batch dim.
    image = image[0, 0].cpu().numpy() 
    
    # Label/Pred: (B, D, H, W) index map. Remove batch dim.
    true_label = label[0].cpu().numpy()
    pred_label = pred[0].cpu().numpy()
    
    # Choose a central slice for visualization
    mid_slice = image.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Transpose for common viewing orientation
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
def evaluate_model(model, test_loader, device) -> tuple:
    """
    Evaluates the model and computes class-wise Dice Similarity Coefficients.

    The function iterates through the test set, computes Dice scores for 
    all classes (including background), and saves visualization images 
    for the first five samples.

    Args:
        model (nn.Module): The loaded 3D U-Net model.
        test_loader (DataLoader): The PyTorch DataLoader for the test set.
        device (str): The execution device ('cuda', 'mps', or 'cpu').

    Returns:
        tuple: (mean_dice_all, mean_dice_per_class)
            - mean_dice_all (float): The overall mean DSC across all classes and samples.
            - mean_dice_per_class (np.ndarray): The mean DSC for each class (C,).
    """
    model.eval()
    all_dice_scores = [] # Store all C class-wise dice scores for all samples
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images) # Outputs are logits (B, C, D, H, W)
            
            # Compute dice score for all classes (include background)
            # NOTE: This section calculates the full (B, C) dice_per_class tensor, 
            # including background (class 0), by re-implementing logic 
            # from dice_coefficient. Training occurred with ommission of class 0 (BG)
            # based on common practice researched for medical segmentation tasks.
            
            pred = torch.nn.functional.softmax(outputs, dim=1)
            # Convert target to one-hot encoding (B, D, H, W) -> (B, C, D, H, W)
            target_oh = torch.nn.functional.one_hot(labels, num_classes=Config.NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            
            dice_scores_list = []
            smooth = 1e-5 # Ensure no potential division by 0
            
            for c in range(Config.NUM_CLASSES): # Iterate over ALL classes 0 to 5
                pred_c = pred[:, c, ...].reshape(pred.shape[0], -1)
                target_c = target_oh[:, c, ...].reshape(labels.shape[0], -1)
                
                intersection = (pred_c * target_c).sum(dim=1)
                union = pred_c.sum(dim=1) + target_c.sum(dim=1)
                
                dice_c = (2. * intersection + smooth) / (union + smooth)
                dice_scores_list.append(dice_c)
            
            # dice_per_class is (B, C) -> includes all classes 0 to 5
            dice_per_class = torch.stack(dice_scores_list, dim=1) 
            
            # Store the scores for all classes
            all_dice_scores.append(dice_per_class.cpu().numpy())
            
            # Prepare prediction for visualization
            pred_index_map = torch.argmax(outputs, dim=1) 
            
            # Save visualization for first few samples
            if idx < 5:
                # Visualization function remains the same and already uses all classes
                visualize_prediction(images, labels, pred_index_map, Config.VIS_DIR, idx, Config.NUM_CLASSES)
    
    # Aggregate scores
    # all_dice_scores is a list of (1, 6) arrays (one per sample)
    all_dice_scores = np.concatenate(all_dice_scores, axis=0) # Shape: (Num_Samples, C)
    
    # Calculate the mean score across all samples for each class
    mean_dice_per_class = all_dice_scores.mean(axis=0) # Shape: (6,)
    mean_dice_all = mean_dice_per_class.mean()
    
    # Output class labels start from 0 because background (class 0) is included
    class_labels = range(Config.NUM_CLASSES)
    
    print("\nDice Scores (Including Background)")
    for i, score in zip(class_labels, mean_dice_per_class): 
        print(f"Class {i}: {score:.4f}")
    print(f"Overall Mean Dice: {mean_dice_all:.4f}")
    
    # Returning all scores now
    return mean_dice_all, mean_dice_per_class

# Main function

def main():
    print("Loading data and model")
    
    # Load test set (from dataset.py)
    _, _, test_loader = get_dataloaders(
        mr_folder=Config.MR_FOLDER,
        label_folder=Config.LABEL_FOLDER,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        val_spatial_size=Config.SPATIAL_SIZE,
        num_classes=Config.NUM_CLASSES,
    )
    
    # Initialize model
    model = UNet3D(in_channels=1, out_channels=Config.NUM_CLASSES, dropout_p=0.2) 
    
    # Load checkpoint
    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    
    print(f"Loaded checkpoint from {Config.CHECKPOINT_PATH}")
    
    # Evaluate - uses the new return values
    mean_dice_all, dice_per_class_all = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Save summary
    os.makedirs(Config.VIS_DIR, exist_ok=True)
    with open(os.path.join(Config.VIS_DIR, "dice_scores_all_classes.txt"), "w") as f:
        f.write(f"Overall Mean Dice (ALL classes): {mean_dice_all:.4f}\n")
        
        # Save class-wise scores
        class_labels = range(Config.NUM_CLASSES) # Includes class 0
        for i, d in zip(class_labels, dice_per_class_all):
            f.write(f"Class {i}: {d:.4f}\n")
    
    print(f"\nResults saved to {Config.VIS_DIR}/dice_scores_all_classes.txt")

# Main router for file
if __name__ == "__main__":
    main()