"""
predict.py - 3D U-Net Evaluation Methods

A script to evaluate the final trained 3D U-Net model on the Prostate 3D test set.
It computes class-wise Dice metrics (including background) and saves 
visualization comparisons of the ground truth versus the model's prediction 
for initial samples, now including Multi-Planar Reconstruction (MPR).

Author: Armaan Aulakh
Date: October 30 2025 (Updated for MPR)
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


# Helper function for Slicing
def get_slice(arr, plane, D, H, W):
    """
    Extracts a central slice from the 3D numpy array (D, H, W) based on the plane.
    Applies necessary transpositions for standard viewing orientation.
    """
    
    # NOTE: Input array 'arr' is now reliably shape (D, H, W)
    
    if plane == 'Axial':
        # Slice across Depth (D). Resulting slice is (H, W). Transpose to (W, H)
        mid_idx = D // 2
        return arr[mid_idx, :, :].transpose() # (H, W) -> (W, H)
    
    elif plane == 'Coronal':
        # Slice across Height (H). Resulting slice is (D, W). Transpose to (W, D)
        mid_idx = H // 2
        return arr[:, mid_idx, :].transpose(1, 0) # (D, W) -> (W, D)
        
    elif plane == 'Sagittal':
        # Slice across Width (W). Resulting slice is (D, H). Transpose to (H, D)
        mid_idx = W // 2
        return arr[:, :, mid_idx].transpose() # (D, H) -> (H, D)
        
    else:
        raise ValueError("Invalid plane specified.")

def visualize_mpr_comparison(image, label, pred, save_path, idx, num_classes, spatial_size) -> None:
    """
    Generates and saves a nine-panel comparison plot showing Input, Ground Truth, 
    and Prediction across the central Axial, Coronal, and Sagittal planes (MPR).
    
    This version includes an explicit transposition to fix dimension order based on the WARNING.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Unpack Dimensions from Config (192, 192, 96) -> (W, H, D)
    W_config, H_config, D_config = spatial_size 

    raw_image = image[0, 0].cpu().numpy() 
    raw_true_label = label[0].cpu().numpy()
    raw_pred_label = pred[0].cpu().numpy()
    
    image_np = np.transpose(raw_image, (2, 1, 0)) # (W, H, D) -> (D, H, W)
    true_label_np = np.transpose(raw_true_label, (2, 1, 0))
    pred_label_np = np.transpose(raw_pred_label, (2, 1, 0))

    D, H, W = image_np.shape
    
    if (D, H, W) != (D_config, H_config, W_config) and (D, H, W) != (D_config, W_config, H_config):
        print(f"WARNING: Tensor shape {raw_image.shape} does not match expected (D, H, W) from Config {D_config, H_config, W_config}. Slice views may be incorrect.")
        
    planes = ['Axial', 'Coronal', 'Sagittal']
    titles = ['Input MRI', 'Ground Truth', 'Prediction']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for col_idx, plane in enumerate(planes):
        
        # Get Slices for the current plane using the helper function
        img_slice = get_slice(image_np, plane, D, H, W)
        gt_slice = get_slice(true_label_np, plane, D, H, W)
        pred_slice = get_slice(pred_label_np, plane, D, H, W)
        
        # 5. Plot slices (Rows)
        axes[0, col_idx].imshow(img_slice, cmap='gray') 
        axes[1, col_idx].imshow(gt_slice, cmap='tab10', vmin=0, vmax=num_classes - 1)
        axes[2, col_idx].imshow(pred_slice, cmap='tab10', vmin=0, vmax=num_classes - 1)

        # 6. Set Titles
        axes[0, col_idx].set_title(f"Input ({plane})", fontsize=14)
        axes[1, col_idx].set_title(f"GT ({plane})", fontsize=14)
        axes[2, col_idx].set_title(f"Pred ({plane})", fontsize=14)

        # 7. Remove Axes
        for row_idx in range(3):
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_{idx:03d}_mpr.png"), dpi=150)
    plt.close(fig)

# Model evaluation
def evaluate_model(model, test_loader, device) -> tuple:
    """
    Evaluates the model and computes class-wise Dice Similarity Coefficients.
    The visualization now uses Multi-Planar Reconstruction (MPR).
    """
    model.eval()
    all_dice_scores = [] # Store all C class-wise dice scores for all samples
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images) # Outputs are logits (B, C, D, H, W)
            
            # Compute dice score (Calculation remains the same)
            pred = torch.nn.functional.softmax(outputs, dim=1)
            target_oh = torch.nn.functional.one_hot(labels, num_classes=Config.NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            
            dice_scores_list = []
            smooth = 1e-5 
            
            for c in range(Config.NUM_CLASSES): # Iterate over ALL classes 0 to 5
                pred_c = pred[:, c, ...].reshape(pred.shape[0], -1)
                target_c = target_oh[:, c, ...].reshape(labels.shape[0], -1)
                
                intersection = (pred_c * target_c).sum(dim=1)
                union = pred_c.sum(dim=1) + target_c.sum(dim=1)
                
                dice_c = (2. * intersection + smooth) / (union + smooth)
                dice_scores_list.append(dice_c)
            
            dice_per_class = torch.stack(dice_scores_list, dim=1) 
            all_dice_scores.append(dice_per_class.cpu().numpy())
            
            # Prepare prediction for visualization
            pred_index_map = torch.argmax(outputs, dim=1) 
            
            # Save visualization for first few samples
            if idx < 5:
                visualize_mpr_comparison(images, labels, pred_index_map, 
                                          Config.VIS_DIR, idx, Config.NUM_CLASSES, 
                                          Config.SPATIAL_SIZE)
    
    # Aggregate scores
    all_dice_scores = np.concatenate(all_dice_scores, axis=0) # Shape: (Num_Samples, C)
    mean_dice_per_class = all_dice_scores.mean(axis=0) # Shape: (6,)
    mean_dice_all = mean_dice_per_class.mean()
    
    # Output
    class_labels = range(Config.NUM_CLASSES)
    print("\nDice Scores (Including Background)")
    for i, score in zip(class_labels, mean_dice_per_class): 
        print(f"Class {i}: {score:.4f}")
    print(f"Overall Mean Dice: {mean_dice_all:.4f}")
    
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