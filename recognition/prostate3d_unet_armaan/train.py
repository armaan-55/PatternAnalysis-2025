"""
train.py - 3D U-Net Training Methods

A script to train the final trained 3D U-Net model on the Prostate 3D training set,
and perform validation on the validation set.
It computes class-wise Dice metrics and overall model loss through each epoch.

Uses checkpoints, logging, and early stopping.

Author: Armaan Aulakh
Date: October 30 2025
"""

import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dataset import get_dataloaders
from modules import UNet3D
from evaluation_functions import DiceLoss, dice_coefficient

class Config:
    """
    Configuration class holding all paths, hyperparameters, and device settings 
    for the training run.

    Attributes:
        DEVICE (str): The execution device ('cuda', 'mps', or 'cpu'). Determined 
                      based on availability.
        BASE_DIR (str): Root directory for data storage, adjusted based on 
                        the detected environment (local vs. remote/Rangpur).
        MR_FOLDER (str): Full path to the directory containing MRI volumes.
        LABEL_FOLDER (str): Full path to the directory containing label volumes.
        LOG_DIR (str): Directory for logging files.
        CHECKPOINT_DIR (str): Directory for saving model checkpoints.
        EPOCHS (int): Total number of epochs to train for.
        LR (float): Initial learning rate for the optimizer.
        BATCH_SIZE (int): Batch size used during training.
        NUM_WORKERS (int): Number of subprocesses for data loading.
        NUM_CLASSES (int): Total number of segmentation classes.
        TRAIN_SPATIAL_SIZE (tuple): Target size (D, H, W) for training data.
        VAL_SPATIAL_SIZE (tuple): Target size (D, H, W) for validation data.
        LOG_FILE (str): Full path to the CSV log file.
        WEIGHT_DECAY (float): L2 regularization factor for the optimizer.
        LR_PATIENCE (int): Number of validation epochs to wait for improvement 
                           before reducing the learning rate.
        LR_FACTOR (float): Factor by which the learning rate will be reduced.
        DICE_WEIGHT (float): Weight of the Dice Loss component in the combined loss.
        CE_WEIGHT (float): Weight of the Cross-Entropy Loss component in the combined loss.
    """

    if torch.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    if torch.mps.is_available():
        BASE_DIR = "recognition/Prostate3D_local"
    else:
        BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open"

    MR_FOLDER = os.path.join(BASE_DIR, "semantic_MRs")
    LABEL_FOLDER = os.path.join(BASE_DIR, "semantic_labels_only")

    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"
    EPOCHS = 50
    LR = 1e-3
    BATCH_SIZE = 1
    NUM_WORKERS = 1
    NUM_CLASSES = 6

    TRAIN_SPATIAL_SIZE = (192, 192, 96)
    VAL_SPATIAL_SIZE = (192, 192, 96)

    LOG_FILE = os.path.join(LOG_DIR, "training_log.txt")

    WEIGHT_DECAY = 1e-5
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    
    DICE_WEIGHT = 1.0
    CE_WEIGHT = 1.0


# Setup and logging
def setup_directories():
    """
    Creates the necessary directories for logs and model checkpoints 
    if they do not already exist.
    """
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

def setup_logging():
    """
    Initializes the training log file (Config.LOG_FILE) and writes the CSV header.
    """
    with open(Config.LOG_FILE, "w") as f:
        f.write("epoch,train_loss,val_dice,learning_rate\n")

def log_metrics(epoch, train_loss, val_dice, lr):
    """
    Appends the current epoch's metrics to the training log file.

    Args:
        epoch (int): The current epoch number.
        train_loss (float): The average training loss for the epoch.
        val_dice (float): The mean Dice score on the validation set.
        lr (float): The current learning rate.
    """
    with open(Config.LOG_FILE, "a") as f:
        f.write(f"{epoch},{train_loss:.6f},{val_dice:.6f},{lr:.8f}\n")

# Combined Loss function using pure PyTorch
def combined_loss_fn(pred, target, criterion_ce, criterion_dice):
    """
    Calculates the weighted sum of Cross-Entropy Loss and Dice Loss.

    Args:
        pred (torch.Tensor): Model logits (B, C, D, H, W).
        target (torch.Tensor): Ground truth sparse index map (B, D, H, W).
        criterion_ce (nn.CrossEntropyLoss): Initialized Cross-Entropy loss object.
        criterion_dice (DiceLoss): Initialized Dice loss object.

    Returns:
        torch.Tensor: The scalar combined weighted loss.
    """
    ce_loss = criterion_ce(pred, target.long())
    dice_loss = criterion_dice(pred, target)
    return Config.CE_WEIGHT * ce_loss + Config.DICE_WEIGHT * dice_loss

# Training + validation loops
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Runs the model through one full pass of the training data.

    Applies forward pass, computes loss, backpropagation, gradient clipping, 
    and optimizer step.

    Args:
        model (nn.Module): The 3D U-Net model.
        train_loader (DataLoader): DataLoader for the training set.
        criterion (callable): The combined loss function.
        optimizer (optim.Optimizer): The initialized optimizer (e.g., AdamW).
        device (str): The device ('cuda', 'mps', 'cpu') to use.
        epoch (int): The current epoch number (for logging/progress bar).

    Returns:
        float: The average loss across all batches in the training loader.
    """
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for images, labels in pbar: 
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels) 
        
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return epoch_loss / len(train_loader)

def validate(model, val_loader, dice_metric_fn, device):
    """
    Validates the model on the validation set and computes the mean Dice Score.

    Performs inference in `torch.no_grad()` context. Reports both the overall 
    mean Dice and the breakdown of Dice per foreground class.

    Args:
        model (nn.Module): The 3D U-Net model.
        val_loader (DataLoader): DataLoader for the validation set (batch_size=1).
        dice_metric_fn (callable): The `dice_coefficient` function imported 
                                   from `evaluation_functions`.
        device (str): The device ('cuda', 'mps', 'cpu') to use.

    Returns:
        float: The average mean Dice score across all validation samples (excluding background).
    """
    model.eval()
    total_dice = 0.0
    all_dice_per_class = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            mean_dice, dice_per_class = dice_metric_fn(outputs, labels, num_classes=Config.NUM_CLASSES)
            total_dice += mean_dice.item()
            all_dice_per_class.append(dice_per_class.mean(dim=0).cpu())  # Average across batch
            
    avg_dice = total_dice / len(val_loader)
    
    # Print per-class breakdown
    if len(all_dice_per_class) > 0:
        class_dice = torch.stack(all_dice_per_class).mean(dim=0)
        print(f"\n  Per-class Dice: {' | '.join([f'C{i+1}: {d:.3f}' for i, d in enumerate(class_dice)])}")
    
    return avg_dice


def train(model, train_loader, val_loader, combined_criterion, dice_metric_fn, optimizer, scheduler, device, epochs):
    """
    The main training loop, handling epoch iteration, validation, scheduling, 
    checkpoint saving, and early stopping.

    Args:
        model (nn.Module): The 3D U-Net model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        combined_criterion (callable): The loss function for training.
        dice_metric_fn (callable): The metric function for validation.
        optimizer (optim.Optimizer): The initialized optimizer.
        scheduler (ReduceLROnPlateau): The learning rate scheduler.
        device (str): The device to run on.
        epochs (int): Total number of epochs defined in Config.
    """
    best_dice, best_epoch = -1.0, 0
    patience_counter = 0
    early_stop_patience = 10  # Stop if no improvement for 10 epochs

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(model, train_loader, combined_criterion, optimizer, device, epoch)
        val_dice = validate(model, val_loader, dice_metric_fn, device)
        
        scheduler.step(val_dice)
        log_metrics(epoch, train_loss, val_dice, current_lr)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Dice={val_dice:.4f}, LR={current_lr:.6e}")

        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            patience_counter = 0  # Reset counter
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'train_loss': train_loss
            }, os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best model saved (Dice={val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stop_patience} epochs)")
                break

    print(f"\nTraining complete. Best Dice={best_dice:.4f} (Epoch {best_epoch})")

# Main
def main():
    """
    Entry point of the training script.

    1. Sets up log and checkpoint directories.
    2. Initializes data loaders.
    3. Instantiates the UNet3D model and moves it to the target device.
    4. Initializes loss functions (Cross-Entropy and Dice).
    5. Initializes optimizer (AdamW) and LR scheduler (ReduceLROnPlateau).
    6. Starts the main training loop.
    """
    setup_directories()
    setup_logging()

    # Get dataloaders from dataset.py
    train_loader, val_loader, _ = get_dataloaders(
        mr_folder=Config.MR_FOLDER,
        label_folder=Config.LABEL_FOLDER,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        train_spatial_size=Config.TRAIN_SPATIAL_SIZE,
        val_spatial_size=Config.VAL_SPATIAL_SIZE,
        num_classes=Config.NUM_CLASSES,
    )

    # UNet3D model from modules.py
    model = UNet3D(in_channels=1, out_channels=Config.NUM_CLASSES, dropout_p=0.2).to(Config.DEVICE)

    # Initialize pure PyTorch loss functions
    criterion_ce = nn.CrossEntropyLoss().to(Config.DEVICE) 
    criterion_dice = DiceLoss(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    # Define the combined criterion function for training
    combined_criterion = lambda pred, target: combined_loss_fn(pred, target, criterion_ce, criterion_dice)
    
    # Get the Dice metric function from evaluation_functions.py
    dice_metric_fn = dice_coefficient 

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=Config.LR_FACTOR, patience=Config.LR_PATIENCE)

    print(f"Model, optimizer, and losses ready on {Config.DEVICE}")
    train(model, train_loader, val_loader, combined_criterion, dice_metric_fn, optimizer, scheduler, Config.DEVICE, Config.EPOCHS)

if __name__ == "__main__":
    main()