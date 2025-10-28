"""
train.py - Training script for 3D U-Net using a combined Cross-Entropy and Dice Loss.
This script is updated to use local pure PyTorch implementations.
"""

import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Local imports
from dataset import get_dataloaders # Import data loading function
from modules import UNet3D # Import the model architecture
from evaluation_functions import DiceLoss, dice_coefficient # Import loss and metric

# Config for compliance on Rangpur
class Config:
    BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open"
    MR_FOLDER = os.path.join(BASE_DIR, "semantic_MRs")
    LABEL_FOLDER = os.path.join(BASE_DIR, "semantic_labels_only")

    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 50
    LR = 1e-3
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    NUM_CLASSES = 6

    TRAIN_SPATIAL_SIZE = (96, 96, 48)
    VAL_SPATIAL_SIZE = (256, 256, 128)

    LOG_FILE = os.path.join(LOG_DIR, "training_log.txt")

    WEIGHT_DECAY = 1e-5
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    
    # Loss weights for combination
    DICE_WEIGHT = 0.5
    CE_WEIGHT = 1.0


# Setup and logging
def setup_directories():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

def setup_logging():
    with open(Config.LOG_FILE, "w") as f:
        f.write("epoch,train_loss,val_dice,learning_rate\n")

def log_metrics(epoch, train_loss, val_dice, lr):
    with open(Config.LOG_FILE, "a") as f:
        f.write(f"{epoch},{train_loss:.6f},{val_dice:.6f},{lr:.8f}\n")

# Combined Loss function using pure PyTorch
def combined_loss_fn(pred, target, criterion_ce, criterion_dice):
    """Calculates the weighted sum of Cross-Entropy and Dice Loss."""
    ce_loss = criterion_ce(pred, target.long())
    dice_loss = criterion_dice(pred, target)
    return Config.CE_WEIGHT * ce_loss + Config.DICE_WEIGHT * dice_loss

# Training + validation loops
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    # Assumes DataLoader returns a tuple: (image_tensor, label_tensor)
    for images, labels in pbar: 
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss using the combined criterion
        loss = criterion(outputs, labels) 
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return epoch_loss / len(train_loader)

def validate(model, val_loader, dice_metric_fn, device):
    """
    Validates model and computes mean Dice Score using the pure PyTorch function.
    """
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        # Assumes DataLoader returns a tuple: (image_tensor, label_tensor)
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Use the pure PyTorch dice_coefficient function
            mean_dice, _ = dice_metric_fn(outputs, labels, num_classes=Config.NUM_CLASSES)
            total_dice += mean_dice.item()
            
    # Calculate the average Dice score over all batches
    return total_dice / len(val_loader)


def train(model, train_loader, val_loader, combined_criterion, dice_metric_fn, optimizer, scheduler, device, epochs):
    best_dice, best_epoch = -1.0, 0

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(model, train_loader, combined_criterion, optimizer, device, epoch)
        
        # Validate using the custom dice_coefficient function
        val_dice = validate(model, val_loader, dice_metric_fn, device)
        
        scheduler.step(val_dice)

        log_metrics(epoch, train_loss, val_dice, current_lr)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Dice={val_dice:.4f}, LR={current_lr:.6e}")

        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'train_loss': train_loss
            }, os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"  ✓ New best model saved (Dice={val_dice:.4f})")

    print(f"\nTraining complete. Best Dice={best_dice:.4f} (Epoch {best_epoch})")

# Main
def main():
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
    model = UNet3D(in_channels=1, out_channels=Config.NUM_CLASSES).to(Config.DEVICE)

    # Initialize pure PyTorch loss functions (nn.CrossEntropyLoss is standard PyTorch)
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