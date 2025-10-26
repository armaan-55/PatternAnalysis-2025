import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
import dataset
import modules
from modules import UNet3D, dice_loss, dice_coefficient
from dataset import Prostate3DDataset, get_paired_paths

# Rangpur logging functionality
LOG_FILE = "logs/dice_per_epoch.txt"
os.makedirs("logs", exist_ok=True)
with open(LOG_FILE, "w") as f:
    f.write("epoch,train_loss,val_dice\n")

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 1
VISUALIZE_EVERY = 2
NUM_CLASSES = 6
TARGET_SHAPE = (256, 256, 128)

# Data loading
print("Loading dataset...")
base_dir = "/home/groups/comp3710/HipMRI_Study_open"
mr_folder = os.path.join(base_dir, "semantic_MRs")
label_folder = os.path.join(base_dir, "semantic_labels_only")

image_paths, label_paths = get_paired_paths(mr_folder, label_folder)

# Split 90% train / 10% validation
split_idx = int(0.9 * len(image_paths))
train_img, val_img = image_paths[:split_idx], image_paths[split_idx:]
train_lbl, val_lbl = label_paths[:split_idx], label_paths[split_idx:]

train_dataset = Prostate3DDataset(train_img, train_lbl, num_classes=NUM_CLASSES, target_shape=TARGET_SHAPE)
val_dataset = Prostate3DDataset(val_img, val_lbl, num_classes=NUM_CLASSES, target_shape=TARGET_SHAPE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val samples")

# Model setup
model = UNet3D(in_channels=1, out_channels=NUM_CLASSES, dropout_p=0.2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Model initialized:", model.__class__.__name__)

# Visualization functions
def show_3d_predictions(model, dataset, epoch=1, n=2, slice_idx=None, device='cpu'):
    """Visualize model predictions for 3D MRI volumes."""
    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(10, 4 * n))
    fig.suptitle(f"Predictions After Epoch {epoch}", fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).cpu()
            pred_labels = torch.argmax(pred[0], dim=0)
            true_labels = torch.argmax(y, dim=0)

            if slice_idx is None:
                slice_idx = pred_labels.shape[2] // 2

            axes[i, 0].imshow(x[0, 0, :, :, slice_idx].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title("Input MRI")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(true_labels[:, :, slice_idx].numpy(), cmap="tab10")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_labels[:, :, slice_idx].numpy(), cmap="tab10")
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
    model.train()


def plot_losses(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, 'bo-', linewidth=2)
    plt.title("Dice Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.show()

# Training loop
def train_3d(model, train_loader, val_dataset, epochs=10, lr=1e-4, visualize_every=1, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    val_dices = []

    print("Starting 3D U-Net training with Dice Loss...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            preds = model(x)
            loss = dice_loss(preds, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Full validation Dice over all validation samples
        model.eval()
        with torch.no_grad():
            dice_scores = []
            for i in range(len(val_dataset)):
                vx, vy = val_dataset[i]
                vx = vx.unsqueeze(0).to(device)
                preds = model(vx)
                dice_val = dice_coefficient(preds, vy.unsqueeze(0).to(device))
                dice_scores.append(dice_val.item())
            mean_dice = np.mean(dice_scores)
            val_dices.append(mean_dice)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Dice={mean_dice:.4f}")

            # Logging for rangpur
            with open(LOG_FILE, "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{mean_dice:.4f}\n")
        if (epoch + 1) % visualize_every == 0:
            show_3d_predictions(model, val_dataset, epoch + 1, device=device)

    print("Training complete!")
    plot_losses(losses)

    # Final average validation Dice over all epochs
    final_avg_dice = np.mean(val_dices)
    print(f"\nFinal average validation Dice over {epochs} epochs: {final_avg_dice:.4f}")

    return losses, val_dices

# Run training
if __name__ == "__main__":
    losses, val_dices = train_3d(model, train_loader, val_dataset,
                                 epochs=EPOCHS, lr=LR,
                                 visualize_every=VISUALIZE_EVERY,
                                 device=DEVICE)

    # Save trained model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/unet3d_dice.pth")
    print("Model saved to checkpoints/unet3d_dice.pth")