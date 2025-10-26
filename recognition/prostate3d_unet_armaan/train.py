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

# Logs and visualization directories
LOG_FILE = "logs/dice_per_epoch.txt"
VIS_DIR = "logs/visualizations"
os.makedirs("logs", exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
with open(LOG_FILE, "w") as f:
    f.write("epoch,train_loss,val_dice\n")

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
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

# Split 80% train / 10% val / 10% test
num_total = len(image_paths)
train_end = int(0.8 * num_total)
val_end = int(0.9 * num_total)

train_img, val_img, test_img = image_paths[:train_end], image_paths[train_end:val_end], image_paths[val_end:]
train_lbl, val_lbl, test_lbl = label_paths[:train_end], label_paths[train_end:val_end], label_paths[val_end:]

train_dataset = Prostate3DDataset(train_img, train_lbl, num_classes=NUM_CLASSES, target_shape=TARGET_SHAPE)
val_dataset = Prostate3DDataset(val_img, val_lbl, num_classes=NUM_CLASSES, target_shape=TARGET_SHAPE)
test_dataset = Prostate3DDataset(test_img, test_lbl, num_classes=NUM_CLASSES, target_shape=TARGET_SHAPE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")

# Model setup
model = UNet3D(in_channels=1, out_channels=NUM_CLASSES, dropout_p=0.2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Model initialized:", model.__class__.__name__)

# Visualization function (saves images instead of showing)
def save_3d_predictions(model, dataset, save_dir, epoch=0, n=2, slice_idx=None, device='cpu', prefix="val"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).cpu()
            pred_labels = torch.argmax(pred[0], dim=0)
            true_labels = torch.argmax(y, dim=0)

            if slice_idx is None:
                slice_idx = pred_labels.shape[2] // 2

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(x[0, 0, :, :, slice_idx].cpu().numpy(), cmap='gray')
            axes[0].set_title("Input MRI"); axes[0].axis("off")
            axes[1].imshow(true_labels[:, :, slice_idx].numpy(), cmap="tab10")
            axes[1].set_title("Ground Truth"); axes[1].axis("off")
            axes[2].imshow(pred_labels[:, :, slice_idx].numpy(), cmap="tab10")
            axes[2].set_title("Prediction"); axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_epoch{epoch}_sample{i}.png"))
            plt.close(fig)
    model.train()

def plot_losses(losses, save_path="logs/losses.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, 'bo-', linewidth=2)
    plt.title("Dice Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

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

        # Validation
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

            with open(LOG_FILE, "a") as f:
                f.write(f"{epoch+1},{avg_loss:.4f},{mean_dice:.4f}\n")

        if (epoch + 1) % visualize_every == 0:
            save_3d_predictions(model, val_dataset, VIS_DIR, epoch+1, device=device, prefix="val")

    print("Training complete!")
    plot_losses(losses)

    # Final average validation Dice
    final_avg_dice = np.mean(val_dices)
    print(f"\nFinal average validation Dice over {epochs} epochs: {final_avg_dice:.4f}")

    return losses, val_dices

# Test evaluation
def evaluate_test(model, test_dataset, device='cpu'):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, y = test_dataset[i]
            x = x.unsqueeze(0).to(device)
            preds = model(x)
            dice_val = dice_coefficient(preds, y.unsqueeze(0).to(device))
            dice_scores.append(dice_val.item())

    mean_test_dice = np.mean(dice_scores)
    print(f"\nFinal Dice on test set: {mean_test_dice:.4f}")

    # Save visualizations of test predictions
    save_3d_predictions(model, test_dataset, VIS_DIR, epoch=0, device=device, prefix="test")
    return mean_test_dice

# Run training
if __name__ == "__main__":
    losses, val_dices = train_3d(model, train_loader, val_dataset,
                                 epochs=EPOCHS, lr=LR,
                                 visualize_every=VISUALIZE_EVERY,
                                 device=DEVICE)

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/unet3d_dice.pth")
    print("Model saved to checkpoints/unet3d_dice.pth")

    # Evaluate on test set
    evaluate_test(model, test_dataset, device=DEVICE)