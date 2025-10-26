import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    """3D UNet for HipMRI Prostate image slices with Dice evaluation"""

    def __init__(self, in_channels=1, out_channels=6, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, 32, dropout_p)
        self.enc2 = self._conv_block(32, 64, dropout_p)
        self.enc3 = self._conv_block(64, 128, dropout_p)

        # Decoder (upsampling)
        self.dec3 = self._conv_block(128 + 64, 64, dropout_p)
        self.dec2 = self._conv_block(64 + 32, 32, dropout_p)
        self.final_conv = nn.Conv3d(32, out_channels, 1)

        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def _conv_block(self, in_ch, out_ch, dropout_p=0.2):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_p),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_p)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(e3), e2], 1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1))
        out = self.final_conv(d2)
        out = torch.softmax(out, dim=1)
        return out
    
def dice_coefficient(pred, target, epsilon=1e-6):
    """Compute mean Dice similarity coefficient per batch."""
    pred = torch.argmax(pred, dim=1)  # [B, H, W, D]
    target = torch.argmax(target, dim=1)  # assuming one-hot target

    dice = 0
    for c in range(pred.max() + 1):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice += (2 * intersection + epsilon) / (union + epsilon)
    return dice / (pred.max() + 1)


def dice_loss(pred, target, epsilon=1e-6):
    """Differentiable dice loss for multi-class segmentation."""
    pred = F.softmax(pred, dim=1)
    target = target.float()

    intersection = torch.sum(pred * target, dim=(2, 3, 4))
    union = torch.sum(pred + target, dim=(2, 3, 4))
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    loss = 1 - dice_score.mean()
    return loss