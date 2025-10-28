import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D Unet model now following architecture described by Jiangtao et al. from https://arxiv.org/pdf/2502.06895

# Decoder block
class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and convolution.
    Pads input feature maps to match encoder size before concatenation.
    """
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        
        layers = [
            nn.Conv3d(skip_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)
        # Compute size difference for padding
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# 3D Unet model
class UNet3D(nn.Module):
    """
    5-Level 3D U-Net with architecture as per source 
    """
    def __init__(self, in_channels=1, out_channels=6, dropout_p=0.0):
        super().__init__()
        BASE = 8 
                
        # Conv3D Block 1: in_channels (e.g., 1) -> 8 (Skip: e1)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, BASE, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(BASE, BASE, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm3d(BASE),
        )
        self.pool1 = nn.MaxPool3d(2) 

        # Conv3D Block 2: 8 -> 16 (Skip: e2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(BASE, BASE*2, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(BASE*2, BASE*2, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm3d(BASE*2),
        )
        self.pool2 = nn.MaxPool3d(2) 

        # Conv3D Block 3: 16 -> 32 (Skip: e3)
        self.enc3 = nn.Sequential(
            nn.Conv3d(BASE*2, BASE*4, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(BASE*4, BASE*4, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm3d(BASE*4),
        )
        self.pool3 = nn.MaxPool3d(2) 

        # Conv3D Block 4: 32 -> 64 (Skip: e4)
        self.enc4 = nn.Sequential(
            nn.Conv3d(BASE*4, BASE*8, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(BASE*8, BASE*8, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm3d(BASE*8),
        )
        self.pool4 = nn.MaxPool3d(2) 

        # --- BOTTLENECK (64 -> 128) ---
        self.bottleneck = nn.Sequential(
            nn.Conv3d(BASE*8, BASE*16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(BASE*16, BASE*16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm3d(BASE*16),
            nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()
        )

        # --- DECODER (Upsampling Path) ---
        
        # UpSampling Block 1: 128 + 64 -> 64
        self.dec4 = DecoderBlock(BASE*16, BASE*8, BASE*8) 
        
        # UpSampling Block 2: 64 + 32 -> 32
        self.dec3 = DecoderBlock(BASE*8, BASE*4, BASE*4)   
        
        # UpSampling Block 3: 32 + 16 -> 16
        self.dec2 = DecoderBlock(BASE*4, BASE*2, BASE*2)   
        
        # UpSampling Block 4: 16 + 8 -> 8
        self.dec1 = DecoderBlock(BASE*2, BASE, BASE)       

        # Final convolution: 8 -> out_channels (e.g., 6)
        self.out_conv = nn.Conv3d(BASE, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                   
        p1 = self.pool1(e1)                 
        e2 = self.enc2(p1)                  
        p2 = self.pool2(e2)                 
        e3 = self.enc3(p2)                  
        p3 = self.pool3(e3)                 
        e4 = self.enc4(p3)                  
        p4 = self.pool4(e4)                 

        # Bottleneck
        b = self.bottleneck(p4)             

        # Decoder
        d4 = self.dec4(b, e4)              
        d3 = self.dec3(d4, e3)              
        d2 = self.dec2(d3, e2)             
        d1 = self.dec1(d2, e1)              

        out = self.out_conv(d1)            
        return out