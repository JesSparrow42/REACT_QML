import os
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

class UNetBlockUp(nn.Module):
    """
    One up-sampling block: up-conv -> concat -> Conv -> Conv.
    Now accepts skip connection channels explicitly.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # upconv reduces channel count from in_channels to out_channels
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, channels = out_channels (from upconv) + skip_channels
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        # If spatial dimensions differ slightly due to pooling/cropping, adjust them.
        if x.shape[2:] != skip.shape[2:]:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            skip = skip[:, :, diffY // 2 : diffY // 2 + x.size()[2], diffX // 2 : diffX // 2 + x.size()[3]]
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetBlockDown(nn.Module):
    """
    One down-sampling block: Conv -> Conv -> optional down-sample.
    """
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.pool = pool
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        before_pool = x
        if self.pool:
            x = self.pool_layer(x)
        return x, before_pool

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super().__init__()
        # Encoder
        self.down1 = UNetBlockDown(in_channels, base_filters, pool=False)              # out: 64
        self.down2 = UNetBlockDown(base_filters, base_filters * 2, pool=True)            # out: 128
        self.down3 = UNetBlockDown(base_filters * 2, base_filters * 4, pool=True)        # out: 256
        self.down4 = UNetBlockDown(base_filters * 4, base_filters * 8, pool=True)        # out: 512

        # Decoder
        # Note: The skip connection in each up block is taken from the same block's "before pooling" output.
        # For down4, skip4 has 512 channels, so we set skip_channels accordingly.
        self.up1 = UNetBlockUp(in_channels=base_filters * 8, skip_channels=base_filters * 8, out_channels=base_filters * 4)
        self.up2 = UNetBlockUp(in_channels=base_filters * 4, skip_channels=base_filters * 4, out_channels=base_filters * 2)
        self.up3 = UNetBlockUp(in_channels=base_filters * 2, skip_channels=base_filters * 2, out_channels=base_filters)

        # Final conv => out_channels
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x1, skip1 = self.down1(x)  # skip1: (B,64,H,W)
        x2, skip2 = self.down2(x1) # skip2: (B,128,H,W)
        x3, skip3 = self.down3(x2) # skip3: (B,256,H/2,W/2)
        x4, skip4 = self.down4(x3) # skip4: (B,512,H/4,W/4)
        # Decoder: each up block gets the corresponding skip connection.
        # Here, x4 (B,512,H/8,W/8) is upsampled and concatenated with skip4 (B,512,H/4,W/4)
        x = self.up1(x4, skip4)  # After up1, output: (B,256,H/4,W/4)
        x = self.up2(x, skip3)   # After up2, output: (B,128,H/2,W/2)
        x = self.up3(x, skip2)   # After up3, output: (B,64,H,W)
        out = self.final_conv(x)
        return torch.sigmoid(out)
