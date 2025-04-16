import os
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

# -----------------------------
# GAN Components
# -----------------------------
class Generator(nn.Module):
    """Modified U-Net without upstream"""
    def __init__(self, latent_dim, output_size=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(inplace=True)
        )

        self.dec1 = self.conv_block(512, 256)  # 8 -> 16
        self.dec2 = self.conv_block(256, 128)  # 16 -> 32
        self.final = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)  # 32 -> 64

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.dec1(x)  # 16x16
        x = self.dec2(x)  # 32x32
        x = self.final(x) # 64x64
        return torch.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), # 8 -> 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False), # 4 -> 1
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Add this final conv to reduce to 1 channel
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.features(x)  # shape: (batch_size, 512, 1, 1)
        x = self.final(x)     # shape: (batch_size, 1, 1, 1)
        return x.view(-1)     # returns (batch_size,)
