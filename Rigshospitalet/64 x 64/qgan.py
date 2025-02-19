import torch
import torch.nn as nn
import torch.nn.functional as func

### TO-DO
# ...nothing?
###
# UNet structure for pix2pix
class Generator(nn.Module):
    """Modified U-Net w/o upstream"""
    def __init__(self, latent_dim, output_size=32):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim # Upstreaming handled by PTGenerator
        self.output_size = output_size
        
        # Initial layer to project latent vector into spatial dimensions
        # Add pt layer
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),  # Project latent vector to a 4x4 spatial map
            nn.ReLU(inplace=True)
        )
        
        # Downstreaming
        self.dec1 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec3 = self.conv_block(128, 64)
        self.dec4 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        """Upsampling block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z):
        # Start with latent vector input
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)  # Reshape to (batch_size, 512, 4, 4)
        
        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        
        return torch.sigmoid(x) # Val: 0-1


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 5 conv. layer as in paper
        self.model = nn.Sequential(
             nn.ZeroPad2d(2), # Control dimensions
             nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(128,256, kernel_size=2, stride=2, padding=1, bias=False),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(256, 1, kernel_size=2, stride=1, padding=0, bias=False),
         )

    def forward(self, x):
        x = x.to(torch.float32) # Ensure format
        return self.model(x).squeeze(3).squeeze(2).squeeze(1) # Remove singleton dimensions

