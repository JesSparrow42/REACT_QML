import torch
import torch.nn as nn


class Critic(nn.Module):
    """A critic model using a classical network"""

    def __init__(self):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            # in: 1 x 28 x 28
            nn.ZeroPad2d(2),
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 4 x 4
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: latent_dim x 1 x 1
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.net(x).squeeze(3).squeeze(2).squeeze(1)


class Generator(nn.Module):
    """A generator model that uses the PT Series to generate an input probability distribution"""

    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # in: Nimages x latent_dim x 1 x 1
            nn.ConvTranspose2d(self.latent_dim, 128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            # out: 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # out: 64 x 8 x 8
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # out: 32 x 16 x 16
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=2, bias=False),
            nn.Sigmoid(),
            # out: 1 x 32 x 32
        )

    def forward(self, x):
        x = x.to(torch.float32).unsqueeze(2).unsqueeze(3)
        return self.net(x)
