#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.distributions import Bernoulli, Distribution
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# Define your ReparameterizedDiagonalGaussian
class ReparameterizedDiagonalGaussian(Distribution):
    arg_constraints = {}
    
    def __init__(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        super().__init__()
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> torch.Tensor:
        return torch.empty_like(self.mu).normal_()
        
    def rsample(self) -> torch.Tensor:
        epsilon = self.sample_epsilon()
        return self.mu + self.sigma * epsilon

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_scale = torch.log(self.sigma)
        return -0.5 * (
            (z - self.mu) ** 2 / self.sigma**2
            + 2 * log_scale
            + torch.log(torch.tensor(2 * torch.pi))
        )

# VAE model
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape: torch.Size, latent_features: int) -> None:
        super(VariationalAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.observation_features, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2 * latent_features)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.observation_features)
        )
        
        # p(z) = N(0, I)
        self.register_buffer('prior_params', 
                             torch.zeros(torch.Size([1, 2 * latent_features])))

    def posterior(self, x: torch.Tensor) -> Distribution:
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size: int) -> Distribution:
        prior_params = self.prior_params.expand(batch_size, -1)
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z: torch.Tensor) -> Distribution:
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape)
        return Bernoulli(logits=px_logits, validate_args=False)


    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        qz = self.posterior(x)
        pz = self.prior(x.size(0))
        z = qz.rsample()
        px = self.observation_model(z)
        return {
            'px': px,
            'pz': pz,
            'qz': qz,
            'z': z
        }

class VariationalInference(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model, x):
        outputs = model(x)
        px, pz, qz, z = outputs['px'], outputs['pz'], outputs['qz'], outputs['z']

        def reduce_sum(tensor):
            return tensor.view(tensor.size(0), -1).sum(dim=1)

        log_px = reduce_sum(px.log_prob(x))
        log_pz = reduce_sum(pz.log_prob(z))
        log_qz = reduce_sum(qz.log_prob(z))

        kl = log_qz - log_pz
        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl
        loss = -beta_elbo.mean()

        diagnostics = {
            'elbo': elbo.detach(),
            'log_px': log_px.detach(),
            'kl': kl.detach()
        }
        return loss, diagnostics, outputs

def main():
    # Hyperparams
    lr = 1e-3
    num_epochs = 10
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset (MNIST)
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    
    # Flatten images
    flatten = lambda x: ToTensor()(x).view(28**2)

    train_data = MNIST(root=".", train=True, transform=flatten, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Create model
    latent_features = 10
    vae = VariationalAutoencoder(input_shape=torch.Size([28*28]), 
                                 latent_features=latent_features).to(device)
    vi = VariationalInference(beta=1.0)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            loss, diagnostics, outputs = vi(vae, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Optionally save the model
    torch.save(vae.state_dict(), "vae.pth")

if __name__ == "__main__":
    main()

