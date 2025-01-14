#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.distributions import Distribution
from torch.utils.data import DataLoader
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import argparse
import pdb

from ptseries.models import PTGenerator

# ---------------------------------------------------------------------
# 1. Define BosonPrior Distribution
# ---------------------------------------------------------------------
class BosonPrior(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, boson_sampler, batch_size, latent_features, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.boson_sampler = boson_sampler
        self.batch_size = batch_size
        self.latent_features = latent_features

    def rsample(self, sample_shape=torch.Size()):
        # Use the PTGenerator's generate method to sample a batch of latent vectors.
        samples = self.boson_sampler.generate(self.batch_size)
        # Convert to a PyTorch tensor if not already
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples, dtype=torch.float32)
        return samples

    def log_prob(self, z):
        # Placeholder: returning zeros
        return torch.zeros(z.shape[0], device=z.device)

# ---------------------------------------------------------------------
# 2. Variational Autoencoder
# ---------------------------------------------------------------------
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape: torch.Size, latent_features: int, boson_sampler_params: dict = None):
        super().__init__()
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = int(np.prod(input_shape))

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

        # Setup BosonSampler if parameters provided
        self.boson_sampler_params = boson_sampler_params
        self.boson_sampler = None
        if boson_sampler_params is not None:
            self.boson_sampler = PTGenerator(**boson_sampler_params)

        # Register default prior params for Gaussian fallback
        self.register_buffer(
            'prior_params',
            torch.zeros(1, 2 * latent_features)
        )

    def posterior(self, x: torch.Tensor) -> Distribution:
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size: int) -> Distribution:
        if self.boson_sampler is not None:
            return BosonPrior(
                boson_sampler=self.boson_sampler, 
                batch_size=batch_size, 
                latent_features=self.latent_features
            )
        else:
            prior_params = self.prior_params.expand(batch_size, -1)
            mu, log_sigma = prior_params.chunk(2, dim=-1)
            return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z: torch.Tensor) -> Distribution:
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape)
        return torch.distributions.Bernoulli(logits=px_logits, validate_args=False)

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

# ---------------------------------------------------------------------
# 3. ReparameterizedDiagonalGaussian Distribution
# ---------------------------------------------------------------------
class ReparameterizedDiagonalGaussian(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, mu: torch.Tensor, log_sigma: torch.Tensor, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> torch.Tensor:
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> torch.Tensor:
        epsilon = self.sample_epsilon()
        return self.mu + self.sigma * epsilon

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_scale = torch.log(self.sigma)
        return -0.5 * (
            ((z - self.mu) ** 2) / (self.sigma ** 2) 
            + 2 * log_scale 
            + torch.log(torch.tensor(2 * torch.pi, device=z.device))
        )

# ---------------------------------------------------------------------
# 4. Variational Inference Module
# ---------------------------------------------------------------------
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
        beta_elbo = log_px - self.beta * kl
        loss = -beta_elbo.mean()

        diagnostics = {
            'elbo': (log_px - kl).detach(),
            'log_px': log_px.detach(),
            'kl': kl.detach()
        }
        return loss, diagnostics, outputs

# ---------------------------------------------------------------------
# 5. Main Function
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--boson_sampler_params",
        default="true",
        choices=["true", "false"],
        help="When 'true' or not given, we use the Boson Sampler. When 'false', we use Gaussian."
    )
    args = parser.parse_args()

    # Define boson_sampler_params based on user choice
    if args.boson_sampler_params == "false":
        boson_params_to_use = None
        print(">> Using standard Gaussian prior.")
    else:
        boson_params_to_use = {
            "input_state": [1, 0, 1, 0, 1, 0, 1, 0],
            "tbi_params": {
                "input_loss": 0.0,
                "detector_efficiency": 1,
                "bs_loss": 0,
                "bs_noise": 0,
                "distinguishable": False,
                "n_signal_detectors": 0,
                "g2": 0,
                "tbi_type": "multi-loop",
                "n_loops": 3,
                "loop_lengths": [2, 3, 4],
                "postselected": True
            },
            "n_tiling": 1
        }
        print(">> Using Boson Sampler as prior.")

    # Hyperparameters
    lr = 1e-3
    num_epochs = 10
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST data
    train_data = MNIST(root=".", train=True, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize VAE
    latent_features = 8
    vae = VariationalAutoencoder(
        input_shape=torch.Size([28*28]),
        latent_features=latent_features,
        boson_sampler_params=boson_params_to_use
    ).to(device)

    vi = VariationalInference(beta=1.0)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)
            loss, diagnostics, outputs = vi(vae, x)
            pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()
