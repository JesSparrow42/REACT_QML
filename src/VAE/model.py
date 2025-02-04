# model.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch import nn
from torch.distributions import Distribution
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
        # Note: DISCRIMINATOR_ITER should be defined externally or passed in
        total_samples = (1 + DISCRIMINATOR_ITER) * self.batch_size
        latent = self.boson_sampler.generate(total_samples).to(self.device)
        latent = torch.chunk(latent, 1 + DISCRIMINATOR_ITER, dim=0)
        return latent

    def log_prob(self, z):
        mean = torch.zeros_like(z)
        std = torch.ones_like(z)
        log_scale = torch.log(std)
        log_prob = -0.5 * (
            ((z - mean) ** 2) / (std ** 2)
            + 2 * log_scale
            + torch.log(torch.tensor(2 * torch.pi, device=z.device))
        )
        return log_prob.sum(dim=-1)

# ---------------------------------------------------------------------
# 2. Variational Autoencoder
# ---------------------------------------------------------------------
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape: torch.Size, latent_features: int, boson_sampler_params: dict = None):
        super().__init__()
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = int(np.prod(input_shape))  # For 128x128, this would be 16384

        # Adjust encoder to accept a larger input vector (16384 vs. 784)
        self.encoder = nn.Sequential(
            nn.Linear(self.observation_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2 * latent_features)
        )

        # Adjust decoder correspondingly
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, self.observation_features)
        )

        # Setup BosonSampler if parameters provided
        self.boson_sampler_params = boson_sampler_params
        self.boson_sampler = None
        if boson_sampler_params is not None:
            self.boson_sampler = PTGenerator(**boson_sampler_params)

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
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}

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

        # Reshape x to match the logits shape (batch_size, 128, 128)
        x_image = x.view(x.size(0), *model.input_shape)
        log_px = reduce_sum(px.log_prob(x_image))
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
# Utility function to save images
# ---------------------------------------------------------------------
def save_images(original, reconstructed, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(original):
        plt.imsave(
            os.path.join(output_dir, f"epoch_{epoch}_original_{i}.png"),
            img.reshape(28, 28),
            cmap="gray"
        )
    for i, img in enumerate(reconstructed):
        plt.imsave(
            os.path.join(output_dir, f"epoch_{epoch}_reconstructed_{i}.png"),
            img.reshape(28, 28),
            cmap="gray"
        )

# ---------------------------------------------------------------------
# Integrated Lightning Module combining model and training logic
# ---------------------------------------------------------------------
class VAE_Lightning(pl.LightningModule):
    def __init__(self, boson_params_to_use, lr, latent_features, output_dir):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.output_dir = output_dir
        self.latent_features = latent_features

        # For 128x128 images, input_shape should be torch.Size([128, 128])
        self.vae = VariationalAutoencoder(
            input_shape=torch.Size([128, 128]),
            latent_features=self.hparams.latent_features,
            boson_sampler_params=self.hparams.boson_params_to_use
        )
        self.vi = VariationalInference(beta=1.0)

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1).to(self.device)
        loss, diagnostics, _ = self.vi(self.vae, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Save images on the first batch of each epoch
        if batch_idx == 0:
            x, _ = batch
            x = x.view(x.size(0), -1).to(self.device)
            with torch.no_grad():
                _, _, outputs = self.vi(self.vae, x)[:3]
                reconstructed = outputs["px"].probs.cpu().numpy()
                original = x.cpu().numpy()
                save_images(original, reconstructed, self.output_dir, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=1e-3)
