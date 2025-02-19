# model.py
import os
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.distributions import Distribution
from ptseries.models import PTGenerator  # your boson generator

from vae.utils import save_images, dice_loss  # and any other helper functions as needed

### To do
# Parameter shift rule?
# Benchmarking
# QM9

# -----------------------------
# VAE Components
# -----------------------------
class BosonPrior(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, boson_sampler, batch_size, latent_features, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.boson_sampler = boson_sampler
        self.batch_size = batch_size
        self.latent_features = latent_features
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

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

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape: torch.Size, latent_features: int, boson_sampler_params: dict = None):
        super().__init__()
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = int(np.prod(input_shape))  # e.g. 128x128 -> 16384

        self.encoder = nn.Sequential(
            nn.Linear(self.observation_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2 * latent_features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, self.observation_features)
        )

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

class VariationalInference(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model, x, target=None):
        outputs = model(x)
        px, pz, qz, z = outputs['px'], outputs['pz'], outputs['qz'], outputs['z']

        def reduce_sum(tensor):
            return tensor.view(tensor.size(0), -1).sum(dim=1)

        if target is None:
            x_target = x.view(x.size(0), *model.input_shape)
        else:
            x_target = target

        log_px = reduce_sum(px.log_prob(x_target))
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

class VAE_Lightning(pl.LightningModule):
    def __init__(self, boson_params_to_use, lr, latent_features, output_dir):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.output_dir = output_dir
        self.latent_features = latent_features

        self.vae = VariationalAutoencoder(
            input_shape=torch.Size([128, 128]),
            latent_features=self.hparams.latent_features,
            boson_sampler_params=self.hparams.boson_params_to_use
        )
        self.vi = VariationalInference(beta=1.0)

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        pet_image, ct_image = batch
        pet_image_flat = pet_image.view(pet_image.size(0), -1).to(self.device)
        ct_image_target = ct_image[:, 0, :, :].to(self.device)
        ct_image_target = F.interpolate(ct_image_target.unsqueeze(1),
                                        size=self.vae.input_shape,
                                        mode='bilinear',
                                        align_corners=False).squeeze(1)

        loss, diagnostics, outputs = self.vi(self.vae, pet_image_flat, target=ct_image_target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pet_image, ct_image = batch
        pet_image_flat = pet_image.view(pet_image.size(0), -1).to(self.device)
        ct_image_target = ct_image[:, 0, :, :].to(self.device)
        # Interpolate to match the VAE's expected input shape (e.g., 128x128)
        ct_image_target = F.interpolate(ct_image_target.unsqueeze(1),
                                        size=self.vae.input_shape,
                                        mode='bilinear',
                                        align_corners=False).squeeze(1)

        loss, diagnostics, outputs = self.vi(self.vae, pet_image_flat, target=ct_image_target)
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            print(f"Validation step at epoch {self.current_epoch}")
            with torch.no_grad():
                reconstructed_ct = outputs["px"].probs.cpu().numpy()
                original_ct = ct_image_target.cpu().numpy()
                # Use the VAE's input shape as the expected shape (e.g., (128,128))
                from vae.utils import save_images
                save_images(original_ct, reconstructed_ct, self.output_dir, self.current_epoch,
                            expected_shape=tuple(self.vae.input_shape))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.lr)



# -----------------------------
# GAN Components
# -----------------------------
class Generator(nn.Module):
    """Modified U-Net without upstream"""
    def __init__(self, latent_dim, output_size=32):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8* 8),
            nn.ReLU(inplace=True)
        )

        self.dec1 = self.conv_block(512, 256) # 8 -> 16
        self.dec2 = self.conv_block(256, 128) # 16 -> 32
        self.dec3 = self.conv_block(128, 64) # 32 -> 64
        self.dec4 = self.conv_block(64,32) # 64 -> 128
        self.dec4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return torch.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
             nn.ZeroPad2d(2),
             nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False), # 128 -> 64
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 64 -> 32
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 32 -> 16
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), # 16 -> 8
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), # 8 -> 4
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False), # 4 -> 1
         )

    def forward(self, x):
        x = x.to(torch.float32)
        # Remove singleton dimensions from output
        return self.model(x).squeeze()

class GAN_Lightning(pl.LightningModule):
    def __init__(self, boson_sampler_params, gen_lr, disc_lr, latent_dim, output_size=64, output_dir="gan_images",
                 pretrain_gen_epochs=250, pretrain_disc_epochs=50):
        super().__init__()
        self.save_hyperparameters(ignore=["output_dir"])
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.latent_dim = latent_dim
        self.output_dir = output_dir

        # Pretraining epochs (generator, then discriminator)
        self.pretrain_gen_epochs = pretrain_gen_epochs
        self.pretrain_disc_epochs = pretrain_disc_epochs

        self.generator = Generator(latent_dim, output_size=output_size)
        self.discriminator = Discriminator()

        # Use PTGenerator for latent sampling if provided; otherwise, fallback to Gaussian noise.
        self.latent_space = PTGenerator(**boson_sampler_params) if boson_sampler_params is not None else None

        # Disable automatic optimization for manual control with multiple optimizers.
        self.automatic_optimization = False

        # These will be set in on_train_start
        self.scaler_gen = None
        self.scaler_disc = None

    def on_train_start(self):
        # Initialize AMP GradScalers if running on CUDA.
        if self.device.type == "cuda":
            self.scaler_gen = torch.cuda.amp.GradScaler()
            self.scaler_disc = torch.cuda.amp.GradScaler()
        else:
            self.scaler_gen = None
            self.scaler_disc = None

    def forward(self, z):
        return self.generator(z)

    def sample_latent(self, batch_size):
        if self.latent_space is not None:
            latent = self.latent_space.generate(batch_size).to(self.device)
        else:
            latent = torch.randn(batch_size, self.latent_dim, device=self.device)
        return latent

    def training_step(self, batch, batch_idx):
        opt_disc, opt_gen = self.optimizers()
        real_images = batch[1].to(self.device)
        # Downscale real images to 128x128 and normalize.
        real_downscaled = F.interpolate(real_images, size=(128,128), mode='bilinear')
        real_downscaled = (real_downscaled - real_downscaled.min()) / (
            real_downscaled.max() - real_downscaled.min() + 1e-8)
        batch_size = real_images.size(0)

        # -------------------------
        # Phase 1: Generator Pre-training
        # -------------------------
        if self.current_epoch < self.pretrain_gen_epochs:
            if self.current_epoch == 0 and batch_idx == 0:
                print("Starting generator pre-training phase.")
            # Freeze discriminator
            for p in self.discriminator.parameters():
                p.requires_grad = False
            latent = self.sample_latent(batch_size)
            if self.scaler_gen:
                with torch.cuda.amp.autocast():
                    generated_ct = self.generator(latent)
                    fake_output_for_gen = self.discriminator(generated_ct)
                    # Use BCE with logits in pre-training with lower adversarial weight
                    adversarial_loss = F.binary_cross_entropy_with_logits(
                        fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                    gen_loss = F.l1_loss(generated_ct, real_downscaled) + 0.1 * adversarial_loss
                opt_gen.zero_grad()
                self.scaler_gen.scale(gen_loss).backward()
                self.scaler_gen.step(opt_gen)
                self.scaler_gen.update()
            else:
                generated_ct = self.generator(latent)
                fake_output_for_gen = self.discriminator(generated_ct)
                adversarial_loss = F.binary_cross_entropy_with_logits(
                    fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                gen_loss = F.l1_loss(generated_ct, real_downscaled) + 0.1 * adversarial_loss
                opt_gen.zero_grad()
                gen_loss.backward()
                opt_gen.step()
            self.log("gen_loss", gen_loss, prog_bar=True)
            return {"gen_loss": gen_loss}

        # -------------------------
        # Phase 2: Discriminator Pre-training
        # -------------------------
        elif self.current_epoch < (self.pretrain_gen_epochs + self.pretrain_disc_epochs):
            if self.current_epoch == self.pretrain_gen_epochs and batch_idx == 0:
                print("Starting discriminator pre-training phase.")
            # Unfreeze discriminator
            for p in self.discriminator.parameters():
                p.requires_grad = True
            # Optionally drop discriminator LR at a chosen sub-epoch (e.g. at pretrain_gen_epochs + 25)
            if self.current_epoch == (self.pretrain_gen_epochs + 25) and batch_idx == 0:
                for g in opt_disc.param_groups:
                    g["lr"] = self.disc_lr * 0.1
            latent = self.sample_latent(batch_size)
            fake_images = self.generator(latent).detach()
            if self.scaler_disc:
                with torch.cuda.amp.autocast():
                    real_output = self.discriminator(real_images)
                    fake_output = self.discriminator(fake_images)
                    disc_loss = (dice_loss(real_output, torch.ones_like(real_output)) +
                                 dice_loss(fake_output, torch.zeros_like(fake_output)))
                opt_disc.zero_grad()
                self.scaler_disc.scale(disc_loss).backward()
                self.scaler_disc.step(opt_disc)
                self.scaler_disc.update()
            else:
                real_output = self.discriminator(real_images)
                fake_output = self.discriminator(fake_images)
                disc_loss = (dice_loss(real_output, torch.ones_like(real_output)) +
                             dice_loss(fake_output, torch.zeros_like(fake_output)))
                opt_disc.zero_grad()
                disc_loss.backward()
                opt_disc.step()
            self.log("disc_loss", disc_loss, prog_bar=True)
            return {"disc_loss": disc_loss}

        # -------------------------
        # Phase 3: Main Training (Joint Updates)
        # -------------------------
        else:
            if self.current_epoch == (self.pretrain_gen_epochs + self.pretrain_disc_epochs) and batch_idx == 0:
                print("Starting main training phase (joint updates).")
            # --- Discriminator Update ---
            for p in self.discriminator.parameters():
                p.requires_grad = True
            latent_disc = self.sample_latent(batch_size)
            fake_images = self.generator(latent_disc).detach()
            if self.scaler_disc:
                with torch.cuda.amp.autocast():
                    real_output = self.discriminator(real_images)
                    fake_output = self.discriminator(fake_images)
                    disc_loss = (dice_loss(real_output, torch.ones_like(real_output)) +
                                 dice_loss(fake_output, torch.zeros_like(fake_output)))
                opt_disc.zero_grad()
                self.scaler_disc.scale(disc_loss).backward()
                self.scaler_disc.step(opt_disc)
                self.scaler_disc.update()
            else:
                real_output = self.discriminator(real_images)
                fake_output = self.discriminator(fake_images)
                disc_loss = (dice_loss(real_output, torch.ones_like(real_output)) +
                             dice_loss(fake_output, torch.zeros_like(fake_output)))
                opt_disc.zero_grad()
                disc_loss.backward()
                opt_disc.step()

            # --- Generator Update ---
            # Freeze discriminator for generator update
            for p in self.discriminator.parameters():
                p.requires_grad = False
            latent_gen = self.sample_latent(batch_size)
            if self.scaler_gen:
                with torch.cuda.amp.autocast():
                    generated_ct = self.generator(latent_gen)
                    fake_output_for_gen = self.discriminator(generated_ct)
                    adversarial_loss = dice_loss(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                    gen_loss = F.l1_loss(generated_ct, real_downscaled) + 0.2 * adversarial_loss
                opt_gen.zero_grad()
                self.scaler_gen.scale(gen_loss).backward()
                self.scaler_gen.step(opt_gen)
                self.scaler_gen.update()
            else:
                generated_ct = self.generator(latent_gen)
                fake_output_for_gen = self.discriminator(generated_ct)
                adversarial_loss = dice_loss(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                gen_loss = F.l1_loss(generated_ct, real_downscaled) + 0.2 * adversarial_loss
                opt_gen.zero_grad()
                gen_loss.backward()
                opt_gen.step()

            self.log("disc_loss", disc_loss, prog_bar=True)
            self.log("gen_loss", gen_loss, prog_bar=True)
            # Optionally, print summary statistics for debugging:
            # print(generated_ct.min(), generated_ct.max(), generated_ct.mean())
            return {"disc_loss": disc_loss, "gen_loss": gen_loss}

    def validation_step(self, batch, batch_idx):
        pet_images, ct_images = batch
        ct_images = ct_images.to(self.device)
        batch_size = ct_images.size(0)

        latent = self.sample_latent(batch_size)
        with torch.no_grad():
            generated_ct = self.generator(latent)

        data_real_downscaled = F.interpolate(ct_images, size=(64, 64), mode='nearest')
        data_real_downscaled = (data_real_downscaled - data_real_downscaled.min()) / (
            data_real_downscaled.max() - data_real_downscaled.min() + 1e-8)
        generated_ct = (generated_ct - generated_ct.min()) / (generated_ct.max() - generated_ct.min() + 1e-8)

        # Save images (for the first batch only)
        if batch_idx == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            for i in range(batch_size):
                real_img = data_real_downscaled[i].cpu().numpy().squeeze()
                gen_img = generated_ct[i].cpu().numpy().squeeze()
                plt.imsave(os.path.join(self.output_dir, f'epoch_{self.current_epoch}_original_{i}.png'),
                           real_img, cmap='gray')
                plt.imsave(os.path.join(self.output_dir, f'epoch_{self.current_epoch}_reconstructed_{i}.png'),
                           gen_img, cmap='gray')

        with torch.no_grad():
            fake_output = self.discriminator(generated_ct)
            adversarial_loss = dice_loss(fake_output, torch.ones_like(fake_output))
            gen_loss = F.l1_loss(generated_ct, data_real_downscaled) + 0.2 * adversarial_loss
        self.log("val_gen_loss", gen_loss, prog_bar=True)
        return {"val_gen_loss": gen_loss}

    def configure_optimizers(self):
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr)
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr)
        return [disc_optimizer, gen_optimizer]

# -----------------------------------------
# Transformer Generator Model Definition
# -----------------------------------------
class TransformerGenerator(nn.Module):
    def __init__(self, input_dim, latent_dim, embed_dim, num_layers, boson_sampler_params=None):
        """
        Args:
            input_dim (int): Flattened CT image dimension (e.g. 128*128 = 16384).
            latent_dim (int): Size of the discrete latent space (number of tokens).
            embed_dim (int): Embedding dimension for both PET features and latent tokens.
            num_layers (int): Number of transformer decoder layers.
            boson_sampler_params (dict, optional): Parameters for initializing the boson sampler.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Learnable embedding for discrete latent tokens
        self.latent_embedding = nn.Embedding(latent_dim, embed_dim)
        # A simple PET encoder: encodes a PET scan into an embedding vector
        self.pet_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )
        # Positional embedding for the latent token
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Transformer decoder (using PyTorch’s built-in modules)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Final projection from transformer output to a flattened CT image
        self.fc_out = nn.Linear(embed_dim, input_dim)

        # Instantiate boson sampler if parameters are provided
        if boson_sampler_params is not None:
            self.boson_sampler = PTGenerator(**boson_sampler_params)
        else:
            self.boson_sampler = None

    def forward(self, pet_image, latent_token):
        """
        Args:
            pet_image (tensor): PET scans of shape (B, 1, H, W).
            latent_token (tensor): Discrete latent token indices of shape (B,).
        Returns:
            ct_flat (tensor): Generated CT images as flattened vectors (B, input_dim).
        """
        batch_size = pet_image.size(0)
        # Encode PET image to a feature vector (memory for transformer)
        pet_feat = self.pet_encoder(pet_image)  # shape: (B, embed_dim)
        memory = pet_feat.unsqueeze(0)           # shape: (1, B, embed_dim)
        # Get embedding for latent tokens and add positional encoding
        token_embed = self.latent_embedding(latent_token)  # shape: (B, embed_dim)
        tgt = token_embed.unsqueeze(0) + self.pos_embedding  # shape: (1, B, embed_dim)
        # Transformer decoder: use the PET features as memory to condition generation
        out = self.transformer_decoder(tgt, memory)  # shape: (1, B, embed_dim)
        out = out.squeeze(0)  # shape: (B, embed_dim)
        # Project transformer output to the flattened CT image space
        ct_flat = self.fc_out(out)  # shape: (B, input_dim)
        return ct_flat

# -----------------------------------------
# Transformer Lightning Module Definition
# -----------------------------------------
class Transformer_Lightning(pl.LightningModule):
    def __init__(self, boson_params_to_use, lr, latent_dim, embed_dim, num_layers, output_dir, input_shape):
        """
        Args:
            boson_params_to_use (dict): Parameters for the boson sampler.
            lr (float): Learning rate.
            latent_dim (int): Discrete latent space size.
            embed_dim (int): Embedding dimension for the transformer.
            num_layers (int): Number of transformer decoder layers.
            output_dir (str): Directory to save generated images.
            input_shape (tuple): Expected shape of CT images (e.g., (128, 128)).
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.output_dir = output_dir
        self.input_shape = input_shape  # e.g. (128, 128)
        self.input_dim = int(np.prod(input_shape))  # e.g. 128*128 = 16384
        self.latent_dim = latent_dim

        self.model = TransformerGenerator(
            input_dim=self.input_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            boson_sampler_params=boson_params_to_use
        )
        self.loss_fn = nn.L1Loss()

    def forward(self, pet_image, latent_token):
        return self.model(pet_image, latent_token)

    def sample_latent_token(self, batch_size):
        """
        Samples a batch of latent tokens by combining the boson sampler’s output with a simulated transformer distribution.
        """
        latent_size = self.hparams.latent_dim  # number of discrete tokens

        if self.model.boson_sampler is not None:
            boson_sample = self.model.boson_sampler.generate(1)  # assume shape (1, latent_dim)
            boson_distribution_tensor = F.softmax(boson_sample, dim=-1).squeeze(0)  # shape: (latent_dim,)
            boson_distribution = boson_distribution_tensor.detach().cpu().numpy()
        else:
            boson_distribution = np.ones(latent_size) / latent_size

        transformer_probs = np.random.dirichlet(np.ones(latent_size))
        alpha = 0.6  # 60% weight for transformer predictions, 40% for boson sampler output.
        combined_probs = alpha * transformer_probs + (1 - alpha) * boson_distribution
        combined_probs /= combined_probs.sum()  # normalize

        tokens = np.random.choice(latent_size, size=batch_size, p=combined_probs)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        return tokens

    def training_step(self, batch, batch_idx):
        pet_image, ct_image = batch
        batch_size = pet_image.size(0)
        # Adjust target CT image: assume ct_image shape is (B, 1, H, W)
        # Use only one channel (e.g. the first) and interpolate to self.input_shape.
        target = ct_image[:, 0, :, :].to(self.device)
        target = F.interpolate(target.unsqueeze(1),
                               size=self.input_shape,
                               mode='bilinear',
                               align_corners=False).squeeze(1)
        target = target.view(batch_size, -1)  # Flatten to (B, prod(input_shape))

        latent_tokens = self.sample_latent_token(batch_size)
        generated_ct_flat = self(pet_image, latent_tokens)
        loss = self.loss_fn(generated_ct_flat, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pet_image, ct_image = batch
        batch_size = pet_image.size(0)
        target = ct_image[:, 0, :, :].to(self.device)
        target = F.interpolate(target.unsqueeze(1),
                               size=self.input_shape,
                               mode='bilinear',
                               align_corners=False).squeeze(1)
        target = target.view(batch_size, -1)

        latent_tokens = self.sample_latent_token(batch_size)
        generated_ct_flat = self(pet_image, latent_tokens)
        loss = self.loss_fn(generated_ct_flat, target)
        self.log("val_loss", loss, prog_bar=True)

        if batch_idx == 0:
            generated_ct = generated_ct_flat.view(batch_size, *self.input_shape).detach().cpu().numpy()
            target_images = target.view(batch_size, *self.input_shape).detach().cpu().numpy()
            os.makedirs(self.output_dir, exist_ok=True)
            for i in range(min(batch_size, 5)):
                plt.imsave(os.path.join(self.output_dir, f'epoch_{self.current_epoch}_generated_{i}.png'),
                           generated_ct[i], cmap='gray')
                plt.imsave(os.path.join(self.output_dir, f'epoch_{self.current_epoch}_target_{i}.png'),
                           target_images[i], cmap='gray')
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
