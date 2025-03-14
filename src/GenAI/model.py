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
from utils import BosonPrior, ReparameterizedDiagonalGaussian, save_images, dice_loss, plot_molecule  # and any other helper functions as needed
from models.vae
from models.gan
from models.unet
from models.graphvae


### To do
# Parameter shift rule?
# Benchmarking - FID score/ KL - DONE(Maybe add MAE)
# QM9 - DONE

class VAE_Lightning(pl.LightningModule):
    def __init__(self, boson_params_to_use, lr, latent_features, output_dir_orig, output_dir_reco):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.output_dir_orig = output_dir_orig
        self.output_dir_reco = output_dir_reco
        self.latent_features = latent_features

        self.vae = VariationalAutoencoder(
            input_shape=torch.Size([128, 128]),
            latent_features=self.hparams.latent_features,
            boson_sampler_params=self.hparams.boson_params_to_use
        )
        self.vi = VariationalInference(beta=1.0)
        self._val_outputs = []

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
        self._val_outputs.append(loss.detach())

        if batch_idx == 0:
            print(f"Validation step at epoch {self.current_epoch}")
            with torch.no_grad():
                reconstructed_ct = outputs["px"].probs.cpu().numpy()
                original_ct = ct_image_target.cpu().numpy()
                # Use the input shape as the expected shape (e.g., (128,128))
                save_images(original_ct, reconstructed_ct, self.output_dir_orig, self.output_dir_reco, self.current_epoch,
                            expected_shape=tuple(self.vae.input_shape))
        return loss

    def on_validation_epoch_end(self):
        # Aggregate stored outputs (e.g. average validation loss)
        if self._val_outputs:
            avg_loss = torch.stack(self._val_outputs).mean().item()
            self._val_outputs.clear()
        else:
            avg_loss = 0.0

        fid_score = self.compute_fid()
        self.log("FID", fid_score, prog_bar=True)
        self.log("avg_val_loss", avg_loss, prog_bar=True)

        metrics = {
            "epoch": self.current_epoch,
            "avg_val_loss": avg_loss,
            "FID": fid_score,
            "lr": self.hparams.lr
        }
        self.write_metrics_to_csv(metrics)

    def compute_fid(self):
        try:
            from evaluate import load_images_for_epoch
            from torchmetrics.image.fid import FrechetInceptionDistance

            real_images = load_images_for_epoch(self.output_dir_orig, self.current_epoch, num_images=32)
            fake_images = load_images_for_epoch(self.output_dir_reco, self.current_epoch, num_images=32)

            fid_metric = FrechetInceptionDistance(feature=64)
            fid_metric.update(real_images, real=True)
            fid_metric.update(fake_images, real=False)
            fid_score = fid_metric.compute()
            return fid_score.item()
        except Exception as e:
            print(f"Error computing FID: {e}")
            return 0.0

    def write_metrics_to_csv(self, metrics: dict):
        import csv, os
        from datetime import datetime

        date_str = datetime.now().strftime("%Y%m%d")
        boson_used = "boson" if self.hparams.boson_params_to_use is not None else "noboson"
        filename = f"vae_{date_str}_lr{self.hparams.lr}_{boson_used}.csv"
        os.makedirs("metrics", exist_ok=True)
        filepath = os.path.join("metrics", filename)
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.lr)


###############################
# Graph VAE Model and Lightning Module
###############################

class GraphVAE_Lightning(pl.LightningModule):

    def __init__(self, latent_features: int, max_nodes: int, lr: float = 1e-3,
                 output_dir_orig: str = "orig_molecules", output_dir_reco: str = "reco_molecules"):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphVAE(latent_features=latent_features, max_nodes=max_nodes)
        self.lr = lr

        self.output_dir_orig = output_dir_orig
        self.output_dir_reco = output_dir_reco

        os.makedirs(self.output_dir_orig, exist_ok=True)
        os.makedirs(self.output_dir_reco, exist_ok=True)

    def forward(self, batch):
        recon, mu, logvar = self.model(batch["node_features"], batch["node_positions"], batch["mask"])
        return recon, mu, logvar


    def compute_loss(self, recon, batch, mu, logvar):
        recon_feat = recon[..., :1]
        recon_pos = recon[..., 1:]
        mask = batch["mask"].unsqueeze(-1)
        loss_feat = F.mse_loss(recon_feat * mask, batch["node_features"] * mask, reduction="sum")
        loss_pos = F.mse_loss(recon_pos * mask, batch["node_positions"] * mask, reduction="sum")
        recon_loss = (loss_feat + loss_pos) / mask.sum()
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon_loss + kl_loss, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        recon, mu, logvar = self.forward(batch)
        loss, recon_loss, kl_loss = self.compute_loss(recon, batch, mu, logvar)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def compute_mae(self, recon, batch):
        recon_feat = recon[..., :1]
        recon_pos = recon[..., 1:]
        mask = batch["mask"].unsqueeze(-1)
        mae_feat = torch.abs(recon_feat * mask - batch["node_features"] * mask).sum() / mask.sum()
        mae_pos = torch.abs(recon_pos * mask - batch["node_positions"] * mask).sum() / mask.sum()
        return mae_feat + mae_pos

    def validation_step(self, batch, batch_idx):
        recon, mu, logvar = self.forward(batch)
        loss, _, _ = self.compute_loss(recon, batch, mu, logvar)
        mae = self.compute_mae(recon, batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)

        if batch_idx == 0:
            print(f"Validation step at epoch {self.current_epoch}")
            with torch.no_grad():
                mask = batch["mask"][0].bool()
                orig_positions = batch["node_positions"][0][mask].detach().cpu().numpy()
                orig_features = batch["node_features"][0][mask].detach().cpu().numpy()

                recon_positions = recon[0, mask, 1:].detach().cpu().numpy()
                recon_features = recon[0, mask, :1].detach().cpu().numpy()

                orig_filepath = os.path.join(self.output_dir_orig, f"original_molecule_epoch_{self.current_epoch}.png")
                reco_filepath = os.path.join(self.output_dir_reco, f"generated_molecule_epoch_{self.current_epoch}.png")

                try:
                    plot_molecule(orig_positions, orig_features, orig_filepath)
                    plot_molecule(recon_positions, recon_features, reco_filepath)

                    print(f"Saved original molecule to: {orig_filepath}")
                    print(f"Saved reconstructed molecule to: {reco_filepath}")
                except Exception as e:
                    print(f"Error saving molecule plot: {e}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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


class UNetLightning(pl.LightningModule):
    """
    A LightningModule using U-Net to predict CT images from PET.
    This version resizes CT targets to match the UNet output size.
    """
    def __init__(self, lr, output_dir):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.output_dir = output_dir
        self.model = UNetGenerator(in_channels=1, out_channels=1, base_filters=64)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pet_image, ct_image = batch
        pred_ct = self.model(pet_image)
        # Resize CT images to match the predicted output size
        ct_image_resized = F.interpolate(ct_image, size=pred_ct.shape[-2:], mode='bilinear', align_corners=False)
        loss = self.loss_fn(pred_ct, ct_image_resized)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pet_image, ct_image = batch
        pred_ct = self.model(pet_image)
        # Resize CT images to match the predicted output size
        ct_image_resized = F.interpolate(ct_image, size=pred_ct.shape[-2:], mode='bilinear', align_corners=False)
        loss = self.loss_fn(pred_ct, ct_image_resized)
        self.log("val_loss", loss, prog_bar=True)

        # Save images for visualization on the first batch
        if batch_idx == 0:
            # Detach and move to CPU
            generated_ct = pred_ct.detach().cpu().numpy()
            target_images = ct_image_resized.detach().cpu().numpy()

            # Squeeze the channel dimension if present (e.g. shape: (B, 1, H, W) -> (B, H, W))
            if generated_ct.shape[1] == 1:
                generated_ct = generated_ct.squeeze(1)
            if target_images.shape[1] == 1:
                target_images = target_images.squeeze(1)

            save_images(
                target_images,             # ground truth
                generated_ct,              # predicted
                self.output_dir_orig,
                self.output_dir_reco,
                self.current_epoch,
                expected_shape=tuple(pred_ct.shape[-2:])
            )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# --- Diffusion Lightning Module ---
class DiffusionLightning(pl.LightningModule):
    def __init__(self, boson_params_to_use, lr, latent_features, output_dir_orig, output_dir_reco, discriminator_iter=0):
        """
        Required parameters:
          - boson_params_to_use: dictionary for boson sampler (or None for Gaussian)
          - lr: learning rate
          - latent_features: number of latent features for the prior
          - output_dir: directory to save images
        Other diffusion parameters (image size, timesteps, beta schedule) are defined here.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.output_dir_orig = output_dir_orig
        self.output_dir_reco = output_dir_reco
        self.discriminator_iter = discriminator_iter

        # Set up boson sampler if provided.
        self.boson_sampler = None
        if boson_params_to_use is not None:
            self.boson_sampler = PTGenerator(**boson_params_to_use)
        self.latent_features = latent_features

        # Hard-coded settings:
        self.image_size = (128, 128)
        self.timesteps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        # We'll condition on PET so:
        #   - The U-Net input is [PET, noisy CT] with 2 channels.
        self.in_channels = 2
        self.out_channels = 1
        self.base_filters = 64

        # Build U-Net with in_channels=2
        self.unet = UNetGenerator(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  base_filters=self.base_filters)

        # Noise schedule
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('alpha_bars', alpha_bars)

        # Prior parameters for Gaussian prior
        prior_init = torch.zeros(1, 2 * self.latent_features)
        self.register_buffer("prior_params", prior_init)

        required_dim = self.image_size[0] * self.image_size[1] * self.out_channels
        if self.latent_features != required_dim:
            self.noise_projector = nn.Linear(self.latent_features, required_dim)
        else:
            self.noise_projector = None

        self.loss_fn = nn.MSELoss()
        self._val_outputs = []

    def prior(self, batch_size: int) -> Distribution:
        if self.boson_sampler is not None:
            return BosonPrior(boson_sampler=self.boson_sampler,
                              batch_size=batch_size,
                              latent_features=self.latent_features)
        else:
            prior_params = self.prior_params.expand(batch_size, -1)
            mu, log_sigma = prior_params.chunk(2, dim=-1)
            return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def q_sample(self, ct_clean, t):
        """
        Add noise to the clean CT image (ct_clean) at timestep t.
        Returns:
          x_t: Noisy CT image,
          noise: Noise that was added,
          sqrt_alpha_bar, sqrt_one_minus_alpha_bar: scaling factors.
        """
        dist = self.prior(batch_size=ct_clean.size(0))
        noise = dist.rsample()  # This returns a tuple when discriminator_iter > 0
        # Extract the first tensor from the tuple if necessary
        if isinstance(noise, (tuple, list)):
            noise = noise[0]

        if self.noise_projector is not None:
            noise = self.noise_projector(noise)
        noise = noise.view(ct_clean.size(0), ct_clean.size(1), self.image_size[0], self.image_size[1])
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * ct_clean + sqrt_one_minus_alpha_bar * noise
        return x_t, noise, sqrt_alpha_bar, sqrt_one_minus_alpha_bar



    def forward(self, pet_image, ct_clean, t):
        """
        For inference you can provide a PET image and a CT image (or noise schedule t) 
        to predict noise. In training, we add noise to the CT.
        """
        # Add noise to CT
        x_t, noise, _, _ = self.q_sample(ct_clean, t)
        # Concatenate conditioning PET image with noisy CT along channel dimension.
        # Assume pet_image has shape (B, 1, H, W) and ct_clean (or x_t) is (B, 1, H, W)
        cond_input = torch.cat([pet_image, x_t], dim=1)  # shape: (B, 2, H, W)
        noise_pred = self.unet(cond_input)
        return noise_pred, noise

    def training_step(self, batch, batch_idx):
        # batch: (pet_image, ct_image)
        pet_image, ct_image = batch
        # Resize both images to (128,128)
        pet_image = F.interpolate(pet_image, size=self.image_size, mode='bilinear', align_corners=False)
        ct_image = F.interpolate(ct_image, size=self.image_size, mode='bilinear', align_corners=False)
        B = ct_image.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=self.device, dtype=torch.long)
        x_t, noise, _, _ = self.q_sample(ct_image, t)
        cond_input = torch.cat([pet_image, x_t], dim=1)
        noise_pred = self.unet(cond_input)
        loss = self.loss_fn(noise_pred, noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pet_image, ct_image = batch
        pet_image = F.interpolate(pet_image, size=self.image_size, mode='bilinear', align_corners=False)
        ct_image = F.interpolate(ct_image, size=self.image_size, mode='bilinear', align_corners=False)
        B = ct_image.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=self.device, dtype=torch.long)
        x_t, noise, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.q_sample(ct_image, t)
        cond_input = torch.cat([pet_image, x_t], dim=1)
        noise_pred = self.unet(cond_input)
        loss = self.loss_fn(noise_pred, noise)
        self.log("val_loss", loss, prog_bar=True)
        self._val_outputs.append(loss.detach())

        # Reconstruct the CT image from predicted noise:
        x0_pred = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar

        if batch_idx == 0:
            x0_pred_np = x0_pred.detach().cpu().numpy()
            ct_clean_np = ct_image.detach().cpu().numpy()
            if x0_pred_np.shape[1] == 1:
                x0_pred_np = x0_pred_np.squeeze(1)
            if ct_clean_np.shape[1] == 1:
                ct_clean_np = ct_clean_np.squeeze(1)
            save_images(ct_clean_np, x0_pred_np, self.output_dir_orig, self.output_dir_reco, self.current_epoch,
                        expected_shape=tuple(x_t.shape[-2:]))
        return loss

    def on_validation_epoch_end(self):
        if self._val_outputs:
            avg_loss = torch.stack(self._val_outputs).mean().item()
            self._val_outputs.clear()
        else:
            avg_loss = 0.0
        
        fid_score = self.compute_fid()
        self.log("FID", fid_score, prog_bar=True)
        self.log("avg_val_loss", avg_loss, prog_bar=True)

        metrics = {
            "epoch": self.current_epoch,
            "avg_val_loss": avg_loss,
            "FID": fid_score,
            "lr": self.hparams.lr
        }
        self.write_metrics_to_csv(metrics)

    def compute_fid(self):
        try:
            from evaluate import load_images_for_epoch
            from torchmetrics.image.fid import FrechetInceptionDistance

            real_images = load_images_for_epoch(self.output_dir_orig, self.current_epoch, num_images=32)
            fake_images = load_images_for_epoch(self.output_dir_reco, self.current_epoch, num_images=32)
            fid_metric = FrechetInceptionDistance(feature=64)
            fid_metric.update(real_images, real=True)
            fid_metric.update(fake_images, real=False)
            fid_score = fid_metric.compute()
            return fid_score.item()
        except Exception as e:
            print(f"Error computing FID: {e}")
            return 0.0

    def write_metrics_to_csv(self, metrics: dict):
        import csv, os
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        boson_used = "boson" if self.hparams.get("boson_params_to_use", None) is not None else "noboson"
        filename = f"diffusion_{date_str}_lr{self.hparams.lr}_{boson_used}.csv"
        os.makedirs("metrics", exist_ok=True)
        filepath = os.path.join("metrics", filename)
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
