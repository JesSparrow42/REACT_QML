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
from vae.utils import save_images, dice_loss, plot_molecule  # and any other helper functions as needed

### To do
# Parameter shift rule?
# Benchmarking - FID score/ KL - DONE(Maybe add MAE)
# QM9 - DONE

# -----------------------------
# VAE Components
# -----------------------------
class BosonPrior(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, boson_sampler, batch_size, latent_features, discriminator_iter=0, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.boson_sampler = boson_sampler
        self.batch_size = batch_size
        self.latent_features = latent_features
        self.discriminator_iter = discriminator_iter

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def rsample(self, sample_shape=torch.Size()):
        total_samples = (1 + self.discriminator_iter) * self.batch_size
        latent = self.boson_sampler.generate(total_samples).to(self.device)
        latent = torch.chunk(latent, 1 + self.discriminator_iter, dim=0)
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
                # Use the VAE's input shape as the expected shape (e.g., (128,128))
                from vae.utils import save_images
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

class GraphVAE(nn.Module):
    def __init__(self, latent_features: int, max_nodes: int, node_feature_dim: int = 1,
                 position_dim: int = 3, hidden_dim: int = 64):
        """
        A simple Graph VAE.
        """
        super().__init__()
        self.latent_features = latent_features
        self.max_nodes = max_nodes
        self.input_dim = node_feature_dim + position_dim  # e.g. 1+3 = 4
        # Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_features)
        self.fc_logvar = nn.Linear(hidden_dim, latent_features)
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * self.input_dim)
        )

    def encode(self, node_features, node_positions, mask):
        x = torch.cat([node_features, node_positions], dim=-1)  # (B, N, 4)
        B, N, _ = x.size()
        h = self.node_encoder(x.view(B * N, self.input_dim))
        h = h.view(B, N, -1)
        mask_expanded = mask.unsqueeze(-1)
        h = h * mask_expanded
        sum_h = h.sum(dim=1)
        counts = mask_expanded.sum(dim=1)
        pooled = sum_h / (counts + 1e-6)
        pooled = self.graph_encoder(pooled)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.size(0)
        out = self.decoder_fc(z)
        out = out.view(B, self.max_nodes, self.input_dim)
        return out

    def forward(self, node_features, node_positions, mask):
        mu, logvar = self.encode(node_features, node_positions, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

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

            from vae.utils import save_images
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
