# train.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import argparse

from vae.model import VariationalAutoencoder, VariationalInference, save_images

class VAE_Lightning(pl.LightningModule):
    def __init__(self, boson_params_to_use, lr, latent_features, output_dir):
        super().__init__()
        self.save_hyperparameters()  # Saves hyperparameters for easy access
        self.lr = lr
        self.output_dir = output_dir
        self.latent_features = latent_features

        self.vae = VariationalAutoencoder(
            input_shape=torch.Size([28*28]),
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
        # Use validation to save images each epoch using first batch
        if batch_idx == 0:
            x, _ = batch
            x = x.view(x.size(0), -1).to(self.device)
            with torch.no_grad():
                _, _, outputs = self.vi(self.vae, x)[:3]
                reconstructed = outputs["px"].probs.cpu().numpy()
                original = x.cpu().numpy()
                save_images(original, reconstructed, self.output_dir, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.lr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--boson_sampler_params",
        default="true",
        choices=["true", "false"],
        help="When 'true' we use the Boson Sampler. When 'false', we use Gaussian."
    )
    args = parser.parse_args()

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
    latent_features = 8
    output_dir = "vae_images"

    # Data loading
    train_data = MNIST(root=".", train=True, transform=ToTensor(), download=True)
    test_data = MNIST(root=".", train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize Lightning module
    model = VAE_Lightning(boson_params_to_use, lr, latent_features, output_dir)

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=20
    )

    # Train the model
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
