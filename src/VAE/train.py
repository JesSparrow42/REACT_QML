# train.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import argparse
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from vae.data import MedicalDataModule
from vae.model import VAE_Lightning

@hydra.main(config_path="../../", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Print the loaded configuration for confirmation
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Optionally set the seed for reproducibility
    torch.manual_seed(cfg.hyperparameters.seed)

    # If the config does not specify, we default to using the Boson sampler.
    boson_flag = cfg.hyperparameters.get("boson_sampler_params", "true")
    if boson_flag == "false":
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

    latent_features = 8
    output_dir = "vae_images"

    ct_folder = 'Rigshospitalet/data/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805'
    pet_folder = 'Rigshospitalet/data/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/1.000000-PET NAC-24000'
    data_module = MedicalDataModule(ct_folder, pet_folder, batch_size=cfg.hyperparameters.batch_size)
    # Initialize Lightning module
    model = VAE_Lightning(boson_params_to_use, cfg.hyperparameters.lr, latent_features, output_dir)

    # Initialize Trainer
    trainer = Trainer(
        default_root_dir="my_logs_dir",
        max_epochs=cfg.hyperparameters.n_epochs,
        #profiler="simple",
        logger=pl.loggers.WandbLogger(project="vae_sparrow"),
        gradient_clip_val=0.5  # adjust as needed
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
