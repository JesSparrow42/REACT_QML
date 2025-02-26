# train.py
import sys
# Preprocess sys.argv to remove extra dashes for Hydra overrides.
new_args = []
for arg in sys.argv[1:]:
    if arg.startswith("--hyperparameters."):
        new_args.append(arg.lstrip("-"))
    else:
        new_args.append(arg)
sys.argv = [sys.argv[0]] + new_args

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf

# Assume you have a MedicalDataModule defined in your project
from data import MedicalDataModule, QM9DataModule
from model import GAN_Lightning, VAE_Lightning, DiffusionLightning, UNetLightning, GraphVAE_Lightning
#from transformer import Transformer_Lightning
from utils import load_separate_checkpoints

def main():
    # Hard-coded parameters and paths
    db_path = r"C:\Users\Jonat\Desktop\Datasets\qm9.db"
    split_path = r"C:\Users\Jonat\Desktop\Datasets\randomsplits_110k_10k_rest.json"
    batch_size = 32
    latent_features = 16
    lr = 1e-3
    epochs = 50
    cache_dir = "cache"

    # Create and set up the data module (this loads/caches the dataset)
    data_module = QM9DataModule(db_path=db_path, split_path=split_path, batch_size=batch_size, cache_dir=cache_dir)
    data_module.setup()

    # Compute the global maximum number of nodes from the training dataset
    global_max = max([data.num_nodes for data in data_module.train_dataset.data_list])
    print("Using global_max =", global_max)
    # Save this value in the data module so that the dataloaders use it.
    data_module.global_max = global_max

    # Instantiate the Lightning model with the global max_nodes
    model = GraphVAE_Lightning(latent_features=latent_features, max_nodes=global_max, lr=lr)

    # Set up a progress bar callback (optional)
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # Create the Trainer
    trainer = Trainer(max_epochs=epochs, callbacks=[progress_bar], logger=True)
    trainer.fit(model, datamodule=data_module)
    print("Training finished.")
