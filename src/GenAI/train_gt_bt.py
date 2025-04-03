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
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf

# Assume you have a MedicalDataModule defined in your project
from data import MedicalDataModule, QM9DataModule, AtomsData
from model import GAN_Lightning, VAE_Lightning, DiffusionLightning, UNetLightning, GraphVAE_Lightning
from BosonSamplerWrapper import BosonLatentGenerator, BosonSamplerTorch
#from transformer import Transformer_Lightning
from utils import load_separate_checkpoints

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.hyperparameters.seed)

    latent_dim = 14

    boson_flag = cfg.hyperparameters.get("boson_sampler_params", "true")
    if boson_flag == "false":
        boson_params_to_use = None
        print(">> Using standard Gaussian prior.")
    else:
        boson_params_to_use = BosonSamplerTorch(
            m=latent_dim,
            num_sources=latent_dim // 2,
            num_loops=2,
            input_loss=0,
            coupling_efficiency=1,
            detector_inefficiency=1,
            multi_photon_prob=0,
            mu=1,
            temporal_mismatch=0,
            spectral_mismatch=0,
            arrival_time_jitter=0,
            bs_loss=1,
            bs_jitter=0,
            phase_noise_std=0,
            systematic_phase_offset=0,
            mode_loss=np.ones(latent_dim),
            dark_count_rate=0,
            use_advanced_nonlinearity=False,
            alpha_nl=0,
            beta_nl=0
        )
        print(">> Using Boson Sampler as prior.")

    # Select the datamodule based on config.
    # Set "datamodule" to "qm9" or "medical" in your config file under hyperparameters.
    datamodule_type = cfg.hyperparameters.get("datamodule", "medical").lower()
    if datamodule_type == "qm9":
        print(">> Using QM9 DataModule.")
        data_module = QM9DataModule(
            db_path=cfg.hyperparameters.qm9_db_path,
            split_path=cfg.hyperparameters.qm9_split_path,
            batch_size=cfg.hyperparameters.batch_size,
            cache_dir=cfg.hyperparameters.qm9_cache_dir
        )
        data_module.setup()
        # Compute and assign the global maximum number of nodes
        global_max = max([data.num_nodes for data in data_module.train_dataset.data_list])
        print("Using global_max =", global_max)
        data_module.global_max = global_max
    elif datamodule_type == "medical":
        print(">> Using Medical DataModule.")
        data_module = MedicalDataModule(
            ct_folder=cfg.hyperparameters.ct_folder,
            pet_folder=cfg.hyperparameters.pet_folder,
            batch_size=cfg.hyperparameters.batch_size
        )
    else:
        raise ValueError(f"Unknown datamodule type: {datamodule_type}")

    # latent_features = cfg.hyperparameters.get("latent_features", 8)
    latent_features = latent_dim
    model_type = cfg.hyperparameters.get("model_type", "vae").lower()

    # Instantiate model and Trainer based on model_type
    if model_type == "gan":
        print(">> Training GAN model.")
        model = GAN_Lightning(
            boson_sampler_params=boson_params_to_use,
            gen_lr=cfg.hyperparameters.gen_lr,
            disc_lr=cfg.hyperparameters.disc_lr,
            latent_dim=latent_features,
            output_size=cfg.hyperparameters.get("gan_output_size", 64),
            output_dir=cfg.hyperparameters.get("output_dir", "vae_images"),
            pretrain_gen_epochs=cfg.hyperparameters.get("pretrain_gen_epochs", 250),
            pretrain_disc_epochs=cfg.hyperparameters.get("pretrain_disc_epochs", 50)
        )
        # Optionally load checkpoints if provided
        gen_ckpt_path = cfg.hyperparameters.get("gen_checkpoint_path", None)
        disc_ckpt_path = cfg.hyperparameters.get("disc_checkpoint_path", None)
        if gen_ckpt_path and disc_ckpt_path:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            start_epoch_gen, start_epoch_disc = load_separate_checkpoints(gen_ckpt_path, disc_ckpt_path, device)
            print(f"Resuming training: generator from epoch {start_epoch_gen}, discriminator from epoch {start_epoch_disc}")
        else:
            print("No checkpoint paths provided; starting training from scratch.")
        wandb_project = "gan_sparrow"
        trainer = Trainer(
            default_root_dir="my_logs_dir",
            max_epochs=cfg.hyperparameters.n_epochs,
            logger=pl.loggers.WandbLogger(project=wandb_project)
        )

    elif model_type == "vae":
        print(">> Training VAE model.")
        model = VAE_Lightning(
            boson_params_to_use,
            cfg.hyperparameters.lr,
            latent_features,
            cfg.hyperparameters.get("output_dir_orig", "vae_images_orig"),
            cfg.hyperparameters.get("output_dir_reco", "vae_images_reco")
        )
        wandb_project = "vae_sparrow"
        trainer = Trainer(
            default_root_dir="my_logs_dir",
            max_epochs=cfg.hyperparameters.n_epochs,
            logger=pl.loggers.WandbLogger(project=wandb_project),
            gradient_clip_val=0.5
        )

    elif model_type == "transformer":
        print(">> Training Transformer model.")
        model = UNetLightning(
            lr=cfg.hyperparameters.lr,
            output_dir=cfg.hyperparameters.get("output_dir", "vae_images")
        )
        wandb_project = "transformer_sparrow"
        trainer = Trainer(
            default_root_dir="my_logs_dir",
            max_epochs=cfg.hyperparameters.n_epochs,
            logger=pl.loggers.WandbLogger(project=wandb_project),
            gradient_clip_val=0.5
        )

    elif model_type == "diffusion":
        print(">> Training Diffusion model.")
        model = DiffusionLightning(
            boson_params_to_use=boson_params_to_use,
            lr=cfg.hyperparameters.lr,
            latent_features=latent_features,
            output_dir_orig=cfg.hyperparameters.get("output_dir_orig", "results/diffusion_orig"),
            output_dir_reco=cfg.hyperparameters.get("output_dir_reco", "results/diffusion_reco"),
            discriminator_iter=cfg.hyperparameters.get("discriminator_iter", 0)
        )
        wandb_project = "diffusion_sparrow"
        trainer = Trainer(
            default_root_dir="my_logs_dir",
            max_epochs=cfg.hyperparameters.n_epochs,
            logger=pl.loggers.WandbLogger(project=wandb_project),
            gradient_clip_val=0.5
        )



    elif model_type == "graphvae":
        print(">> Training GraphVAE model.")
        # Ensure that global_max is computed if not already (e.g. if using a non-QM9 module)
        if datamodule_type != "qm9":
            global_max = max([data.num_nodes for data in data_module.train_dataset.data_list])
            print("Using global_max =", global_max)
            data_module.global_max = global_max
        model = GraphVAE_Lightning(
            latent_features=latent_features,
            max_nodes=data_module.global_max,
            lr=cfg.hyperparameters.lr,
            output_dir_orig=cfg.hyperparameters.get("output_dir_orig", "results/orig_molecules"),
            output_dir_reco=cfg.hyperparameters.get("output_dir_reco", "results/reco_molecules")
        )
        wandb_project = "graphvae_sparrow"
        trainer = Trainer(
            default_root_dir="my_logs_dir",
            max_epochs=cfg.hyperparameters.n_epochs,
            logger=pl.loggers.WandbLogger(project=wandb_project)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
