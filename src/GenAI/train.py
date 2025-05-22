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

# Import your data modules and models
from data import MedicalDataModule, QM9DataModule, AtomsData, ReactDataModule
from model import GAN_Lightning, VAE_Lightning, DiffusionLightning, UNetLightning, GraphVAE_Lightning
# Import the custom boson sampler (only needed for the custom option)
from BosonSamplerWrapper import BosonLatentGenerator, BosonSamplerTorch
from utils import load_separate_checkpoints

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.hyperparameters.seed)
    
    # --------------------------------------------------------------------------------------
    # Boson Sampler configuration
    # --------------------------------------------------------------------------------------
    # Set the boson flag ("false" to use a standard Gaussian prior, otherwise use boson sampler).
    boson_flag = cfg.hyperparameters.get("boson_sampler_params", "true")
    if boson_flag == "false":
        boson_params_to_use = None
        # Use default latent_features for cases where no boson sampler is applied.
        latent_features = cfg.hyperparameters.get("latent_features", 8)
        print(">> Using standard Gaussian prior.")
    else:
        # Select boson sampler type via config parameter (either "ptseries" or "custom")
        boson_sampler_type = cfg.hyperparameters.get("boson_sampler_type", "ptseries").lower()
        if boson_sampler_type == "ptseries":
            latent_features = cfg.hyperparameters.get("latent_features", 8)
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
            print(">> Using PTSeries Boson Sampler as prior.")
        elif boson_sampler_type == "custom":
            # For the custom sampler we use a latent dimension (default 14 if not specified)
            latent_features = cfg.hyperparameters.get("latent_features", 14)
            boson_params_to_use = BosonSamplerTorch(
                m=latent_features,
                num_sources=latent_features // 2,
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
                mode_loss=np.ones(latent_features),
                dark_count_rate=0,
                use_advanced_nonlinearity=False,
                chi=1
            )
            print(">> Using Custom Boson Sampler as prior.")
        else:
            raise ValueError("Unknown boson_sampler_type specified. Options: 'ptseries' or 'custom'")
    
    # --------------------------------------------------------------------------------------
    # Data Module Setup
    # --------------------------------------------------------------------------------------
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
    elif datamodule_type == "react":
        print(">> Using React DataModule.")
        data_module = ReactDataModule(
            csv_path=cfg.hyperparameters.react_csv_path,
            mesh_dir=cfg.hyperparameters.react_mesh_dir,
            batch_size=cfg.hyperparameters.get("batch_size", 32),
            val_split=cfg.hyperparameters.get("react_val_split", 0.2),
            test_split=cfg.hyperparameters.get("react_test_split", 0.1),
            mesh_ext=cfg.hyperparameters.get("react_mesh_ext", ".ply"),
            transform_matrix=cfg.hyperparameters.get("react_transform_matrix", None),
            num_workers=cfg.hyperparameters.get("react_num_workers", 4)
        )
    else:
        raise ValueError(f"Unknown datamodule type: {datamodule_type}")

    # --------------------------------------------------------------------------------------
    # Model and Trainer Selection
    # --------------------------------------------------------------------------------------
    model_type = cfg.hyperparameters.get("model_type", "vae").lower()
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
        # Compute global_max if using non-QM9 data_module
        if datamodule_type != "qm9":
            global_max = max([data.num_nodes for data in data_module.train_dataset.data_list])
            print("Using global_max =", global_max)
            data_module.global_max = global_max
        # Optionally include the boson sampler in GraphVAE if desired (default: False)
        if cfg.hyperparameters.get("graphvae_include_boson", False):
            model = GraphVAE_Lightning(
                boson_params_to_use,
                latent_features=latent_features,
                max_nodes=data_module.global_max,
                lr=cfg.hyperparameters.lr,
                output_dir_orig=cfg.hyperparameters.get("output_dir_orig", "results/orig_molecules"),
                output_dir_reco=cfg.hyperparameters.get("output_dir_reco", "results/reco_molecules")
            )
        else:
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
    
    # --------------------------------------------------------------------------------------
    # Start Training
    # --------------------------------------------------------------------------------------
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
