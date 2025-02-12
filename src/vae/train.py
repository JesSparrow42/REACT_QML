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
from data import MedicalDataModule
from model import GAN_Lightning, VAE_Lightning, Transformer_Lightning
#from transformer import Transformer_Lightning
from utils import load_separate_checkpoints

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.hyperparameters.seed)

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

    latent_features = cfg.hyperparameters.get("latent_features", 8)
    output_dir = cfg.hyperparameters.get("output_dir", "vae_images")

    ct_folder = cfg.hyperparameters.get("ct_folder", "Rigshospitalet/data/CT")
    pet_folder = cfg.hyperparameters.get("pet_folder", "Rigshospitalet/data/PET")
    data_module = MedicalDataModule(ct_folder, pet_folder, batch_size=cfg.hyperparameters.batch_size)

    model_type = cfg.hyperparameters.get("model_type", "vae").lower()

    if model_type == "gan":
        print(">> Training GAN model.")
        model = GAN_Lightning(
            boson_sampler_params=boson_params_to_use,
            gen_lr=cfg.hyperparameters.gen_lr,
            disc_lr=cfg.hyperparameters.disc_lr,
            latent_dim=latent_features,
            output_size=cfg.hyperparameters.get("gan_output_size", 64),
            output_dir=output_dir,
            pretrain_gen_epochs=cfg.hyperparameters.get("pretrain_gen_epochs", 250),
            pretrain_disc_epochs=cfg.hyperparameters.get("pretrain_disc_epochs", 50)
        )
        # Load separate checkpoints for generator and discriminator if provided
        gen_ckpt_path = cfg.hyperparameters.get("gen_checkpoint_path", None)
        disc_ckpt_path = cfg.hyperparameters.get("disc_checkpoint_path", None)
        if gen_ckpt_path and disc_ckpt_path:
            # Make sure the model is moved to the proper device before loading.
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
            output_dir
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
        # Instantiate the Transformer Lightning module with its specific hyperparameters.
        model = Transformer_Lightning(
            boson_params_to_use=boson_params_to_use,
            lr=cfg.hyperparameters.lr,
            latent_dim=latent_features,
            embed_dim=cfg.hyperparameters.get("embed_dim", 256),
            num_layers=cfg.hyperparameters.get("num_layers", 4),
            output_dir=output_dir,
            # Assuming input_shape is provided as a list (e.g., [128, 128]).
            input_shape=tuple(cfg.hyperparameters.get("input_shape", [128, 128]))
        )
        wandb_project = "transformer_sparrow"
        trainer = Trainer(
            default_root_dir="my_logs_dir",
            max_epochs=cfg.hyperparameters.n_epochs,
            logger=pl.loggers.WandbLogger(project=wandb_project),
            gradient_clip_val=0.5
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
