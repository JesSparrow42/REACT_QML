import torch
import torch.optim as optim
import torch.nn as nn
from qgan import Generator, Discriminator
from loss import generator_loss, dice_loss
import torch.nn.functional as F
import os
from data_loader import create_data_loader
from ptseries.models import PTGenerator
from ptseries.algorithms.gans.utils import infiniteloop
from utils import *

### TO-DO ###
# 1. Save/initialise weights
# 2. Hypertuning
# 3. Look at wheter iterations are worthwhile to implement
# 4. Loss goes negative. Check loss calculation & learning rate
###

def main():
    ### HYPERPARAMETERS
    boson_sampler_params = {
        "input_state": [1, 0, 1, 0, 1, 0, 1, 0], # |1,0,1,0,1,0,1,0>
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
            "loop_lengths": [2,3,4],
            "postselected": True
        },
        "n_tiling": 1
    }
    DISCRIMINATOR_ITER = 3000
    NUM_ITER = 1
    DISC_LR = 2e-6
    GEN_LR = 0.0004
    ###

    ct_folder = 'NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805'
    pet_folder = 'NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/1.000000-PET NAC-24000'

    num_workers = 4  # Adjust this
    dicom_files = create_data_loader(ct_folder=ct_folder, pet_folder=pet_folder, num_workers=num_workers, augment=False)

    # Create latent space
    latent_space = PTGenerator(**boson_sampler_params)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Create GAN
    generator = Generator(len(boson_sampler_params["input_state"]),output_size=(512,512))
    discriminator = Discriminator()
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Load data
    looper = infiniteloop(dicom_files)
    batch = next(looper)
    pet_images, ct_images = batch
    data_real = ct_images.to(device)
    pet_images = pet_images.to(device)
    batch_size = pet_images.shape[0]
    
    # Optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=GEN_LR)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=DISC_LR)


    # Mixed Precision Scalers
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_disc = torch.cuda.amp.GradScaler()

    latent = latent_space.generate((1 + DISCRIMINATOR_ITER) * batch_size).to(device)
    latent = torch.chunk(latent, 1 + DISCRIMINATOR_ITER, dim=0)
        
    gen_checkpoint_path = 'model_checkpoints/generator_epoch_3000.pt'
    disc_checkpoint_path = 'model_checkpoints/discriminator_epoch_3000.pt'
    start_epoch_gen, _ = load_weights(generator, gen_optimizer, gen_checkpoint_path, device)
    start_epoch_disc, _ = load_weights(discriminator, disc_optimizer, disc_checkpoint_path, device)

    # Pre-train Generator
    print("Starting Generator Pre-training")
    for epoch in range(start_epoch_gen, 250):
        for p in discriminator.parameters():
            p.requires_grad_(False)
        with torch.cuda.amp.autocast():
            generated_ct = generator(latent[DISCRIMINATOR_ITER])
            data_real_downscaled = F.interpolate(data_real, size=(64, 64), mode='nearest')
            data_real_downscaled = (data_real_downscaled - data_real_downscaled.min()) / (data_real_downscaled.max() - data_real_downscaled.min())
            generated_ct = (generated_ct - generated_ct.min()) / (generated_ct.max() - generated_ct.min())

            fake_output_for_gen = discriminator(generated_ct)
            adversarial_loss = F.binary_cross_entropy_with_logits(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
            gen_loss = F.l1_loss(generated_ct, data_real_downscaled) + 0.1 * adversarial_loss

        gen_optimizer.zero_grad()
        scaler_gen.scale(gen_loss).backward()
        scaler_gen.step(gen_optimizer)
        scaler_gen.update()
        print(f"Epoch {epoch+1}, Gen Loss: {gen_loss.item()}")
    
    # Pre-train Discriminator
    print("Starting Discriminator Pre-training")
    for epoch in range(start_epoch_disc, 50):
        for p in discriminator.parameters():
            p.requires_grad_(True)
        if epoch + 1 == 25:
            print("Dropping disc. learning rate")
            disc_optimizer = optim.Adam(discriminator.parameters(), lr=DISC_LR*0.1)
        
        with torch.no_grad():
            data_fake = generator(latent[epoch]).detach()
        with torch.cuda.amp.autocast():
            real_output = discriminator(data_real)
            fake_output = discriminator(data_fake)

            real_target = torch.ones_like(real_output)
            fake_target = torch.zeros_like(fake_output)

            disc_loss_real = dice_loss(real_output, real_target)
            disc_loss_fake = dice_loss(fake_output, fake_target)
            disc_loss = disc_loss_real + disc_loss_fake
        
        disc_optimizer.zero_grad()
        scaler_disc.scale(disc_loss).backward()
        scaler_disc.step(disc_optimizer)
        scaler_disc.update()
        print(f"Epoch {epoch+1}, Disc Loss: {disc_loss.item()}")

    gen_losses = []
    disc_losses = []

    # Main training loop follows the same structure as pre-training but with mixed precision applied
    print("Starting main training")
    for n in range(NUM_ITER):
        for epoch in range(DISCRIMINATOR_ITER):
            # Train Discriminator
            for p in discriminator.parameters():
                p.requires_grad_(True)
            with torch.no_grad():
                data_fake = generator(latent[epoch]).detach()
            with torch.cuda.amp.autocast():
                real_output = discriminator(data_real)
                fake_output = discriminator(data_fake)

                real_target = torch.ones_like(real_output)
                fake_target = torch.zeros_like(fake_output)

                disc_loss_real = dice_loss(real_output, real_target)
                disc_loss_fake = dice_loss(fake_output, fake_target)
                disc_loss = disc_loss_real + disc_loss_fake
            
            disc_optimizer.zero_grad()
            scaler_disc.scale(disc_loss).backward()
            scaler_disc.step(disc_optimizer)
            scaler_disc.update()

            # Train Generator
            for p in discriminator.parameters():
                p.requires_grad_(False)
            with torch.cuda.amp.autocast():
                generated_ct = generator(latent[DISCRIMINATOR_ITER])
                data_real_downscaled = F.interpolate(data_real, size=(64, 64), mode='nearest')
                data_real_downscaled = (data_real_downscaled - data_real_downscaled.min()) / (data_real_downscaled.max() - data_real_downscaled.min())
                generated_ct = (generated_ct - generated_ct.min()) / (generated_ct.max() - generated_ct.min())

                fake_output_for_gen = discriminator(generated_ct)
                adversarial_loss = dice_loss(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                gen_loss = F.l1_loss(generated_ct, data_real_downscaled) + 0.2 * adversarial_loss
            
            gen_optimizer.zero_grad()
            scaler_gen.scale(gen_loss).backward()
            scaler_gen.step(gen_optimizer)
            scaler_gen.update()
            
            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            print(f"Epoch {epoch+1}, Gen Loss: {gen_loss.item()}, Disc Loss: {disc_loss.item()}")
            if (epoch + 1) % 250 == 0:
                # Save model logic remains the same
                model_save_folder = 'model_checkpoints'
                os.makedirs(model_save_folder, exist_ok=True)

                generator_save_path = os.path.join(model_save_folder, f'generator_epoch_{epoch + 1}.pt')
                discriminator_save_path = os.path.join(model_save_folder, f'discriminator_epoch_{epoch + 1}.pt')
                save_weights(generator, gen_optimizer, epoch + 1, generator_save_path, gen_loss.item())
                save_weights(discriminator, disc_optimizer, epoch + 1, discriminator_save_path, disc_loss.item())
        plot_losses(gen_losses, disc_losses, phase="Main Training")
        print(f'Iteration, {n}, Gen Loss: {gen_loss.item()}, Disc Loss: {disc_loss.item()}')
if __name__ == '__main__':
    main()
