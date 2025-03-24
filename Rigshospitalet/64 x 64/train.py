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
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
import matplotlib.pyplot as plt

def get_activations(images, model, batch_size=50, device='cpu'):
    """Computes the activations of the pool3 layer for all images."""
    model.eval()
    activations = []
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to(device)
            # Resize images to 299x299 if needed
            if batch.shape[-1] != 299 or batch.shape[-2] != 299:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            pred = model(batch)
            activations.append(pred.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    return activations


def calculate_activation_statistics(activations):
    """Calculates the mean and covariance of the activations."""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculates the Fréchet Inception Distance between two distributions."""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid


def get_inception_features(x, inception):
    """Extracts features from the Inception v3 network up to the final pooling layer."""
    x = inception.Conv2d_1a_3x3(x)
    x = inception.Conv2d_2a_3x3(x)
    x = inception.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = inception.Conv2d_3b_1x1(x)
    x = inception.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = inception.Mixed_5b(x)
    x = inception.Mixed_5c(x)
    x = inception.Mixed_5d(x)
    x = inception.Mixed_6a(x)
    x = inception.Mixed_6b(x)
    x = inception.Mixed_6c(x)
    x = inception.Mixed_6d(x)
    x = inception.Mixed_6e(x)
    x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
    x = torch.flatten(x, 1)
    return x


def compute_fid(real_images, fake_images, batch_size=50, device='cpu'):
    """Computes the FID score between real and fake images."""
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    # Override the forward method to return features from the pool3 layer
    inception.forward = lambda x: get_inception_features(x, inception)
    
    # If images are grayscale, repeat channels to have 3 channels
    if real_images.shape[1] == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
    if fake_images.shape[1] == 1:
        fake_images = fake_images.repeat(1, 3, 1, 1)
    
    # Scale images from [0, 1] to [-1, 1] as expected by Inception
    real_images = real_images * 2 - 1
    fake_images = fake_images * 2 - 1
    
    activations_real = get_activations(real_images, inception, batch_size, device)
    activations_fake = get_activations(fake_images, inception, batch_size, device)
    
    mu_real, sigma_real = calculate_activation_statistics(activations_real)
    mu_fake, sigma_fake = calculate_activation_statistics(activations_fake)
    
    fid_value = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_value

def main():
    ### HYPERPARAMETERS # Only optimize boson sampler parameters
    boson_sampler_params = {
        "input_state": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], # |1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0> # Higher photon number occupancy
        "tbi_params": {
            "input_loss": 0.0,
            "detector_efficiency": 1,
            "bs_loss": 0,
            "bs_noise": 0,
            "distinguishable": False, # Keep False
            "n_signal_detectors": 0,
            "g2": 0,
            "tbi_type": "multi-loop",
            "n_loops": 2,
            "loop_lengths": [1,2],
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

    # Create latent space # train 
    latent_space = PTGenerator(**boson_sampler_params) # Centering to make more like gaussian
    # Global mean or mean per mode

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
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
    scaler_gen = torch.amp.GradScaler()
    scaler_disc = torch.amp.GradScaler()

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
        with torch.amp.autocast('mps'):
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
        with torch.amp.autocast('mps'):
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
    fid_scores = []
    fid_epochs = []

    # Main training loop follows the same structure as pre-training but with mixed precision applied
    print("Starting main training")
    for n in range(NUM_ITER):
        for epoch in range(DISCRIMINATOR_ITER):
            # Train Discriminator
            for p in discriminator.parameters():
                p.requires_grad_(True)
            with torch.no_grad():
                data_fake = generator(latent[epoch]).detach()
            with torch.amp.autocast('mps'):
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
            with torch.amp.autocast('mps'):
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

                with torch.no_grad():
                    fake_ct = generator(latent[DISCRIMINATOR_ITER]).detach()
                    fake_ct = (fake_ct - fake_ct.min()) / (fake_ct.max() - fake_ct.min())
                    real_ct = F.interpolate(data_real, size=(64, 64), mode='nearest')
                    real_ct = (real_ct - real_ct.min()) / (real_ct.max() - real_ct.min())
                fid = compute_fid(real_ct, fake_ct, batch_size=10, device=device)
                fid_scores.append(fid)
                fid_epochs.append(epoch + 1)
                print(f"Epoch {epoch+1}, FID score: {fid}")

    # Compute FID score after training
    print("Computing FID score...")
    with torch.no_grad():
        # Generate fake CT images using the generator
        fake_ct = generator(latent[DISCRIMINATOR_ITER]).detach()
        fake_ct = (fake_ct - fake_ct.min()) / (fake_ct.max() - fake_ct.min())
        # Prepare real CT images by downscaling
        real_ct = F.interpolate(data_real, size=(64, 64), mode='nearest')
        real_ct = (real_ct - real_ct.min()) / (real_ct.max() - real_ct.min())
    fid_score = compute_fid(real_ct, fake_ct, batch_size=10, device=device)
    print("FID score:", fid_score)

    # Plot FID score graph
    plt.figure()
    plt.plot(fid_epochs, fid_scores, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.title("FID Score over Training Epochs")
    plt.show()

if __name__ == '__main__':
    main()
