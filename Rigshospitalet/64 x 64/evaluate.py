import torch
import torch.nn.functional as F
from qgan import Generator
from ptseries.models import PTGenerator
from data_loader import create_data_loader
import os
from pathlib import Path
from PIL import Image
import numpy as np

### TO-DO ###
# 1. Reconstruction of 3D images
# 2. Option to save a DCM file
###

def load_generator(model_path, device, input_state_length):
    # Initialize the generator model
    generator = Generator(input_state_length)
    generator.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    generator.to(device)
    generator.eval()  # Set the generator to evaluation mode
    return generator

def generate_ct_images(generator, latent_space, data_loader, device, output_folder='generated_ct_images'):
    os.makedirs(output_folder, exist_ok=True)
    for i, (pet_images, _) in enumerate(data_loader):
        pet_images = pet_images.to(device)
        
        # Generate latent vectors based on the batch size
        batch_size = pet_images.size(0)
        latent_vectors = latent_space.generate(batch_size).to(device)
        
        # Generate CT images
        with torch.no_grad():
            generated_ct_images = generator(latent_vectors)
        
        # Normalize generated images for saving
        generated_ct_images = (generated_ct_images - generated_ct_images.min()) / (generated_ct_images.max() - generated_ct_images.min())
        
        # Save generated images
        for j in range(batch_size):
            image_path = os.path.join(output_folder, f'generated_ct_image_{i * batch_size + j}.png')
            save_image(generated_ct_images[j], image_path)
            print(f"Saved generated CT image at {image_path}")

def save_image(tensor, path):
    # Convert the tensor to a NumPy array
    image = tensor.cpu().numpy()
    
    # Check if the image is grayscale (single channel) or RGB (3 channels)
    if image.shape[0] == 1:  # Grayscale image
        image = image.squeeze(0)  # Remove the channel dimension for PIL compatibility
    else:
        image = image.transpose(1, 2, 0)  # Convert from CHW to HWC format for RGB
    
    # Scale to 0-255 and convert to uint8
    image = (image * 255).astype(np.uint8)
    
    # Save as an image
    image = Image.fromarray(image)
    image.save(path)


def main():
    # Hyperparameters
    boson_sampler_params = {
        "input_state": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], # |1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0>
        "tbi_params": {
            "input_loss": 0.0,
            "detector_efficiency": 1,
            "bs_loss": 0,
            "bs_noise": 0,
            "distinguishable": False,
            "n_signal_detectors": 0,
            "g2": 0,
            "tbi_type": "multi-loop",
            "n_loops": 2,
            "loop_lengths": [1,2],
            "postselected": True
        },
        "n_tiling": 1
    }
    
    # Paths
    pet_folder = 'NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/1.000000-PET NAC-24000'
    ct_folder = 'NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805'
    generator_model_path = 'model_checkpoints/generator_epoch_3000.pt'
    output_folder = 'generated_ct_images'
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load the generator model
    input_state_length = len(boson_sampler_params["input_state"])
    generator = load_generator(generator_model_path, device, input_state_length)
    
    # Create the latent space generator
    latent_space = PTGenerator(**boson_sampler_params)
    
    # Create data loader for PET images only (CT images not needed here)
    pet_data_loader = create_data_loader(pet_folder=pet_folder, ct_folder=ct_folder, num_workers=4,shuffle=False)
    
    # Generate and save CT images
    generate_ct_images(generator, latent_space, pet_data_loader, device, output_folder)

if __name__ == '__main__':
    main()
