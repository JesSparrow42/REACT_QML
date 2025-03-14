# python evaluate.py path/to/real_image_folder/ path/to/fake_image_folder/
#!/usr/bin/env python
import os
import re
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

# Transform: Resize, convert to tensor, scale to [0,255], and cast to uint8.
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
])

def extract_epoch(filename: str) -> int:
    """
    Extracts the epoch number from a filename of the form 'epoch_{number}_reconstructed_{anything}'.
    Returns 0 if no match is found.
    """
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0

def load_images_for_epoch(folder: str, epoch_number: int, num_images: int = 32) -> torch.Tensor:
    """
    Loads and transforms up to num_images from a given folder, filtering by the specified epoch_number.
    
    Args:
        folder (str): Path to the folder containing images.
        epoch_number (int): The epoch to filter on.
        num_images (int): Max number of images to load.
        
    Returns:
        torch.Tensor: A tensor of shape (N, 3, 299, 299) with the loaded images.
    """
    # Get all filenames in the folder
    all_filenames = os.listdir(folder)
    
    # Filter: only keep images that match the epoch_number
    filtered_filenames = [
        f for f in all_filenames
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and extract_epoch(f) == epoch_number
    ]
    
    # Sort the filtered filenames (if needed, by sub-index or simply lexicographically)
    filtered_filenames = sorted(filtered_filenames)
    
    # If more than num_images, take the last num_images (or first, depending on preference)
    if len(filtered_filenames) > num_images:
        filtered_filenames = filtered_filenames[-num_images:]
    
    # Load and transform images
    images = []
    for filename in filtered_filenames:
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {path}: {e}")
    
    if not images:
        raise ValueError(f"No images found for epoch {epoch_number} in folder: {folder}")
    
    return torch.stack(images)

def main():
    parser = argparse.ArgumentParser(description="Evaluate FID score between two image folders for a specific epoch.")
    parser.add_argument("real_folder", type=str, help="Folder with real images.")
    parser.add_argument("fake_folder", type=str, help="Folder with generated (fake) images.")
    parser.add_argument("epoch_number", type=int, help="Epoch number to filter on.")
    args = parser.parse_args()

    print(f"Loading images for epoch {args.epoch_number} from real folder: {args.real_folder}")
    real_images = load_images_for_epoch(args.real_folder, args.epoch_number, num_images=32)
    print(f"Loaded {real_images.size(0)} real images.")

    print(f"Loading images for epoch {args.epoch_number} from fake folder: {args.fake_folder}")
    fake_images = load_images_for_epoch(args.fake_folder, args.epoch_number, num_images=32)
    print(f"Loaded {fake_images.size(0)} fake images.")

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature=64)

    # Update FID with real and fake images
    fid_metric.update(real_images, real=True)
    fid_metric.update(fake_images, real=False)

    fid_score = fid_metric.compute()
    print("FID Score:", fid_score.item())

if __name__ == "__main__":
    main()
