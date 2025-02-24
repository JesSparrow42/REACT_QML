# python evaluate.py path/to/real_image_folder/ path/to/fake_image_folder/

#!/usr/bin/env python
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

# Define the transform for the Inception network.
# Images are resized to (299,299) and normalized with ImageNet statistics.
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_images_from_folder(folder: str) -> torch.Tensor:
    """Loads and transforms all images from a given folder.
    
    Args:
        folder (str): Path to the folder containing images.
        
    Returns:
        torch.Tensor: A tensor of shape (N, 3, 299, 299) with the loaded images.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                img = transform(img)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {path}: {e}")
    if not images:
        raise ValueError(f"No images found in folder: {folder}")
    return torch.stack(images)

def main():
    parser = argparse.ArgumentParser(description="Evaluate FID score between two image folders.")
    parser.add_argument("real_folder", type=str, help="Folder with real images.")
    parser.add_argument("fake_folder", type=str, help="Folder with generated (fake) images.")
    args = parser.parse_args()

    print("Loading real images from:", args.real_folder)
    real_images = load_images_from_folder(args.real_folder)
    print(f"Loaded {real_images.size(0)} real images.")

    print("Loading fake images from:", args.fake_folder)
    fake_images = load_images_from_folder(args.fake_folder)
    print(f"Loaded {fake_images.size(0)} fake images.")

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature=64)  # feature=64 is default

    # Update FID with real and fake images
    fid_metric.update(real_images, real=True)
    fid_metric.update(fake_images, real=False)

    fid_score = fid_metric.compute()
    print("FID Score:", fid_score.item())

if __name__ == "__main__":
    main()
