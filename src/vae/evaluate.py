import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from PIL import Image
import os

# Suppose you have two folders: one with real molecule images and one with generated ones.
# Images should be resized to (299,299) (Inception's expected input size) and normalized to [0,1].

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    # Inception network typically expects images normalized as below:
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = os.path.join(folder, filename)
            image = Image.open(path).convert("RGB")
            image = transform(image)
            images.append(image)
    # Stack images into a tensor: (N, 3, 299, 299)
    return torch.stack(images)

# Load your real and generated images
real_images = load_images_from_folder("path/to/real_images")
fake_images = load_images_from_folder("path/to/generated_images")

# Initialize the FID metric
fid_metric = FrechetInceptionDistance(feature=64)  # feature dim can be adjusted if needed

# Update metric with real images (set real=True)
fid_metric.update(real_images, real=True)

# Update metric with fake images (set real=False)
fid_metric.update(fake_images, real=False)

# Compute FID score
fid_score = fid_metric.compute()
print("FID Score:", fid_score.item())
