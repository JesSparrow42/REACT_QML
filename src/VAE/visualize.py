import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define input path
input_folder = "NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805"  # Replace with your DICOM folder path

def load_dicom_slices(folder_path):
    """Load DICOM slices from the specified folder into a 3D NumPy array."""
    dicom_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    
    # Read DICOM slices and sort by ImagePositionPatient (if available)
    dicom_data = [pydicom.dcmread(file) for file in dicom_files]
    dicom_data.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0)

    # Extract pixel data
    slices = [ds.pixel_array for ds in dicom_data]
    pixel_spacing = dicom_data[0].PixelSpacing  # Extract pixel spacing (x, y)
    slice_thickness = dicom_data[0].SliceThickness  # Extract slice thickness
    spacing = (*pixel_spacing, slice_thickness)
    
    # Stack slices into a 3D array
    volume = np.stack(slices, axis=-1)
    return volume, spacing

def normalize_array(array):
    """Normalize a NumPy array to the range 0-255."""
    return ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)

def visualize_vertical_section(volume, spacing, orientation="sagittal", output_path=None):
    """
    Visualize a fixed vertical cross-section of the 3D volume.
    Args:
        volume: 3D NumPy array.
        spacing: Tuple of voxel spacing (dx, dy, dz).
        orientation: "sagittal" (x-axis) or "coronal" (y-axis).
        output_path: Optional path to save the visualization.
    """
    if orientation == "sagittal":
        # Extract sagittal slice from the middle of the x-axis
        middle_index = volume.shape[1] // 2
        vertical_slice = volume[:, middle_index, :]
    elif orientation == "coronal":
        # Extract coronal slice from the middle of the y-axis
        middle_index = volume.shape[0] // 2
        vertical_slice = volume[middle_index, :, :]
        # Transpose to ensure correct orientation for visualization
        vertical_slice = vertical_slice.T
    else:
        raise ValueError("Invalid orientation. Choose 'sagittal' or 'coronal'.")

    # Normalize the slice for visualization
    vertical_slice_normalized = normalize_array(vertical_slice)

    # Adjust aspect ratio for correct voxel spacing
    if orientation == "coronal":
        aspect_ratio = spacing[2] / spacing[1]  # Depth to pixel width ratio
    else:
        aspect_ratio = spacing[2] / spacing[0]  # Depth to pixel height ratio

    # Display the vertical slice
    plt.figure(figsize=(10, 10))
    plt.imshow(vertical_slice_normalized, cmap="gray", aspect=aspect_ratio)
    plt.title(f"Vertical Cross-Section ({orientation.capitalize()})")
    plt.axis("off")
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

# Load DICOM slices
print("Loading DICOM slices...")
volume, spacing = load_dicom_slices(input_folder)

# Visualize a vertical section (sagittal and coronal examples)
#print("Visualizing sagittal section...")
#visualize_vertical_section(volume, spacing, orientation="sagittal", output_path="sagittal_section.jpg")

print("Visualizing coronal section...")
visualize_vertical_section(volume, spacing, orientation="coronal", output_path="coronal_section.jpg")

print("Vertical sections saved as 'sagittal_section.jpg' and 'coronal_section.jpg'")

def load_image_slices(folder_path):
    """
    Load 2D image slices from a folder and stack them into a 3D volume.
    Args:
        folder_path: Path to the folder containing image slices.
    Returns:
        volume: 3D NumPy array representing the stacked image slices.
    """
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    slices = [np.array(Image.open(img)) for img in image_files]
    
    # Ensure all images have the same dimensions
    slices = [slice_ if len(slice_.shape) == 2 else slice_[:, :, 0] for slice_ in slices]  # Convert RGB to grayscale if needed
    volume = np.stack(slices, axis=-1)
    return volume

def normalize_array(array):
    """Normalize a NumPy array to the range 0-255."""
    return ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)

def visualize_vertical_section_from_images(volume, spacing, orientation="sagittal", output_path=None):
    """
    Visualize a vertical cross-section from a 3D volume built from 2D images.
    Args:
        volume: 3D NumPy array representing the stacked image slices.
        spacing: Tuple of voxel spacing (dx, dy, dz).
        orientation: "sagittal" (x-axis) or "coronal" (y-axis).
        output_path: Optional path to save the visualization.
    """
    if orientation == "sagittal":
        middle_index = volume.shape[1] // 2
        vertical_slice = volume[:, middle_index, :]
    elif orientation == "coronal":
        middle_index = volume.shape[0] // 2
        vertical_slice = volume[middle_index, :, :]
        # Transpose for correct orientation
        vertical_slice = vertical_slice.T
    else:
        raise ValueError("Invalid orientation. Choose 'sagittal' or 'coronal'.")

    # Normalize for visualization
    vertical_slice_normalized = normalize_array(vertical_slice)

    # Adjust aspect ratio for correct voxel spacing
    if orientation == "coronal":
        aspect_ratio = spacing[2] / spacing[1]  # Depth to pixel width ratio
    else:
        aspect_ratio = spacing[2] / spacing[0]  # Depth to pixel height ratio

    # Display the vertical slice
    plt.figure(figsize=(10, 10))
    plt.imshow(vertical_slice_normalized, cmap="gray", aspect=aspect_ratio)
    plt.title(f"Vertical Cross-Section ({orientation.capitalize()})")
    plt.axis("off")
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

# Define input folder containing 2D slices
input_folder = "generated_ct_images"  # Replace with your folder path

# Define voxel spacing (you may need to adjust this based on real data)
pixel_spacing = (1.0, 1.0)  # Example spacing in x and y (e.g., mm per pixel)
slice_thickness = 2.5  # Example spacing in z (e.g., mm between slices)
spacing = (*pixel_spacing, slice_thickness)

# Load image slices and stack into a 3D volume
print("Loading image slices...")
volume = load_image_slices(input_folder)

# Visualize vertical cross-sections (sagittal and coronal examples)
#print("Visualizing sagittal section...")
#visualize_vertical_section_from_images(volume, spacing, orientation="sagittal", output_path="sagittal_section_from_images.jpg")

print("Visualizing coronal section...")
visualize_vertical_section_from_images(volume, spacing, orientation="coronal", output_path="coronal_section_from_images.jpg")

print("Vertical sections saved as 'sagittal_section_from_images.jpg' and 'coronal_section_from_images.jpg'")
