# combined_data.py
import os
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import pydicom
from pydicom import dcmread
from pydicom.dataset import FileDataset
from pathlib import Path
import pytorch_lightning as pl

# ---------------- Utility Functions ----------------

def __generate_uid_suffix() -> str:
    """Generate and return a new UID suffix."""
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

def generate_SeriesInstanceUID() -> str:
    """Generate and return a new SeriesInstanceUID."""
    return '1.3.12.2.1107.5.2.38.51014.{}11111.0.0.0'.format(__generate_uid_suffix())

def generate_SOPInstanceUID(i: int) -> str:
    """Generate and return a new SOPInstanceUID."""
    return '1.3.12.2.1107.5.2.38.51014.{}{}'.format(__generate_uid_suffix(), i)

def to_dcm(ct_tensor, output_dir, pixel_spacing=(1.0, 1.0), slice_thickness=1.0):
    """
    Converts a 3D tensor representing a CT scan into a series of DICOM files.

    Args:
        ct_tensor (torch.Tensor): Input tensor with shape [Depth, Height, Width].
        output_dir (str): Directory to save the generated DICOM files.
        pixel_spacing (tuple): Spacing between pixels (default is (1.0, 1.0)).
        slice_thickness (float): Thickness of each slice (default is 1.0).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    ct_array = ct_tensor.cpu().numpy()
    ct_array = (ct_array * 2000 - 1000).astype(np.int16)

    def create_dicom_metadata():
        ds = FileDataset("", {}, file_meta=pydicom.dataset.FileMetaDataset(), preamble=b"\0" * 128)
        ds.PatientName = "Generated Patient"
        ds.PatientID = "123456"
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.Modality = "CT"
        ds.SeriesDescription = "Generated CT"
        ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
        ds.Manufacturer = "GeneratedData"
        ds.PixelSpacing = list(pixel_spacing)
        ds.SliceThickness = slice_thickness
        return ds

    for i in range(ct_array.shape[0]):
        slice_data = ct_array[i, :, :]
        dicom_slice = create_dicom_metadata()
        dicom_slice.InstanceNumber = i + 1
        dicom_slice.ImagePositionPatient = [0.0, 0.0, i * dicom_slice.SliceThickness]
        dicom_slice.Rows, dicom_slice.Columns = slice_data.shape
        dicom_slice.BitsStored = 16
        dicom_slice.BitsAllocated = 16
        dicom_slice.HighBit = 15
        dicom_slice.PixelRepresentation = 1
        dicom_slice.RescaleIntercept = -1000
        dicom_slice.RescaleSlope = 1
        dicom_slice.PixelData = slice_data.tobytes()
        output_path = os.path.join(output_dir, f"slice_{i + 1:04d}.dcm")
        dicom_slice.save_as(output_path)
        print(f"Saved: {output_path}")

def get_sort_files_dict(path, path2):
    """Sort DICOM files by modality and slice location."""
    files = {'CT': {}, 'PT': {}}
    for p in Path(path).rglob('*'):
        if p.is_file() and not p.name.startswith('.'):
            try:
                ds = dcmread(str(p))
                modality = ds.get('Modality', 'Unknown')
                slice_location = ds.get('SliceLocation', None)
                if modality and slice_location is not None:
                    files[modality][slice_location] = p
            except Exception as e:
                print(f"\tSkipping {p}: Not a valid DICOM file? Error: {e}")
                continue
    for p in Path(path2).rglob('*'):
        if p.is_file() and not p.name.startswith('.'):
            try:
                ds = dcmread(str(p))
                modality = ds.get('Modality', 'Unknown')
                slice_location = ds.get('SliceLocation', None)
                if modality and slice_location is not None:
                    files[modality][slice_location] = p
            except Exception as e:
                print(f"\tSkipping {p}: Not a valid DICOM file? Error: {e}")
                continue
    sorted_files = {}
    for k, d in files.items():
        sorted_files[k] = dict(sorted(d.items()))
        print(f"Found {len(sorted_files[k])} {k} files")
    return sorted_files

# ---------------- Dataset and DataModule ----------------

class DICOMDataset(Dataset):
    def __init__(self, ct_folder, pet_folder, patch_size=(128, 128), augment=True):
        self.ct_folder = ct_folder
        self.pet_folder = pet_folder
        self.patch_size = patch_size
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.GaussianBlur(3)
        ])
        self.ct_files = sorted([os.path.join(ct_folder, f) for f in os.listdir(ct_folder) if f.endswith('.dcm')])
        self.pet_files = sorted([os.path.join(pet_folder, f) for f in os.listdir(pet_folder) if f.endswith('.dcm')])
        assert len(self.ct_files) == len(self.pet_files), "CT and PET folders must contain the same number of files."
        print(f"Loaded {len(self.ct_files)} CT and {len(self.pet_files)} PET files")

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        ct_file = self.ct_files[idx]
        pet_file = self.pet_files[idx]
        
        # Read DICOM files
        ct_ds = dcmread(ct_file)
        pet_ds = dcmread(pet_file)
        
        # Get the pixel arrays (already cast to float32, but still in numpy, likely uint8 later)
        ct_image = ct_ds.pixel_array.astype(np.float32)
        pet_image = pet_ds.pixel_array.astype(np.float32)
        
        # Convert numpy arrays to PIL images
        to_pil = ToPILImage()
        ct_image_pil = to_pil(ct_image)
        pet_image_pil = to_pil(pet_image)
        
        # Apply augmentations if required
        if self.augment:
            ct_image_pil = self.transform(ct_image_pil)
            pet_image_pil = self.transform(pet_image_pil)
        
        # Convert PIL images to tensors using ToTensor(), which returns Float tensors scaled to [0,1]
        to_tensor = transforms.ToTensor()
        ct_image_tensor = to_tensor(ct_image_pil)
        pet_image_tensor = to_tensor(pet_image_pil)
        
        return pet_image_tensor, ct_image_tensor


def create_dataset(ct_folder, pet_folder, augment=True):
    """
    Returns a DICOMDataset instance instead of a DataLoader.
    """
    if not os.path.isdir(ct_folder):
        raise ValueError(f"The CT folder path {ct_folder} does not exist or is not a directory.")
    if not os.path.isdir(pet_folder):
        raise ValueError(f"The PET folder path {pet_folder} does not exist or is not a directory.")
    return DICOMDataset(ct_folder, pet_folder, augment=augment)

class MedicalDataModule(pl.LightningDataModule):
    def __init__(self, ct_folder, pet_folder, batch_size=64, num_workers=4, augment=False):
        super().__init__()
        self.ct_folder = ct_folder
        self.pet_folder = pet_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

    def setup(self, stage=None):
        self.dataset = create_dataset(
            ct_folder=self.ct_folder,
            pet_folder=self.pet_folder,
            augment=self.augment
        )

        # Optionally, split the dataset into train and val portions
        if stage == 'fit' or stage is None:
            dataset_size = len(self.dataset)
            split = int(0.8 * dataset_size)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, [split, dataset_size - split]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True               # Drop last is fine for training
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False              # Ensure validation batch is returned even if incomplete
        )
