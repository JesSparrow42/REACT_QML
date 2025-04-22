#!/usr/bin/env python
# combined_data.py
import os
import json
import pickle
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import ToPILImage
import trimesh
import pydicom
from pydicom import dcmread
from pydicom.dataset import FileDataset
from pathlib import Path
from ase.db import connect
from ase import Atoms
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

        # Clip to a window and scale to [0,1]
        ct_min, ct_max = -1000, 400
        ct_image = np.clip(ct_image, ct_min, ct_max)
        ct_image = (ct_image - ct_min) / (ct_max - ct_min)

        # For PET images, you might use a different normalization or similar
        pet_image = np.clip(pet_image, pet_image.min(), pet_image.max())
        pet_image = (pet_image - pet_image.min()) / (pet_image.max() - pet_image.min())

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


###############################
# QM9 Data Object Classes
###############################

class BaseData:
    """A dict-like base class for data objects."""
    def __init__(self, **kwargs):
        # Use object.__setattr__ to avoid recursion
        object.__setattr__(self, "tensors", {})
        for key, value in kwargs.items():
            self.__setattr__(key, value)
    def __getattr__(self, key):
        tensors = object.__getattribute__(self, "tensors")
        if key in tensors:
            return tensors[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    def __setattr__(self, key, value) -> None:
        if isinstance(value, torch.Tensor):
            self.tensors[key] = value
            if key in self.__dict__:
                del self.__dict__[key]
        else:
            object.__setattr__(self, key, value)
            if key in self.tensors:
                del self.tensors[key]
    def validate(self) -> bool:
        for key, tensor in self.tensors.items():
            assert isinstance(tensor, torch.Tensor), f"'{key}' is not a tensor!"
        return True
    def to(self, device: torch.device) -> None:
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}

class Data(BaseData):
    """A data object describing a homogeneous graph."""
    def __init__(self,
                 node_features: torch.Tensor,
                 edge_index: torch.Tensor = torch.tensor([]),
                 edge_features: torch.Tensor | None = None,
                 targets: torch.Tensor | None = None,
                 global_features: torch.Tensor | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.global_features = global_features
        self.targets = targets
    @property
    def num_nodes(self) -> int:
        return self.tensors.get("num_nodes", self.node_features.shape[0])
    @property
    def num_edges(self) -> int:
        return self.tensors.get("num_edges", self.edge_index.shape[0])

class AtomsData(Data):
    """A data object describing atoms as a graph with spatial information."""
    def __init__(self,
                 node_positions: torch.Tensor,
                 energy: torch.Tensor | None = None,
                 forces: torch.Tensor | None = None,
                 magmoms: torch.Tensor | None = None,
                 cell: torch.Tensor | None = None,
                 volume: torch.Tensor | None = None,
                 stress: torch.Tensor | None = None,
                 pbc: torch.Tensor | None = None,
                 edge_shift: torch.Tensor | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.node_positions = node_positions
        self.energy = energy
        self.forces = forces
        self.magmoms = magmoms
        self.cell = cell
        self.volume = volume
        self.stress = stress
        self.pbc = pbc
        self.edge_shift = edge_shift
    def validate(self) -> bool:
        super().validate()
        assert self.node_positions.shape[0] == self.num_nodes
        assert self.node_positions.ndim == 2
        return True
    def any_pbc(self) -> bool:
        return self.pbc is not None and bool(torch.any(self.pbc))

###############################
# Collate Function for Graph VAE
###############################

def collate_graph_vae(list_of_data: list[AtomsData], global_max: int) -> dict:
    """
    Collate a list of AtomsData objects into a batch for the generative model.
    Pads each molecule to a fixed number of nodes equal to the global maximum.
    Returns a dict with:
      - node_features: Tensor of shape (B, global_max, 1)
      - node_positions: Tensor of shape (B, global_max, 3)
      - mask: Tensor of shape (B, global_max) with 1 for valid nodes, 0 for padding
      - num_nodes: Tensor of shape (B,) number of nodes per molecule
    """
    feat_list, pos_list, mask_list = [], [], []
    for data in list_of_data:
        n = data.num_nodes
        pad_n = global_max - n
        feat = data.node_features.float()
        pos = data.node_positions.float()
        pad_feat = F.pad(feat, (0, 0, 0, pad_n), value=0.0)
        pad_pos = F.pad(pos, (0, 0, 0, pad_n), value=0.0)
        mask = torch.cat([torch.ones(n), torch.zeros(pad_n)])
        feat_list.append(pad_feat)
        pos_list.append(pad_pos)
        mask_list.append(mask)
    batch = {
        "node_features": torch.stack(feat_list, dim=0),
        "node_positions": torch.stack(pos_list, dim=0),
        "mask": torch.stack(mask_list, dim=0),
        "num_nodes": torch.tensor([data.num_nodes for data in list_of_data])
    }
    return batch

###############################
# QM9 Dataset with Caching
###############################

def ase_to_atomsdata(atoms: Atoms, energy_property: str = "energy_U0") -> AtomsData:
    """
    Convert an ASE Atoms object from QM9 into an AtomsData object.
    """
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.float32).view(-1, 1)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32)
    pbc = torch.tensor([False, False, False])
    energy = torch.tensor([atoms.info.get(energy_property, 0.0)], dtype=torch.float32)
    edge_index = torch.empty((0, 2), dtype=torch.long)  # Not used in the VAE
    data = AtomsData(
        node_features=atomic_numbers,
        edge_index=edge_index,
        node_positions=positions,
        energy=energy,
        cell=cell,
        pbc=pbc,
        edge_shift=None
    )
    data.validate()
    return data

class QM9Dataset(Dataset):
    def __init__(self, db_path: str, split_path: str, split: str = "train", cache_file: str = None):
        """
        Loads QM9 data from the ASE database for a given split.
        If cache_file is provided and exists, loads the dataset from it.
        Otherwise, builds it from the DB and then caches it.
        """
        self.split = split
        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading cached {split} dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                self.data_list = pickle.load(f)
        else:
            print(f"Loading QM9 {split} data from database...")
            with open(split_path, "r") as f:
                splits = json.load(f)
            indices = splits.get(split, [])
            self.data_list = []
            with connect(db_path) as db:
                for i, row in enumerate(db.select()):
                    if i in indices:
                        atoms = row.toatoms()
                        data_obj = ase_to_atomsdata(atoms, energy_property="energy_U0")
                        self.data_list.append(data_obj)
            if cache_file is not None:
                print(f"Caching {split} dataset to {cache_file}")
                with open(cache_file, "wb") as f:
                    pickle.dump(self.data_list, f)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

class QM9DataModule(pl.LightningDataModule):
    def __init__(self, db_path: str, split_path: str, batch_size: int = 32, cache_dir: str = "cache"):
        super().__init__()
        self.db_path = db_path
        self.split_path = split_path
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    def setup(self, stage: str = None):
        self.train_dataset = QM9Dataset(
            self.db_path, self.split_path, split="train",
            cache_file=os.path.join(self.cache_dir, "qm9_train.pkl")
        )
        self.val_dataset = QM9Dataset(
            self.db_path, self.split_path, split="validation",
            cache_file=os.path.join(self.cache_dir, "qm9_val.pkl")
        )
        self.test_dataset = QM9Dataset(
            self.db_path, self.split_path, split="test",
            cache_file=os.path.join(self.cache_dir, "qm9_test.pkl")
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: collate_graph_vae(batch, global_max=self.global_max),
            shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: collate_graph_vae(batch, global_max=self.global_max),
            shuffle=False
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: collate_graph_vae(batch, global_max=self.global_max),
            shuffle=False
        )

###############################################################################
# REACT data
###############################################################################


def _load_measurements(csv_path: str) -> pd.DataFrame:
    """
    Load and parse the custom measurements CSV, flatten multi-index columns,
    and return a numeric DataFrame.
    """
    # Read raw lines (adjust encoding if needed)
    with open(csv_path, 'r', encoding='utf-16') as f:
        lines = f.read().splitlines()

    # Skip metadata (first 4 lines)
    data_lines = lines[4:]
    # Split the next three lines into names, ids, and values
    split_lines = [data_lines[i].split('\t') for i in range(3)]
    names, ids, vals = split_lines

    # Construct multi-index DataFrame
    cols = pd.MultiIndex.from_arrays([names, ids])
    df = pd.DataFrame([vals], columns=cols)

    # Flatten and convert to numeric
    df_flat = df.copy()
    df_flat.columns = [f"{name}_{id}" for name, id in df.columns]
    df_numeric = df_flat.astype(float)
    return df_numeric


class MeasurementsMeshDataset(Dataset):
    """
    Dataset pairing measurement vectors with 3D meshes.
    Assumes that the CSV and the mesh files are aligned by filename or sorted order.
    """
    def __init__(
        self,
        csv_path: str,
        mesh_dir: str,
        mesh_ext: str = '.ply',
        transform_matrix: np.ndarray | None = None
    ):
        self.measurements = _load_measurements(csv_path)
        # List and sort mesh files
        self.mesh_paths = sorted(
            [os.path.join(mesh_dir, f)
             for f in os.listdir(mesh_dir)
             if f.lower().endswith(mesh_ext)]
        )
        assert len(self.measurements) == len(self.mesh_paths), \
            "Number of measurement rows must equal number of mesh files."
        self.transform = transform_matrix

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        # Measurements vector
        meas = torch.tensor(
            self.measurements.iloc[idx].values,
            dtype=torch.float32
        )
        # Load mesh with trimesh
        mesh = trimesh.load(self.mesh_paths[idx], process=False)
        if self.transform is not None:
            mesh.apply_transform(self.transform)

        # Convert mesh to torch Tensors
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)

        return meas, {'vertices': vertices, 'faces': faces}


class ReactDataModule(pl.LightningDataModule):
    """
    LightningDataModule for measurements + mesh data.

    Args:
        csv_path: Path to the measures.csv file
        mesh_dir: Directory containing mesh files
        batch_size: Batch size for DataLoader
        val_split: Fraction of data for validation (e.g. 0.2)
        test_split: Fraction of data for testing (e.g. 0.1)
        mesh_ext: File extension for meshes (default: .ply)
        transform_matrix: Optional 4x4 transform to apply to all meshes
        num_workers: Number of workers for DataLoader
    """
    def __init__(
        self,
        csv_path: str,
        mesh_dir: str,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        mesh_ext: str = '.ply',
        transform_matrix: np.ndarray | None = None,
        num_workers: int = 4
    ):
        super().__init__()
        self.csv_path = csv_path
        self.mesh_dir = mesh_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.mesh_ext = mesh_ext
        self.transform_matrix = transform_matrix
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        # Full dataset
        full = MeasurementsMeshDataset(
            csv_path=self.csv_path,
            mesh_dir=self.mesh_dir,
            mesh_ext=self.mesh_ext,
            transform_matrix=self.transform_matrix
        )
        n = len(full)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_val - n_test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full,
            [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

