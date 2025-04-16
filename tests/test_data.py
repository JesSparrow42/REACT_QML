import pytest
import torch
import os
import numpy as np
import datetime
import json
from ase import Atoms
from ase.db import connect
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from GenAI.data import MedicalDataModule, QM9DataModule, DICOMDataset, QM9Dataset

# ---------- Medical Data Module Test ----------

@pytest.fixture
def dummy_dicom_folders(tmpdir):
    ct_folder = tmpdir.mkdir("ct")
    pet_folder = tmpdir.mkdir("pet")

    def create_dummy_dicom(file_path):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = 'CT'
        ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.datetime.now().strftime('%H%M%S')

        # Required attributes for pixel data
        ds.Rows, ds.Columns = 128, 128
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        pixel_array = (np.random.rand(128, 128) * 255).astype(np.uint16)
        ds.PixelData = pixel_array.tobytes()
        ds.save_as(str(file_path))

    for i in range(10):
        create_dummy_dicom(ct_folder.join(f"{i:04d}.dcm"))
        create_dummy_dicom(pet_folder.join(f"{i:04d}.dcm"))

    return str(ct_folder), str(pet_folder)

def test_medical_datamodule_setup(dummy_dicom_folders):
    ct_folder, pet_folder = dummy_dicom_folders
    dm = MedicalDataModule(ct_folder=ct_folder, pet_folder=pet_folder, batch_size=2)
    dm.setup()
    assert len(dm.train_dataset) == 8
    assert len(dm.val_dataset) == 2

def test_medical_dataloader(dummy_dicom_folders):
    ct_folder, pet_folder = dummy_dicom_folders
    dm = MedicalDataModule(ct_folder=ct_folder, pet_folder=pet_folder, batch_size=2)
    dm.setup()

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    pet_batch, ct_batch = batch

    assert pet_batch.shape == (2, 1, 128, 128)
    assert ct_batch.shape == (2, 1, 128, 128)


# ---------- QM9 Data Module Test ----------
@pytest.fixture
def dummy_qm9_db(tmpdir):
    db_path = tmpdir.join("qm9_dummy.db")
    split_path = tmpdir.join("splits.json")
    cache_dir = tmpdir.mkdir("cache")

    # Create dummy ASE database with at least 5 minimal atom records
    with connect(str(db_path)) as db:
        for _ in range(5):
            atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
            db.write(atoms, data={"energy_U0": -76.4})

    # Create dummy splits file
    splits = {
        "train": [0, 1, 2],
        "validation": [3],
        "test": [4]
    }
    split_path.write(json.dumps(splits))

    return str(db_path), str(split_path), str(cache_dir)

def test_qm9_datamodule_setup(dummy_qm9_db):
    db_path, split_path, cache_dir = dummy_qm9_db
    dm = QM9DataModule(db_path=db_path, split_path=split_path, batch_size=2, cache_dir=cache_dir)
    dm.global_max = 5  # setting global max explicitly for test
    dm.setup()

    assert len(dm.train_dataset) == 3
    assert len(dm.val_dataset) == 1
    assert len(dm.test_dataset) == 1

def test_qm9_dataloader(dummy_qm9_db):
    db_path, split_path, cache_dir = dummy_qm9_db
    dm = QM9DataModule(db_path=db_path, split_path=split_path, batch_size=2, cache_dir=cache_dir)
    dm.global_max = 5
    dm.setup()

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    assert batch["node_features"].shape == (2, 5, 1)
    assert batch["node_positions"].shape == (2, 5, 3)
    assert batch["mask"].shape == (2, 5)
    assert batch["num_nodes"].shape == (2,)
