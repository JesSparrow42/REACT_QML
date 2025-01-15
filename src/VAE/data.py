import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_loader import create_data_loader  # assuming this returns a Dataset

class MedicalDataModule(pl.LightningDataModule):
    def __init__(self, ct_folder, pet_folder, batch_size=64, num_workers=4, augment=False):
        super().__init__()
        self.ct_folder = ct_folder
        self.pet_folder = pet_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

    def setup(self, stage=None):
        # Called on every GPU separately; you can use this to split data if needed.
        # Here, we assume a single dataset returned by create_data_loader.
        self.dataset = create_data_loader(
            ct_folder=self.ct_folder,
            pet_folder=self.pet_folder,
            num_workers=self.num_workers,
            augment=self.augment
        )

    def train_dataloader(self):
        # Create and return training DataLoader
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,             # Shuffle data for training
            num_workers=self.num_workers
        )
