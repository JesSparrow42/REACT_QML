#!/usr/bin/env python
import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from ase.db import connect
from ase import Atoms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

###############################
# Data Object Classes
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

###############################
# Graph VAE Model and Lightning Module
###############################

class GraphVAE(nn.Module):
    def __init__(self, latent_features: int, max_nodes: int, node_feature_dim: int = 1,
                 position_dim: int = 3, hidden_dim: int = 64):
        """
        A simple Graph VAE.
        """
        super().__init__()
        self.latent_features = latent_features
        self.max_nodes = max_nodes
        self.input_dim = node_feature_dim + position_dim  # e.g. 1+3 = 4
        # Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_features)
        self.fc_logvar = nn.Linear(hidden_dim, latent_features)
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * self.input_dim)
        )
    def encode(self, node_features, node_positions, mask):
        x = torch.cat([node_features, node_positions], dim=-1)  # (B, N, 4)
        B, N, _ = x.size()
        h = self.node_encoder(x.view(B * N, self.input_dim))
        h = h.view(B, N, -1)
        mask_expanded = mask.unsqueeze(-1)
        h = h * mask_expanded
        sum_h = h.sum(dim=1)
        counts = mask_expanded.sum(dim=1)
        pooled = sum_h / (counts + 1e-6)
        pooled = self.graph_encoder(pooled)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        B = z.size(0)
        out = self.decoder_fc(z)
        out = out.view(B, self.max_nodes, self.input_dim)
        return out
    def forward(self, node_features, node_positions, mask):
        mu, logvar = self.encode(node_features, node_positions, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class GraphVAE_Lightning(pl.LightningModule):
    def __init__(self, latent_features: int, max_nodes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphVAE(latent_features=latent_features, max_nodes=max_nodes)
        self.lr = lr
    def forward(self, batch):
        recon, mu, logvar = self.model(batch["node_features"], batch["node_positions"], batch["mask"])
        return recon, mu, logvar
    def compute_loss(self, recon, batch, mu, logvar):
        recon_feat = recon[..., :1]
        recon_pos = recon[..., 1:]
        mask = batch["mask"].unsqueeze(-1)
        loss_feat = F.mse_loss(recon_feat * mask, batch["node_features"] * mask, reduction="sum")
        loss_pos = F.mse_loss(recon_pos * mask, batch["node_positions"] * mask, reduction="sum")
        recon_loss = (loss_feat + loss_pos) / mask.sum()
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon_loss + kl_loss, recon_loss, kl_loss
    def training_step(self, batch, batch_idx):
        recon, mu, logvar = self.forward(batch)
        loss, recon_loss, kl_loss = self.compute_loss(recon, batch, mu, logvar)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        recon, mu, logvar = self.forward(batch)
        loss, _, _ = self.compute_loss(recon, batch, mu, logvar)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

###############################
# Main Training Code
###############################

def main():
    # Hard-coded parameters and paths
    db_path = r"/mnt/c/Users/JonatanEmilSvendsen/SparrowQML/data/qm9.db"
    split_path = r"/mnt/c/Users/JonatanEmilSvendsen/SparrowQML/data/randomsplits_110k_10k_rest.json"
    batch_size = 32
    latent_features = 16
    lr = 1e-3
    epochs = 50
    cache_dir = "cache"

    # Create and set up the data module (this loads/caches the dataset)
    data_module = QM9DataModule(db_path=db_path, split_path=split_path, batch_size=batch_size, cache_dir=cache_dir)
    data_module.setup()

    # Compute the global maximum number of nodes from the training dataset
    global_max = max([data.num_nodes for data in data_module.train_dataset.data_list])
    print("Using global_max =", global_max)
    # Save this value in the data module so that the dataloaders use it.
    data_module.global_max = global_max

    # Instantiate the Lightning model with the global max_nodes
    model = GraphVAE_Lightning(latent_features=latent_features, max_nodes=global_max, lr=lr)

    # Set up a progress bar callback (optional)
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # Create the Trainer
    trainer = Trainer(max_epochs=epochs, callbacks=[progress_bar], logger=True)
    trainer.fit(model, datamodule=data_module)
    print("Training finished.")

if __name__ == "__main__":
    main()
