from torch.utils.data import Dataset, DataLoader
from random import randrange


class SimpleDataset(Dataset):
    """Class used to format data and labels as a PyTorch Dataset

    See PyTorch documentation for more info.

    Args:
        data (torch.tensor): PyTorch tensor of shape (n_data, dim_data)
        labels (torch.tensor): PyTorch tensor of shape (n_data, 1)
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = randrange(len(self.data))
        return self.data[idx], self.labels[idx]


def create_dataloader(data, labels, batch_size=32, shuffle=True):
    """Creates a dataloader for the data and labels

    Args:
        data (torch.tensor): PyTorch tensor of shape (n_data, dim_data)
        labels (torch.tensor): PyTorch tensor of shape (n_data, 1)
        batch_size (int, optional): batch size
        shuffle (bool, optional): whether to shuffle the data

    Returns:
        PyTorch DataLoader for the data and labels
    """

    dataset = SimpleDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
