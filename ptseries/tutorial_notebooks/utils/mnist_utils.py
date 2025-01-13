import torch
import torchvision
import matplotlib.pyplot as plt


def load_mnist(batch_size=32):
    """Returns dataloaders for MNIST dataset. If it is not already downloaded, downloads it to a data folder."""

    mnist_data = torchvision.datasets.MNIST("tutorial_notebooks/mnist_utils/data/", train=True, download=True)

    train_data = mnist_data.data.unsqueeze(1) / 256

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader


def plot_batch(batch_tensor, n_row=5):
    """Plot a batch of images. The batch must be a numpy array (batch_size x n_channels x height x width)"""

    grid_img = torchvision.utils.make_grid(torch.tensor(batch_tensor), nrow=n_row)
    plt.figure(figsize=(10, 8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
