import torch
import torchvision

def mnist(path, transform, train=True, download=True, batch_size=64, shuffle=True):
    """
    download mnist dataset at {path}
    """
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=train, download=download, transform=transform),
        batch_size=batch_size, shuffle=shuffle
    )