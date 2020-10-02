from __future__ import print_function
from data import download_daquar, Daquar
from torchvision import transforms
import os
import torch
import torchvision
import torch.nn as nn

BATCH_SIZE = 64

if __name__ == "__main__":

    files = download_daquar()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = Daquar(files["images"], transform)

    # Dataset loader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True
    )
    
    for idx, img in enumerate(train_loader):
        print(idx, img)
