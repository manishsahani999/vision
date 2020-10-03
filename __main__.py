from __future__ import print_function
import os
import os.path
import time
import json
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data as fryday_vision

from tqdm import tqdm
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

import datasets as fryday_dataset

BATCH_SIZE = 64


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.conv1 = nn.Conv2d(1, 10, 5)
#         self.conv2 = nn.Conv2d(10, 20, 5)
#         self.drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 64)  # 6*6 from image dimension
#         self.fc2 = nn.Linear(64, 10)


#     def forward(self, x, y):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax()


if __name__ == "__main__":
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    # )

    # train_loader = ds.mnist(
    #     os.path.abspath("./data/mnist"), train=True, transform=transform
    # )
    # test_loader = ds.mnist(
    #     os.path.abspath("./data/mnist"), train=False, transform=transform, batch_size=1000
    # )

    # examples = enumerate(train_loader)
    # batch_idx, (example_data, example_targets) = next(examples)

    # print(example_data.shape)
    # # print(example_targets)

    # network = Net()
    # # # print(network.parameters())

    # optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)

    # # train the network
    # for epoch in tqdm(range(5)):
    #     network.train()
    #     for batch_idx, (data, targets) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         output = network(data)
    #         loss = F.nll_loss(output, targets)
    #         loss.backward()
    #         optimizer.step()
    #         if batch_idx % 100 == 0 :print(
    #             "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
    #                 epoch,
    #                 batch_idx * len(data),
    #                 len(train_loader.dataset),
    #                 100.0 * batch_idx / len(train_loader),
    #                 loss.item(),
    #             )
    #         )
    #         torch.save(network.state_dict(), "./results/model.pth")
    #         torch.save(optimizer.state_dict(), "./results/optimizer.pth")

    # network.eval()
    # test_loss = 0
    # correct = 0

    # with torch.no_grad():
    #     for data, target in test_loader:
    #         output = network(data)
    #         test_loss += F.nll_loss(output, target, size_average=False).item()
    #         pred = output.data.max(1, keepdim=True)[1]
    #         correct += pred.eq(target.data.view_as(pred)).sum()
    # test_loss /= len(test_loader.dataset)
    # # test_losses.append(test_loss)

    # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    daquar_processed_paths = fryday_dataset.DaquarDataset()._paths

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = fryday_dataset.Daquar(daquar_processed_paths, transform)

    # Dataset loader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
    )

    # dataset_sizes = {x: len(train_loader[x]) for x in ['train', 'val']}

    # print(next(iter(train_loader)))
    # model.eval()
    # output = model(next(iter(train_loader)))
    # print(output)
    # for batch_idx, (v, q, a) in enumerate(train_loader):
    # for question in enumerate(q):
    #     for i, idx in enumerate(question):
    #         print(i)
    #     break
    # break

