from __future__ import print_function

import os
import time
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datasets as fd

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from datasets.vocabulary import Vocabulary
from net.daquar import Net

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
    daquar_processed_paths = fd.DaquarDataFolder().paths
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    print(daquar_processed_paths)
    dataset = fd.Daquar(daquar_processed_paths, transform)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
    )

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')


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

    # vocab = Vocabulary('hello')
    # vocab.add_sentence('the sentecne tihsniwspes sjs')
    # for i in range(vocab._num_words):
    #     print(vocab.to_word(i))

    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # dataset = fd.Daquar(daquar_processed_paths, transform)

    # # Dataset loader
    # train_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
    # )

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

