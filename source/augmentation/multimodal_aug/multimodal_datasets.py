import numpy as np
import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt

from time import time
from torchvision import datasets, transforms
from torch import nn, optim

matplotlib.use('agg')
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

data_root = '/home/hqvo2/Datasets/MNIST'
trainset = datasets.MNIST(os.path.join(data_root, 'train_set'), download=True, train=True, transform=transform)
valset = datasets.MNIST(os.path.join(data_root, 'test_set'), download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()
one_hot_labels = nn.functional.one_hot(labels)

print(images.shape)
print(labels.shape)
print(one_hot_labels)


# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
# plt.savefig('./test.png')

# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
# plt.savefig('./test_2.png')

