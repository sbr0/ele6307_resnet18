import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
from torchvision import datasets, transforms
import torch.utils.data

import resnet # resnet with ReLU changed to Softplus, and max pooling to average pooling

batch_size = 150
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
epochs = 164


transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Fetch cifar-10 dataset
trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
valset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)


network = resnet.ResNet18(3, resnet.ResBlock, outputs=1000)
network.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[80, 121], gamma=0.1)

network.fit(criterion, optimizer, scheduler, nb_epochs=epochs, trainloader=trainloader)
print('Test accuracy: {:.2f}%'.format(network.test_acc(valloader)*100))
