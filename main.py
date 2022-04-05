import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import torch.utils.data
# from torch.utils.data.sampler import SubsetRandomSample


class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample):
    super().__init__()
    if downsample:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
      self.shortcut = nn.Sequential()

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, input):
    shortcut = self.shortcut(input)
    input = nn.ReLU()(self.bn1(self.conv1(input)))
    input = nn.ReLU()(self.bn2(self.conv2(input)))
    input = input + shortcut
    return nn.ReLU()(input)

class ResNet18(nn.Module):
  def __init__(self, in_channels, resblock, outputs=1000):
    super().__init__()
    self.layer0 = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU()
    )

    self.layer1 = nn.Sequential(
      resblock(64, 64, downsample=False),
      resblock(64, 64, downsample=False)
    )

    self.layer2 = nn.Sequential(
      resblock(64, 128, downsample=True),
      resblock(128, 128, downsample=False)
    )

    self.layer3 = nn.Sequential(
      resblock(128, 256, downsample=True),
      resblock(256, 256, downsample=False)
    )


    self.layer4 = nn.Sequential(
      resblock(256, 512, downsample=True),
      resblock(512, 512, downsample=False)
    )

    self.gap = torch.nn.AdaptiveAvgPool2d(1)
    self.fc = torch.nn.Linear(512, outputs)

  def forward(self, input):
    input = self.layer0(input)
    input = self.layer1(input)
    input = self.layer2(input)
    input = self.layer3(input)
    input = self.layer4(input)
    input = self.gap(input)
    input = input.view(input.size(0),-1)
    input = self.fc(input)

    return input

  # TODO: untested
  def fit(self, criterion, optimizer, nb_epochs=5, trainloader=torch.utils.data.dataloader.DataLoader):
    total_step = len(trainloader)
    loss_list = []
    acc_list = []
    for epoch in range(nb_epochs):
      for i, (images, labels) in enumerate(trainloader):
        images, labels = images.cuda(), labels.cuda()
        outputs = self.forward(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
        if (i + 1) % 100 == 0:
          print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, nb_epochs, i + 1, total_step, loss.item(),
                     (correct / total) * 100))
  
      print('Epoch [{}/{}], Average accuracy: {:.2f}%'.format(epoch+1, nb_epochs, sum(acc_list)/len(acc_list)*100))
      acc_list = []

# from torchsummary import summary

resnet18 = ResNet18(3, ResBlock, outputs=1000)
if torch.cuda.is_available():
  print("CUDA available")
resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# summary(resnet18, (3, 224, 224))

# TODO: untested
batch_size = 150
calibration_size = 0.2
learning_rate = 0.001

transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

# Fetch cifar-10 dataset
trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
valset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)


num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(calibration_size * num_train))
calibration_idx = indices[:split]
calibration_sampler = torch.utils.data.SubsetRandomSampler(calibration_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)
calibration_loader = torch.utils.data.DataLoader(trainset, batch_size=len(calibration_sampler),sampler=calibration_sampler)


network = resnet18
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
network.cuda()
network.train()

print(criterion)
print(optimizer)
print(trainloader)
network.fit(criterion, optimizer, 10, trainloader)
