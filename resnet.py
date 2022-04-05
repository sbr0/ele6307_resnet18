import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# resnet with ReLU changed to Softplus, and max pooling to average pooling

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
    input = nn.Softplus()(self.bn1(self.conv1(input)))
    input = nn.Softplus()(self.bn2(self.conv2(input)))
    input = input + shortcut
    return nn.Softplus()(input)

class ResNet18(nn.Module):
  def __init__(self, in_channels, resblock, outputs=1000):
    super().__init__()
    self.layer0 = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
      nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.Softplus()
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

  def fit(self, criterion, optimizer, scheduler, nb_epochs=5, trainloader=torch.utils.data.dataloader.DataLoader):
    self.train()
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

      curr_lr = optimizer.param_groups[0]['lr']
      print('Epoch [{}/{}], Average accuracy: {:.2f}%, LR: {:.3f}'.format(epoch+1, nb_epochs, sum(acc_list)/len(acc_list)*100, curr_lr))
      acc_list = []
      scheduler.step()

  def test_acc(self,valloader=torch.utils.data.dataloader.DataLoader):
    self.eval()
    acc_list = []
    for i, (images, labels) in enumerate(valloader):
      images, labels = images.cuda(), labels.cuda()
      outputs = self.forward(images)

      total = labels.size(0)
      _, predicted  = torch.max(outputs.data, 1)
      correct = (predicted == labels).sum().item()
      acc_list.append(correct / total)
    return sum(acc_list)/len(acc_list)
