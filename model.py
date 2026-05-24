import torch.nn as nn
import torch


class ResNetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.batchNorm1 = nn.BatchNorm2d(num_features=out_channels)
    self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
    self.batchNorm2 = nn.BatchNorm2d(num_features=out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.shortcut = nn.Identity()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_channels=in_channels, out_channels=out_channels,
              kernel_size=1, stride=stride, bias=False
          ),
          nn.BatchNorm2d(num_features=out_channels)
      )

  def forward(self, x):
    out = self.conv1(x)
    out = self.batchNorm1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.batchNorm2(out)
    out = out + self.shortcut(x)
    return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
      super().__init__()

      self.in_channels = 64

      self.conv1 = nn.Conv2d(
        3, 64, kernel_size=7, stride=2, padding=3, bias=False
      )
      self.bn1 = nn.BatchNorm2d(64)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      self.layer1 = self._make_layer(64,  2, stride=1)
      self.layer2 = self._make_layer(128, 2, stride=2)
      self.layer3 = self._make_layer(256, 2, stride=2)
      self.layer4 = self._make_layer(512, 2, stride=2)

      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
      layers = []

      layers.append(ResNetBlock(self.in_channels, out_channels, 3, stride))
      self.in_channels = out_channels

      for _ in range(1, num_blocks):
          layers.append(ResNetBlock(out_channels, out_channels, 3))

      return nn.Sequential(*layers)

    def forward(self, x):
      x = self.relu(self.bn1(self.conv1(x)))
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)

      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)

      return x