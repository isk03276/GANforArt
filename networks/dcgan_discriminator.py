from networks.common import DoubleConv, Down

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        x = F.leaky_relu(self.batch_norm1(self.conv1(x)), 0.2, True)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.batch_norm3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.batch_norm4(self.conv4(x)), 0.2, True)
        x = F.leaky_relu(self.batch_norm5(self.conv5(x)), 0.2, True)
        x = F.leaky_relu(self.batch_norm6(self.conv6(x)), 0.2, True)
        x = F.leaky_relu(self.batch_norm7(self.conv7(x)), 0.2, True)
        x = F.leaky_relu(self.batch_norm8(self.conv8(x)))
        x = torch.sigmoid(self.conv9(x))
        return x
