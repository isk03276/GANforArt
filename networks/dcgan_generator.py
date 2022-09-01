import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.common import Up


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(
            in_channels=100,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(256)

        self.dconv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.dconv3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.dconv4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.dconv5 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm5 = nn.BatchNorm2d(256)

        self.dconv6 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm6 = nn.BatchNorm2d(128)

        self.dconv7 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm7 = nn.BatchNorm2d(128)

        self.dconv8 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm8 = nn.BatchNorm2d(64)

        self.dconv9 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, z):
        x = F.leaky_relu(self.batch_norm1(self.dconv1(z)), inplace=True)
        x = F.leaky_relu(self.batch_norm2(self.dconv2(x)), inplace=True)
        x = F.leaky_relu(self.batch_norm3(self.dconv3(x)), inplace=True)
        x = F.leaky_relu(self.batch_norm4(self.dconv4(x)), inplace=True)
        x = F.leaky_relu(self.batch_norm5(self.dconv5(x)), inplace=True)
        x = F.leaky_relu(self.batch_norm6(self.dconv6(x)), inplace=True)
        x = F.leaky_relu(self.batch_norm7(self.dconv7(x)), inplace=True)
        x = F.leaky_relu(self.batch_norm8(self.dconv8(x)), inplace=True)
        x = torch.tanh(self.dconv9(x))
        return x
