import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, nc, nf):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(nc, nf, 4, 2, 1, bias=False)
        self.leakyRelu1 = nn.LeakyReLU(0.2, inplace=True)
        #self.bn1

        self.conv2 = nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nf*2)
        self.leakyRelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nf*4)
        self.leakyRelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nf*8)
        self.leakyRelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(nf*8, 1, 4, 1, 0, bias=True)

    def forward(self, x):
        x = self.leakyRelu1(self.conv1(x))
        x = self.leakyRelu2(self.bn2(self.conv2(x)))
        x = self.leakyRelu3(self.bn3(self.conv3(x)))
        x = self.leakyRelu3(self.bn4(self.conv4(x)))
        x = F.sigmoid(self.conv5(x))

        return x