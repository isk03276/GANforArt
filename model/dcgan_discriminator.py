import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.module):
    def __init__(self, nc, nf):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(nc, nf, 4, 2, 1, bias=False)
        #self.bn1

        self.conv2 = nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nf*2)

        self.conv3 = nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nf*4)

        self.conv4 = nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nf*8)

        self.conv5 = nn.Conv2d(nf*8, 1, 4, 1, 0, bias=True)

    def forward(self, x):
        x = nn.LeakyReLU(self.conv1(x), 0.2, inplace=True)
        x = nn.LeakyReLU(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x = nn.LeakyReLU(self.bn3(self.conv3(x)), 0.2, inplace=True)
        x = nn.LeakyReLU(self.bn4(self.conv4(x)), 0.2, inplace=True)
        x = F.sigmoid(self.conv5(x))

        return x