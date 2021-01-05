import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, nz=100, nc=3, nf=128, image_size=64):
        super(Generator, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(nz, nf*8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(nf*8)

        self.dconv2 = nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nf*4)

        self.dconv3 = nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nf*2)

        self.dconv4 = nn.ConvTranspose2d(nf*2, nf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nf)

        self.donv5 = nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False)

    def forward(self, z):
        x = F.relu(self.bn1(self.dconv1(z)))
        x = F.relu(self.bn2(self.dconv2(x)))
        x = F.relu(self.bn3(self.dconv3(x)))
        x = F.relu(self.bn4(self.dconv4(x)))
        x = F.relu(self.dconv5(x))

        return x
