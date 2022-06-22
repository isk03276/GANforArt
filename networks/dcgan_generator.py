import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.common import Up


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(in_channels = 100,
                                         out_channels= 1024,
                                         kernel_size = 4,
                                         stride = 1,
                                         padding = 0,
                                         bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        
        self.dconv2 = nn.ConvTranspose2d(in_channels = 1024,
                                         out_channels= 512,
                                         kernel_size = 4,
                                         stride = 2,
                                         padding = 1,
                                         bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.dconv3 = nn.ConvTranspose2d(in_channels = 512,
                                         out_channels= 256,
                                         kernel_size = 4,
                                         stride = 2,
                                         padding = 1,
                                         bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.dconv4 = nn.ConvTranspose2d(in_channels = 256,
                                         out_channels= 128,
                                         kernel_size = 4,
                                         stride = 2,
                                         padding = 1,
                                         bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.dconv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False)

    def forward(self, z):
        x = F.relu(self.bn1(self.dconv1(z)))
        x = F.relu(self.bn2(self.dconv2(x)))
        x = F.relu(self.bn3(self.dconv3(x)))
        x = F.relu(self.bn4(self.dconv4(x)))
        x = torch.tanh(self.dconv5(x))

        return x
