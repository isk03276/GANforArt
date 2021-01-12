import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, nc, nf, output_num):
        super(Discriminator, self).__init__()
        self.output_num = output_num

        self.conv1 = nn.Conv2d(nc, 32, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.leakyRelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(65)
        self.leakyRelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.leakyRelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.leakyRelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1, bias=True)
        self.bn5 = nn.BatchNorm2d(512)
        self.leakyRelu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv2d(512, 512, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.leakyRelu6 = nn.LeakyReLU(0.2, inplace=True)

        self.dr_fc1 = nn.Linear(512 * 4 * 4, 1)

        self.dc_fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.leakyRelu7 = nn.LeakyReLU(0.2, inplace=True)
        self.dc_fc2 = nn.Linear(1024, 512)
        self.leakyRelu8 = nn.LeakyReLU(0.2, inplace=True)
        self.dc_fc3 = nn.Linear(512, self.output_num)

    def forward(self, x):
        x = self.leakyRelu1(self.bn1(self.conv1(x)))
        x = self.leakyRelu2(self.bn2(self.conv2(x)))
        x = self.leakyRelu3(self.bn3(self.conv3(x)))
        x = self.leakyRelu4(self.bn4(self.conv4(x)))
        x = self.leakyRelu5(self.bn5(self.conv5(x)))
        x = self.leakyRelu6(self.bn6(self.conv6(x)))

        dr = torch.sigmoid(self.dr_fc1(x))

        dc = self.leakyRelu7(self.dc_fc1(x))
        dc = self.leakyRelu8(self.dc_fc2(dc))
        dc = torch.softmax(self.dc_fc3(dc))

        return dr, dc
