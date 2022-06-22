from networks.common import DoubleConv, Down

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=1, bias=True)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
