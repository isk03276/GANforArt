import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.dataset_manager import DatasetManager
from model.dcgan_generator import Generator
from model.dcgan_discriminator import Discriminator
import torchvision.utils as vutils

class DCGAN:
    def __init__(self):
        self.nz = 100
        self.image_size = 64
        self.nc = 3
        self.ngf = 64
        self.ndf = 64
        self.batch_size = 128
        self.dataset_path = "../../data/wikiart/"

        self.data_loader = DatasetManager(self.dataset_path,
                                          img_size=self.image_size,
                                          batch_size=self.batch_size)
        self.generator = Generator(nz=100, nc=self.nc, nf=self.ngf,
                                   image_size=self.image_size)
        self.discriminator = Discriminator()

    def test_generator(self):
        random_noise = torch.randn(self.batch_size, self.nz, 1, 1)
        with torch.no_grad():
            output = self.generator(random_noise)

        vutils.save_image(output[0].data, "test.png", normalize=True)

    def test_discriminator(self):
        pass

    def train(self):
        pass



if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.test_generator()
