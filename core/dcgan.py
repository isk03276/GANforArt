import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.dataset_manager import DatasetManager
from model.dcgan_generator import Generator
from model.dcgan_discriminator import Discriminator
import torchvision.utils as vutils


class DCGAN:
    def __init__(self, nz=100, image_size=64, nc=3, ngf=64,
                 ndf=64, batch_size=128, epoch=100,
                 dataset_path="../../data/wikiart", lr=0.001):

        self.nz = nz
        self.image_size = image_size
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.epoch = epoch
        self.lr = lr

        self.data_loader = DatasetManager(self.dataset_path,
                                          img_size=self.image_size,
                                          batch_size=self.batch_size).dataset_loader
        self.generator = Generator(nz=self.nz, nc=self.nc, nf=self.ngf,
                                   image_size=self.image_size)
        self.discriminator = Discriminator(nc=self.nc, nf=self.ndf)

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.lr)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.lr)
        self.bce_loss_func = nn.BCELoss()

    def test_generator(self):
        random_noise = torch.randn(self.batch_size, self.nz, 1, 1)
        with torch.no_grad():
            fake_image = self.generator(random_noise)

        vutils.save_image(fake_image[0].data, "test.png", normalize=True)

    def test_discriminator(self):
        random_noise = torch.randn(self.batch_size, self.nz, 1, 1)
        with torch.no_grad():
            fake_image = self.generator(random_noise)
        d_value = self.discriminator(fake_image)

        print(d_value[0])

    def train(self):
        for e in range(self.epoch):
            for i, (real_images, labels) in enumerate(self.data_loader):
                # update discriminator
                random_noise = torch.randn(self.batch_size, self.nz, 1, 1)
                fake_images = self.generator(random_noise)
                fake_predicts = self.discriminator(fake_images).view(-1, 1)
                fake_labels = torch.zeros(self.batch_size, 1)
                fake_loss = self.bce_loss_func(fake_predicts, fake_labels)

                real_predicts = self.discriminator(real_images).view(-1, 1)
                real_labels = torch.ones(self.batch_size, 1)
                real_loss = self.bce_loss_func(real_predicts, real_labels)
                discriminator_loss = fake_loss + real_loss

                self.discriminator.zero_grad()
                discriminator_loss.backward()
                self.optimizer_d.step()

                # update generator
                generator_loss = self.bce_loss_func(fake_predicts, real_labels)

                self.generator.zero_grad()
                generator_loss.backward()
                self.optimizer_g.step()


if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train()
