import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.dataset_manager import DatasetManager
import torchvision.utils as vutils

from core.gan import GAN


class DCGAN(GAN):
    def __init__(self, nz=100, image_size=64, nc=3, ngf=64,
                 ndf=64, batch_size=128, epoch=10,
                 dataset_path="../../data/wikiart", lr=0.0002,
                 beta1=0.5, beta2=0.999):
        super(DCGAN, self).__init__(nz, image_size, nc, ngf, ndf)
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.epoch = epoch
        self.lr = lr
        self.save_dir = '../results/'
        self.beta1 = beta1
        self.beta2 = beta2

        self.data_loader = DatasetManager(self.dataset_path,
                                          img_size=self.image_size,
                                          batch_size=self.batch_size).dataset_loader

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.lr, betas=(self.beta1, self.beta2))
        self.bce_loss_func = nn.BCELoss()

    def test_generator(self, save_dir, i):
        random_noise = torch.randn(self.batch_size, self.nz, 1, 1)
        with torch.no_grad():
            fake_image = self.generator(random_noise)

        vutils.save_image(fake_image[0].data,
                          self.save_dir+"generator_test_{}.png".format(i),
                          normalize=True)

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

                # update
                random_noise = torch.randn(self.batch_size, self.nz, 1, 1)
                fake_images = self.generator(random_noise)
                fake_predicts = self.discriminator(fake_images).view(-1, 1)
                generator_loss = self.bce_loss_func(fake_predicts, real_labels)

                self.generator.zero_grad()
                generator_loss.backward()
                self.optimizer_g.step()
                print("gogo!", i)
            self.test_generator(e)
            self.save_dir("epoch_{}".foramt(e))


if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.train()
