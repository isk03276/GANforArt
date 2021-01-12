import torch
from model.dcgan_generator import Generator
from model.dcgan_discriminator import Discriminator

class GAN:
    def __init__(self, nz, image_size, nc, ngf, ndf):
        self.nz = nz
        self.image_size = image_size
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf

        self.generator = Generator(nz=self.nz, nc=self.nc, nf=self.ngf,
                                   image_size=self.image_size)
        self.discriminator = Discriminator(nc=self.nc, nf=self.ndf)

    def save(self, file_name):
        torch.save({
            'generator' : self.generator.state_dict(),
            'discriminator' : self.discriminator.state_dict(),
            'optimizer_g' : self.optimizer_g.state_dict(),
            'optimizer_d' : self.optimizer_d.state_dict()
        }, self.save_dir+file_name)

    def load(self, file_name):
        torch.load({
            'generator' : self.generator.state_dict(),
            'discriminator' : self.discriminator.state_dict(),
            'optimizer_g' : self.optimizer_g.state_dict(),
            'optimizer_d' : self.optimizer_d.state_dict()
        }, self.save_dir+file_name)