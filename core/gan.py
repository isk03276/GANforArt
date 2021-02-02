import torch


class GAN:
    def __init__(self, nz, image_size, nc, ngf, ndf, model="dcgan"):
        self.nz = nz
        self.image_size = image_size
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if model == "dcgan":
            from model.dcgan_generator import Generator
            from model.dcgan_discriminator import Discriminator
            self.generator = Generator(nz=self.nz, nc=self.nc,
                                       nf=self.ngf, image_size=self.image_size).to(self.device)
            self.discriminator = Discriminator(nc=self.nc, nf=self.ndf).to(self.device)
        elif model == "can":
            from model.dcgan_generator import Generator #### 수정해야 함
            from model.can_discriminator import Discriminator
            self.generator = Generator(nz=self.nz, nc=self.nc,
                                       nf=self.ngf, image_size=self.image_size).to(self.device)
            self.discriminator = Discriminator(nc=self.nc, nf=self.ndf).to(self.device)

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