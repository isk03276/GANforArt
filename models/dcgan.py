import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from networks.dcgan_generator import Generator
from networks.dcgan_discriminator import Discriminator


class DCGAN(BaseModel):
    def __init__(self, batch_size, lr=0.0002, beta1=0.5, beta2=0.999):
        self.batch_size = batch_size
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        super().__init__()
        
    def _init(self):
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.lr, betas=(self.beta1, self.beta2))
        self.loss_func = nn.BCELoss()
        
    
    def get_discriminator_loss(self, real_images, fake_images):
        real_images_scores = self.discriminate_images(real_images)
        real_labels = torch.ones(self.batch_size, 1).to(self.device)
        fake_images_scores = self.discriminate_images(fake_images)
        fake_labels = torch.zeros(self.batch_size, 1).to(self.device)
        
        real_loss = self.loss_func(real_images_scores, real_labels)
        fake_loss = self.loss_func(fake_images_scores, fake_labels)
        
        loss = real_loss + fake_loss
        return loss
    
    def get_generator_loss(self, fake_images:torch.Tensor):
        fake_images_scores = self.discriminate_images(fake_images)
        real_labels = torch.ones(self.batch_size, 1).to(self.device)
        
        loss = self.loss_func(fake_images_scores, real_labels)
        return loss
    