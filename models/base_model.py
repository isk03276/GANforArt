import os
from abc import ABC, abstractmethod

import torch

from utils.torch import optimize


class BaseModel(ABC):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.discriminator = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.loss_function = None
        self._init()
        
    @abstractmethod
    def _init(self):
        pass
    
    @abstractmethod
    def get_discriminator_loss(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        pass

    @abstractmethod
    def get_generator_loss(self, fake_images: torch.Tensor):
        pass
        
    def generate_random_noise(self, noise_size:int = 100):
        return torch.randn(self.batch_size, noise_size, 1, 1).to(self.device)
    
    def generate_fake_images(self):
        random_noise = self.generate_random_noise()
        fake_images = self.generator(random_noise)
        return fake_images
    
    def discriminate_images(self, images:torch.Tensor):
        scores = self.discriminator(images).view(-1, 1)
        return scores
    
    def train_generator(self, fake_images:torch.Tensor):
        loss = self.get_generator_loss(fake_images)
        optimize(self.generator, self.optimizer_g, loss)
        return loss
    
    def train_discriminator(self, real_images:torch.Tensor, fake_images:torch.Tensor):
        real_images = real_images.to(self.device)
        loss = self.get_discriminator_loss(real_images, fake_images.detach())
        optimize(self.discriminator, self.optimizer_d, loss)
        return loss
    
    def save_model(self, path:str, file_name:str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict()
        }, path + file_name)
    
    def load_model(self, path:str, load_only_generator:bool=False):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        if not load_only_generator:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d']) 
