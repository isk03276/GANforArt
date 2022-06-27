from typing import Union
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.torch import tensor_to_numpy_array
from utils.image import channel_first_to_last


class Monitor:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.fig = None
        self.ax = None
        self.plot_size = None
        self.plt_images = []
        
    def init_monitor(self, width, height):
        self.plot_size = math.ceil(math.sqrt(self.batch_size))
        self.fig, self.ax = plt.subplots(self.plot_size, self.plot_size, constrained_layout=True)
        temp_image = np.zeros((width, height))
        if self.batch_size == 1:
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)
            self.plt_images.append(self.ax.imshow(temp_image))
        else:
            for i in range(self.batch_size):
                row = i // self.plot_size
                col = i - row * self.plot_size
                self.ax[row][col].get_xaxis().set_visible(False)
                self.ax[row][col].get_yaxis().set_visible(False)
                plt_image = self.ax[row][col].imshow(temp_image)
                self.plt_images.append(plt_image)
        self.fig.show()
    
    def monitor_images(self, images:Union[torch.Tensor, np.ndarray], interval:int = 1):
        assert len(images.shape) == 4 # Batch_size X Width X Height X Channels
        assert images.shape[0] == self.batch_size
        
        if type(images) == torch.Tensor:
            images = tensor_to_numpy_array(images)
        images = channel_first_to_last(images)
        
        if self.fig is None or self.ax is None:
            self.init_monitor(images.shape[1], images.shape[2]) # W * H
            
        for i, image in enumerate(images):
            self.plt_images[i].set_data((image*255).astype(np.uint8))
            
        self.fig.canvas.draw_idle()
        plt.pause(interval)

            
            
        
    