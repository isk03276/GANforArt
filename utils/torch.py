import numpy as np
import torch


def optimize(model, optimizer, loss):
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
def tensor_to_numpy_array(tensor:torch.Tensor)-> np.ndarray:
    return tensor.detach().cpu().numpy()