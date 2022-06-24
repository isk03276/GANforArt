import numpy as np

def channel_first_to_last(image: np.ndarray)-> np.ndarray:
    if len(image.shape) == 4:
        return np.moveaxis(image, 1, 3)
    elif len(image.shape) == 3:    
        return np.moveaxis(image, 0, 2)
    else:
        raise IndexError

def channel_last_to_first(image: np.ndarray)-> np.ndarray:
    if len(image.shape) == 4:
        return np.moveaxis(image, 3, 1)
    elif len(image.shape) == 3:    
        return np.moveaxis(image, 2, 0)
    else:
        raise IndexError