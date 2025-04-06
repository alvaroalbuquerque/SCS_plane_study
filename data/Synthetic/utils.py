import numpy as np

def uniform_image(intensity=0.5, image_size=(256, 256)):
    return np.ones(image_size, dtype=np.float64)*intensity