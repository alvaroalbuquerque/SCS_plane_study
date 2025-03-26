import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def add_gaussian_noise(image=np.ones((256, 256)), mean=0, sigma=20):
    '''
    Add gaussian noise to an image.

    Parameters:
        image (numpy.ndarray(np.uint8)): Input image (default is a 256x256 array of ones)
        mean (float): Mean of the Gaussian noise distribution (default 0)
        sigma (float): Standard deviation of the Gaussian noise distribution (default 20)
    
    Returns:
        numpy.ndarray(np.float64): Noisy image
    '''
    # Check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # Check if dtype is uint8
    if image.dtype != np.uint8:
        raise ValueError("Image must be of dtype uint8")
    
    # Check if pixel values are in range [0, 255]
    if image.min() < 0 or image.max() > 255:
        raise ValueError("Pixel values must be in the range [0, 255]")
    
    row,col= image.shape
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

def add_speckle_noise(image=np.ones((256, 256)), mean=0, sigma=10):
    """
    Add speckle noise to an image.
    
    Speckle noise is a multiplicative noise that follows the equation:
    J = I + n*I = I(1 + n)
    where I is the input image and n is Gaussian noise with mean and sigma.
    
    Parameters:
    image (numpy.ndarray(np.uint8)): Input image (Intensities in range [0, 255])
    mean (float): Mean of the Gaussian distribution
    sigma (float): Standard deviation of the Gaussian distribution
    
    Returns:
    numpy.ndarray(np.uint8): Image with added speckle noise
    """

    # Check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # Check if dtype is uint8
    if image.dtype != np.uint8:
        raise ValueError("Image must be of dtype uint8")
    
    # Check if pixel values are in range [0, 255]
    if image.min() < 0 or image.max() > 255:
        raise ValueError("Pixel values must be in the range [0, 255]")
    # extracts shape
    row,col= image.shape
    # generates gaussian values
    gauss = np.random.normal(mean,sigma,(row,col))
    # makes sure the shape of the values match the image
    gauss = gauss.reshape(row,col)
    # generates noised image
    noisy = image + image * gauss
    return noisy

# def salt_and_pepper_noise_v2(salt_prob=0.3, pepper_prob=0.3, image=np.ones((256, 256))):
#     """
#     Add salt and pepper noise to an image.
    
#     Parameters:
#     image (PIL.Image): Input image
#     salt_prob (float): Probability of salt noise (white pixels), range [0, 1]
#     pepper_prob (float): Probability of pepper noise (black pixels), range [0, 1]
    
#     Returns:
#     PIL.Image: Image with added salt and pepper noise
#     """
#     # Input validation
#     if not 0 <= salt_prob <= 1 or not 0 <= pepper_prob <= 1:
#         raise ValueError("Probabilities must be between 0 and 1")
#     if salt_prob + pepper_prob > 1:
#         raise ValueError("Sum of probabilities must not exceed 1")
    
#     # Convert image to numpy array
#     img_array = np.array(image)
    
#     # Create copy of image
#     noised_image = img_array.copy()
    
#     # Generate random noise mask
#     mask = np.random.random(img_array.shape[:2])
    
#     # Add salt noise (white pixels)
#     salt_mask = mask < salt_prob
#     if len(img_array.shape) == 3:  # Color image
#         for i in range(img_array.shape[2]):
#             noised_image[salt_mask, i] = 255
#     else:  # Grayscale image
#         noised_image[salt_mask] = 255
    
#     # Add pepper noise (black pixels)
#     pepper_mask = (mask >= salt_prob) & (mask < salt_prob + pepper_prob)
#     if len(img_array.shape) == 3:  # Color image
#         for i in range(img_array.shape[2]):
#             noised_image[pepper_mask, i] = 0
#     else:  # Grayscale image
#         noised_image[pepper_mask] = 0
    
#     return noised_image

# def poisson_noise(image=np.ones((256, 256))):
#     vals = len(np.unique(image))
#     vals = 2 ** np.ceil(np.log2(vals))
#     noisy = np.random.poisson(image * vals) / float(vals)
#     return noisy


# def speckle_noise_gs(image=np.ones((256, 256))):
#     row,col = image.shape
#     gauss = np.random.randn(row,col)
#     gauss = gauss.reshape(row,col)        
#     noisy = image + image * gauss
#     return noisy

if __name__ == '__main__':
    try:
        print("Running noise_generator.py")
    except Exception as e:
        print("An exception occurred:\n\t\t{}".format(str(e)))