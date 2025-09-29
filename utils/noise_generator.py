import numpy as np

def add_gaussian_noise(image=np.ones((256, 256)), mean=0, sigma=20):
    '''
    Add gaussian noise to an image.

    Parameters:
        image (numpy.ndarray(np.float64)): Input image (default is a 256x256 array of ones)
        mean (float): Mean of the Gaussian noise distribution (default 0)
        sigma (float): Standard deviation of the Gaussian noise distribution (default 20)
    
    Returns:
        numpy.ndarray(np.float64): Noisy image
    '''
    # check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # check if dtype is float64
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    # check if pixel values are in range [0, 1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    # extract the shape
    row,col= image.shape
    # creates the gaussian distribution
    gauss = np.random.normal(mean,sigma,(row,col))
    # adds the noise
    noisy = image + gauss
    # extracts the min and max of the noised image
    noisy_min = noisy.min()
    noisy_max = noisy.max()    
    # normalize to [0, 1]
    normalized = (noisy - noisy_min) / (noisy_max - noisy_min)
    return normalized

def add_speckle_noise(image=np.ones((256, 256)), mean=0, sigma=0.2):
    """
    Add speckle noise to an image.

    Parameters:
    image (numpy.ndarray(np.float64)): Input image (Intensities in range [0, 1])
    mean (float): Mean of the Gaussian distribution
    sigma (float): Standard deviation of the Gaussian distribution
    
    Returns:
    numpy.ndarray(np.float64): Image with added speckle noise
    """
    # check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # check if dtype is float64
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    # check if pixel values are in range [0, 1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    # extracts shape
    row,col= image.shape
    # generates gaussian values
    gauss = np.random.normal(mean,sigma,(row,col))
    # makes sure the shape of the values match the image
    gauss = gauss.reshape(row,col)
    # generates noised image
    noisy = image + image * gauss
    # extracts the min and max of the noised image
    noisy_min = noisy.min()
    noisy_max = noisy.max()    
    # normalize to [0, 1]
    normalized = (noisy - noisy_min) / (noisy_max - noisy_min)
    return normalized

def add_sp_noise(image=np.ones((256, 256)), salt_prob=0.3, pepper_prob=0.3):
    """
    Add salt and pepper noise to an image.
    
    Parameters:
    image (numpy.ndarray(np.float64)): Input image (Intensities in range [0, 1])
    salt_prob (float): Probability of salt noise (white pixels), range [0, 1]
    pepper_prob (float): Probability of pepper noise (black pixels), range [0, 1]
    
    Returns:
    numpy.ndarray(np.float64): Image with added salt and pepper noise
    """
    # check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # check if dtype is float64
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    # check if pixel values are in range [0, 1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    
    # input validation
    if not 0 <= salt_prob <= 1 or not 0 <= pepper_prob <= 1:
        raise ValueError("Probabilities must be between 0 and 1")
    if salt_prob + pepper_prob > 1:
        raise ValueError("Sum of probabilities must not exceed 1")
    
    # create copy of image
    noisy = image.copy()
    
    # generate random noise mask
    mask = np.random.random(image.shape[:2])
    
    # add salt noise (white pixels)
    salt_mask = mask < salt_prob
    # rgb image
    if len(image.shape) == 3:  
        for i in range(image.shape[2]):
            noisy[salt_mask, i] = 1
    else:  
        # grayscale image
        noisy[salt_mask] = 1
    
    # add pepper noise (black pixels)
    pepper_mask = (mask >= salt_prob) & (mask < salt_prob + pepper_prob)
    # rgb image
    if len(image.shape) == 3:  
        for i in range(image.shape[2]):
            noisy[pepper_mask, i] = 0
    else:  
        # grayscale image
        noisy[pepper_mask] = 0
    
    return noisy

def add_poisson_noise(image=np.ones((256, 256)), factor=1.0):
    """
    Add poisson noise to an image.
    
    Parameters:
    image (numpy.ndarray(np.float64)): Input image (Intensities in range [0, 1])
    factor (float): Scaling factor to control noise intensity. Higher values produce more noise. 
                    Factor=0 produces no noise. (default 1.0)
    
    Returns:
    numpy.ndarray(np.float64): Image with added poisson noise
    """
    # Check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # Check if dtype is float64
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    # Check if pixel values are in range [0, 1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    
    # Special case: if factor=0, return original image without noise
    if factor == 0:
        return image.copy()
    
    # Scale image to adjust noise level (higher values = more noise)
    # Use the original image as lambda values for Poisson distribution
    # Higher factor = more variation from original value
    
    # Generate Poisson noise
    # For each pixel value λ, generate a random value from Poisson(λ)
    noisy = np.random.poisson(image * 255.0 / factor) * factor / 255.0
    
    # Clip values to valid range
    noisy = np.clip(noisy, 0.0, 1.0)
    
    return noisy

def add_poisson_noise_normalized(image=np.ones((256, 256)), factor=1.0):
    """
    Add poisson noise to an image.
    
    Parameters:
    image (numpy.ndarray(np.float64)): Input image (Intensities in range [0, 1])
    factor (float): Scaling factor to control noise intensity. Higher values produce more noise. 
                    Factor=0 produces no noise. (default 1.0)
    
    Returns:
    numpy.ndarray(np.float64): Image with added poisson noise
    """
    # Check if input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # Check if dtype is float64
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    # Check if pixel values are in range [0, 1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    
    # Special case: if factor=0, return original image without noise
    if factor == 0:
        return image.copy()
    
    # Scale image to adjust noise level (higher values = more noise)
    # Use the original image as lambda values for Poisson distribution
    # Higher factor = more variation from original value
    
    # Generate Poisson noise
    # For each pixel value λ, generate a random value from Poisson(λ)
    noisy = np.random.poisson(image * 255.0 / factor) * factor / 255.0
    
    # extracts the min and max of the noised image
    noisy_min = noisy.min()
    noisy_max = noisy.max()    
    # normalize to [0, 1]
    normalized = (noisy - noisy_min) / (noisy_max - noisy_min)
    
    return normalized


def add_gaussian_noise_0to1(image=np.ones((256, 256)), intensity=0.5):
    '''
    Add gaussian noise to an image with intensity parameter from 0 to 1.

    Parameters:
        image (numpy.ndarray(np.float64)): Input image (default is a 256x256 array of ones)
        intensity (float): Noise intensity from 0 (no noise) to 1 (maximum noise), range [0, 1]
    
    Returns:
        numpy.ndarray(np.float64): Noisy image
    '''
    # Input validation
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    
    if not 0 <= intensity <= 1:
        raise ValueError("Intensity must be between 0 and 1")
    
    # Define min and max sigma values for gaussian noise
    sigma_min = 0.0
    sigma_max = 1.0
    
    # Map intensity to sigma parameter
    sigma = sigma_min + intensity * (sigma_max - sigma_min)
    
    # Call the original function
    return add_gaussian_noise(image, mean=0, sigma=sigma)

def add_speckle_noise_0to1(image=np.ones((256, 256)), intensity=0.5):
    """
    Add speckle noise to an image with intensity parameter from 0 to 1.

    Parameters:
    image (numpy.ndarray(np.float64)): Input image (Intensities in range [0, 1])
    intensity (float): Noise intensity from 0 (no noise) to 1 (maximum noise), range [0, 1]
    
    Returns:
    numpy.ndarray(np.float64): Image with added speckle noise
    """
    # Input validation
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    
    if not 0 <= intensity <= 1:
        raise ValueError("Intensity must be between 0 and 1")
    
    # Define min and max sigma values for speckle noise
    sigma_min = 0.0
    sigma_max = 1.0
    
    # Map intensity to sigma parameter
    sigma = sigma_min + intensity * (sigma_max - sigma_min)
    
    # Call the original function
    return add_speckle_noise(image, mean=0, sigma=sigma)

def add_sp_noise_0to1(image=np.ones((256, 256)), intensity=0.5):
    """
    Add salt and pepper noise to an image with intensity parameter from 0 to 1.
    
    Parameters:
    image (numpy.ndarray(np.float64)): Input image (Intensities in range [0, 1])
    intensity (float): Noise intensity from 0 (no noise) to 1 (maximum noise), range [0, 1]
    
    Returns:
    numpy.ndarray(np.float64): Image with added salt and pepper noise
    """
    # Input validation
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    
    if not 0 <= intensity <= 1:
        raise ValueError("Intensity must be between 0 and 1")
    
    # Define min and max probability values for salt and pepper noise
    prob_min = 0.0
    prob_max = 0.5  # Maximum of 0.5 for each (salt + pepper = 1.0 max)
    
    # Map intensity to probability parameters
    prob = prob_min + intensity * (prob_max - prob_min)
    
    # Call the original function with equal salt and pepper probabilities
    return add_sp_noise(image, salt_prob=prob, pepper_prob=prob)

def add_poisson_noise_0to1(image=np.ones((256, 256)), intensity=0.5):
    """
    Add poisson noise to an image with intensity parameter from 0 to 1.
    
    Parameters:
    image (numpy.ndarray(np.float64)): Input image (Intensities in range [0, 1])
    intensity (float): Noise intensity from 0 (no noise) to 1 (maximum noise), range [0, 1]
    
    Returns:
    numpy.ndarray(np.float64): Image with added poisson noise
    """
    # Input validation
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if image.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Pixel values must be in the range [0, 1]")
    
    if not 0 <= intensity <= 1:
        raise ValueError("Intensity must be between 0 and 1")
    
    # Define min and max factor values for poisson noise
    factor_min = 0.0    # Very small value instead of 0 to avoid division issues
    factor_max = 20.0
    
    # Map intensity to factor parameter
    factor = factor_min + intensity * (factor_max - factor_min)
    
    # Call the original function
    return add_poisson_noise(image, factor=factor)



if __name__ == '__main__':
    try:
        print("Running noise_generator.py")
    except Exception as e:
        print("An exception occurred:\n\t\t{}".format(str(e)))