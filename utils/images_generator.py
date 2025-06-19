import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def generate_histogram(img, isNormalized=True, title='Histogram of Img', range=(0, 255)):
    '''
        Plots a histogram of the image
    '''
    if not isNormalized:
        normalized_img = normalize_img_to_0_255(img)
    else:
        normalized_img = img
    flattened_img = normalized_img.flatten()
    plt.hist(flattened_img, bins=256, range=range, color='blue', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)

def normalize_img_to_0_255(img):
    '''
        Normalizes between 0 and 255
    '''
    img_norm = img - img.min()
    img_norm /= img_norm.max()
    img_norm *= 255
    return img_norm

def uniform_image(intensity=127, image_size=(256, 256)):
    return np.ones(image_size, dtype=np.int8)*intensity

def colored_noise(k, image_size=(256, 256)):

    # Uniform white noise
    white_noise = np.random.uniform(-0.5, 0.5, image_size)
    
    # Applies the Fourier transform and shifts low frequencies to the center
    y = np.fft.fftshift(np.fft.fft2(white_noise))

    ''' "matriz de frequência": 
    No caso 1-D, esta seria uma matriz das frequências reais que correspondem às amplitudes 
    dadas pela transformada. 
    No caso 2-D, esta é a distância do centro do nosso espaço de Fourier deslocado, 
    pois quanto mais longe vamos das bordas, maior será a frequência capturada naquele ponto'''
    _x, _y = np.mgrid[0:y.shape[0], 0:y.shape[1]]
    f = np.hypot(_x - y.shape[0] / 2, _y - y.shape[1] / 2)

    # ruido modificado
    y_2 = y / f**(k/2)

    colored_noise = np.nan_to_num(y_2, nan=0, posinf=0, neginf=0)


    # Retira o deslocamento e calcula a inversa da transformada 
    colored_noise = np.fft.ifft2(np.fft.ifftshift(colored_noise)).real

    
    # Normalize to [0, 1] range
    min_val = colored_noise.min()
    max_val = colored_noise.max()
    normalized_noise = (colored_noise - min_val) / (max_val - min_val)
    

    return colored_noise

def logistic_map(x0, r, num_iterations, image_size=(256, 256)):
    '''
        r in [0, 4]
        x0 in [0, 1]

        For  0 < 𝑟 < 1, the population will eventually die out, converging to 0.
        For  1 < 𝑟 < 3, the population will stabilize at a fixed point.
        For  3 < 𝑟 < 3.57, the system exhibits periodic behavior.
        For 𝑟 > 3.57, the system shows chaotic behavior, though there are some windows of periodicity.
    '''    
    width, height = image_size
    image = np.zeros(image_size)

    x = x0
    for i in range(num_iterations):
        x = r * x * (1 - x)

    for i in range(width):
        for j in range(height):
            x = r * x * (1 - x)
            image[i, j] = x

    return image

def henon_map(a, b, x0, y0, num_iterations, image_size=(256, 256)):
    """
    Generates a 2D array of Hénon map data.
    For values of a = 1.4 and b = 0.3 it presents chaotic behaviour

    Parameters:
    a (float): The fixed a value.
    b (float): The fixed b value.
    x0 (float): The initial x value.
    y0 (float): The initial y value.

    Returns:
    np.ndarray: A henon map image.
    """
    width, height = image_size
    image_y = np.zeros(image_size)
    image_x = np.zeros(image_size)

    x, y = x0, y0
    # Run through the initial iterations to reach long-term behavior
    for _ in range(num_iterations):
        x_next = 1 - a * x**2 + y
        y_next = b * x
        x, y = x_next, y_next

    for i in range(width):
        for j in range(height):
            x_next = 1 - a * x**2 + y
            y_next = b * x
            x, y = x_next, y_next
            image_y[i, j] = y
            image_x[i, j] = x

    return image_x, image_y

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    
def salt_and_pepper(salt_prob=0.3, pepper_prob=0.3, image_size=(256, 256)):
    """
    Generates an image with salt and pepper noise.

    Parameters:
    height (int): Height of the image.
    width (int): Width of the image.
    salt_prob (float): Probability of a pixel being salt (white).
    pepper_prob (float): Probability of a pixel being pepper (black).

    Returns:
    np.ndarray: The generated salt and pepper image.
    """
    image = np.random.choice(
        [0, 255, 127],
        size=image_size,
        p=[pepper_prob, salt_prob, 1 - salt_prob - pepper_prob]
    ).astype(np.uint8)
    return image

def salt_and_pepper_noise(salt_prob=0.3, pepper_prob=0.3, image=np.ones((256, 256))):
    """
    Adds salt and pepper noise to an existing image using NumPy.

    Parameters:
    image (np.ndarray): Input image.
    salt_prob (float): Probability of a pixel being salt (white).
    pepper_prob (float): Probability of a pixel being pepper (black).

    Returns:
    np.ndarray: The image with salt and pepper noise added.
    """
    
    # Ensure the image is in the correct format (values between 0 and 255 for an 8-bit image)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    noisy_image = np.copy(image)

    # Salt noise
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255
    
    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[tuple(pepper_coords)] = 0

    return noisy_image


def salt_and_pepper_noise_v2(salt_prob=0.3, pepper_prob=0.3, image=np.ones((256, 256))):
    """
    Add salt and pepper noise to an image.
    
    Parameters:
    image (PIL.Image): Input image
    salt_prob (float): Probability of salt noise (white pixels), range [0, 1]
    pepper_prob (float): Probability of pepper noise (black pixels), range [0, 1]
    
    Returns:
    PIL.Image: Image with added salt and pepper noise
    """
    # Input validation
    if not 0 <= salt_prob <= 1 or not 0 <= pepper_prob <= 1:
        raise ValueError("Probabilities must be between 0 and 1")
    if salt_prob + pepper_prob > 1:
        raise ValueError("Sum of probabilities must not exceed 1")
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create copy of image
    noised_image = img_array.copy()
    
    # Generate random noise mask
    mask = np.random.random(img_array.shape[:2])
    
    # Add salt noise (white pixels)
    salt_mask = mask < salt_prob
    if len(img_array.shape) == 3:  # Color image
        for i in range(img_array.shape[2]):
            noised_image[salt_mask, i] = 255
    else:  # Grayscale image
        noised_image[salt_mask] = 255
    
    # Add pepper noise (black pixels)
    pepper_mask = (mask >= salt_prob) & (mask < salt_prob + pepper_prob)
    if len(img_array.shape) == 3:  # Color image
        for i in range(img_array.shape[2]):
            noised_image[pepper_mask, i] = 0
    else:  # Grayscale image
        noised_image[pepper_mask] = 0
    
    return noised_image

def gaussian_noise_gs(mean=0.0, sigma=20, image=np.ones((256, 256))):
    '''
    Args:
        mean (float): Mean of the Gaussian noise distribution (default 0)
        sigma (float): Standard deviation of the Gaussian noise distribution (default 20)
        image (numpy.ndarray): Input image (default is a 256x256 array of ones)
    
    Returns:
        numpy.ndarray: Noisy image with pixel values clipped to [0, 255]
    '''
    row,col= image.shape
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    noisy_clipped = np.clip(noisy, 0, 255)
    return noisy_clipped

def add_gaussian_noise_giva(imagem, media=0, sigma=25, same_noise=False): #rui­do aditivo
    if same_noise:
        # Gera um rui­do para cada pixel em apenas 1 canal (dimensao: altura x largura)
        ruido_2d = np.random.normal(media, sigma, imagem.shape[:2])
        # Repete o rui­do para os 3 canais, garantindo que cada pixel tera o mesmo valor em R, G e B
        ruido = np.stack([ruido_2d]*3, axis=-1)
    else:
        # Gera rui­do independente para cada pixel em cada canal (dimensao: altura x largura x 3)
        ruido = np.random.normal(media, sigma, imagem.shape)
    
    # Adiciona o rui­do a imagem
    imagem_ruidosa = imagem + ruido
    
    # Garante que os valores dos pixels permanecam no intervalo [0, 255]
    imagem_ruidosa = np.clip(imagem_ruidosa, 0, 255)
    
    return imagem_ruidosa.astype(np.uint8)

def poisson_noise(image=np.ones((256, 256))):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def periodic_noise(frequency_x, frequency_y, amplitude, image=np.ones((256, 256))):
    """
    Adds periodic sinusoidal noise to an existing image.

    Parameters:
    image (np.ndarray): Input image.
    frequency_x (float): Frequency of the sinusoidal noise along the x-axis.
    frequency_y (float): Frequency of the sinusoidal noise along the y-axis.
    amplitude (float): Amplitude of the noise.

    Returns:
    np.ndarray: The image with periodic noise added.
    """
    rows, cols = image.shape[:2]
    
    # Create the x and y coordinates
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Generate sinusoidal noise
    noise = amplitude * np.sin(2 * np.pi * (frequency_x * X / cols + frequency_y * Y / rows))
    
    # If the image is RGB, apply noise to each channel; if grayscale, apply directly
    if len(image.shape) == 3:  # RGB image
        noisy_image = np.copy(image).astype(float)
        for i in range(3):  # Apply the noise to each color channel
            noisy_image[:, :, i] += noise
    else:  # Grayscale image
        noisy_image = image.astype(float) + noise
    
    # Clip values to the valid range for an image (0 to 255)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def add_speckle_noise(image, intensity=0.5, mean=0, sigma=0.1):
    """
    Add speckle noise to an image.
    
    Speckle noise is a multiplicative noise that follows the equation:
    J = I + n*I = I(1 + n)
    where I is the input image and n is Gaussian noise with mean and sigma.
    
    Parameters:
    image (PIL.Image): Input image
    intensity (float): Noise intensity factor (0 to 1)
    mean (float): Mean of the Gaussian distribution
    sigma (float): Standard deviation of the Gaussian distribution
    
    Returns:
    PIL.Image: Image with added speckle noise
    """
    # Convert image to numpy array
    img_array = np.array(image).astype(float) / 255.0
    
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, img_array.shape)
    
    # Apply speckle noise equation: I + n*I = I(1 + n)
    # Scale the noise by intensity factor
    noised_image = img_array + intensity * noise * img_array
    
    # Clip values to valid range [0, 1]
    noised_image = np.clip(noised_image, 0, 1)
    
    # Return the array
    return (noised_image * 255).astype(np.uint8)

def add_speckle_noise_v2(image, mean=0, sigma=10):
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
    # clips noised image to values [0, 255]
    noisy_clipped = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy_clipped

def speckle_noise_gs(image=np.ones((256, 256))):
    row,col = image.shape
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)        
    noisy = image + image * gauss
    return noisy

def fractal_surface(image_size=(256, 256), roughness=0.5, iterations=7):
    """
    Generate a fractal surface using the 2D midpoint displacement algorithm.
    
    Parameters:
    width (int): Width of the output image
    height (int): Height of the output image
    roughness (float): Controls the roughness of the fractal (0.0 to 1.0)
    iterations (int): Number of iterations to perform
    
    Returns:
    numpy.ndarray: 2D array representing the fractal surface
    """
    # Size should be 2^n + 1
    (width, height) = image_size
    size = max(width, height)
    n = int(np.ceil(np.log2(size - 1)))
    size = 2**n + 1
    
    # Initialize grid with zeros
    grid = np.zeros((size, size))
    
    # Set the four corners to random initial values
    grid[0, 0] = random.random()
    grid[0, size-1] = random.random()
    grid[size-1, 0] = random.random()
    grid[size-1, size-1] = random.random()
    
    # Apply the midpoint displacement algorithm
    step = size - 1
    scale = 1.0 * roughness
    
    while step > 1:
        half_step = step // 2
        
        # Diamond step
        for y in range(half_step, size, step):
            for x in range(half_step, size, step):
                avg = (grid[y-half_step, x-half_step] +  # top-left
                       grid[y-half_step, x+half_step] +  # top-right
                       grid[y+half_step, x-half_step] +  # bottom-left
                       grid[y+half_step, x+half_step]) / 4.0  # bottom-right
                
                grid[y, x] = avg + (random.random() * 2 - 1) * scale
        
        # Square step
        for y in range(0, size, half_step):
            for x in range((y + half_step) % step, size, step):
                total = 0
                count = 0
                
                # Check the four adjacent cells
                if y >= half_step:  # top
                    total += grid[y-half_step, x]
                    count += 1
                if y + half_step < size:  # bottom
                    total += grid[y+half_step, x]
                    count += 1
                if x >= half_step:  # left
                    total += grid[y, x-half_step]
                    count += 1
                if x + half_step < size:  # right
                    total += grid[y, x+half_step]
                    count += 1
                
                avg = total / count
                grid[y, x] = avg + (random.random() * 2 - 1) * scale
        
        step = half_step
        scale *= roughness
    
    # Crop to requested dimensions
    return grid[:height, :width]

def fractal_surface_hurst(image_size=(256, 256), hurst=0.7, delta0=1.0, iterations=None):
    """
    Generate a fractal surface using the random midpoint displacement algorithm with Hurst exponent.
    Source: Ribeiro HV, Zunino L, Lenzi EK, Santoro PA, Mendes RS
            (2012) Complexity-Entropy Causality Plane as a Complexity
            Measure for Two-Dimensional Patterns. PLoS ONE 7(8): e40689.
            https://doi.org/10.1371/journal.pone.0040689
    
    Parameters:
    image_size (tuple): Width and height of the output image
    hurst (float): Hurst exponent controlling the fractal dimension (0.0 to 1.0)
                  Higher values create smoother surfaces
    delta0 (float): Initial standard deviation for the random displacement
    iterations (int): Number of iterations to perform (if None, calculated from size)
    
    Returns:
    numpy.ndarray: 2D array representing the fractal surface
    """
    # Size should be 2^k + 1
    (width, height) = image_size
    size = max(width, height)
    
    # Calculate k based on size or use provided iterations
    if iterations is None:
        k = int(np.ceil(np.log2(size - 1)))
    else:
        k = iterations
    
    size = 2**k + 1
    
    # Fractal dimension D = 3 - h, where h is the Hurst exponent
    h = hurst
    D = 3 - h
    
    # Initialize grid with zeros
    grid = np.zeros((size, size))
    
    # Set the four corners to random initial values
    grid[0, 0] = np.random.normal(0, delta0)
    grid[0, size-1] = np.random.normal(0, delta0)
    grid[size-1, 0] = np.random.normal(0, delta0)
    grid[size-1, size-1] = np.random.normal(0, delta0)
    
    # Apply the midpoint displacement algorithm with Hurst exponent
    delta = delta0
    
    for i in range(k):
        step = 2**(k-i)
        half_step = step // 2
        
        # Diamond step: add midpoints of squares
        for y in range(0, size-1, step):
            for x in range(0, size-1, step):
                # Calculate midpoint coordinates
                mid_y = y + half_step
                mid_x = x + half_step
                
                if mid_y < size and mid_x < size:
                    # Average of the four corners
                    avg = (grid[y, x] +               # top-left
                           grid[y, x+step] +          # top-right
                           grid[y+step, x] +          # bottom-left
                           grid[y+step, x+step]) / 4  # bottom-right
                    
                    # Displace by Gaussian random value with current delta
                    grid[mid_y, mid_x] = avg + np.random.normal(0, delta)
        
        # Square step: add midpoints of edges
        for y in range(0, size, half_step):
            for x in range(0, size, half_step):
                # Skip already calculated points
                if (y % step == 0 and x % step == 0) or (y % step == half_step and x % step == half_step):
                    continue
                
                count = 0
                total = 0
                
                # Check the four adjacent points (in a square pattern)
                if y >= half_step:  # top
                    total += grid[y-half_step, x]
                    count += 1
                if y + half_step < size:  # bottom
                    total += grid[y+half_step, x]
                    count += 1
                if x >= half_step:  # left
                    total += grid[y, x-half_step]
                    count += 1
                if x + half_step < size:  # right
                    total += grid[y, x+half_step]
                    count += 1
                
                if count > 0:
                    avg = total / count
                    grid[y, x] = avg + np.random.normal(0, delta)
        
        # Update delta based on Hurst exponent: δₖ = δ₀·2^(-k·h)
        delta = delta0 * (2 ** (-(i+1) * h))
    
    # Normalize values to 0-1 range for easier visualization
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    
    # Crop to requested dimensions
    return grid[:height, :width]


if __name__ == '__main__':
    try:
        # Generate fractal surfaces with different Hurst exponents
        hurst_values = [0.1, 0.5, 0.9]
        surfaces = []

        # Generate the three surfaces
        for h in hurst_values:
            surfaces.append(fractal_surface_hurst((256, 256), h))

        # Create a figure with three subplots side by side
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot each surface in its own subplot
        for i, (h, surface) in enumerate(zip(hurst_values, surfaces)):
            im = axs[i].imshow(surface, cmap='hsv')
            axs[i].set_title(f'Hurst = {h}')
            plt.colorbar(im, ax=axs[i], label='Elevation')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            print(surface.shape)
            print(surface.min(), surface.max())

        plt.tight_layout()
      
        # plt.figure(figsize=(10, 8))
        # plt.imshow(surface, cmap='hsv')
        # plt.colorbar(label='Elevation')
        plt.show()
    except Exception as e:
        print("An exception occurred:\n\t\t{}".format(str(e)))