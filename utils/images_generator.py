import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def generate_histogram(img, isNormalized=True, title='Histogram of Img'):
    '''
        Plots a histogram of the image
    '''
    if not isNormalized:
        normalized_img = normalize_img_to_0_255(img)
    else:
        normalized_img = img
    flattened_img = normalized_img.flatten()
    plt.hist(flattened_img, bins=256, range=(0, 255), color='blue', alpha=0.7)
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

    # ruido branco uniforme
    white_noise = np.random.uniform(-0.5, 0.5, image_size)
    
    # Aplica a transformada e desloca as baixas frequencias para o centro
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

    # Normaliza o resultado
    # colored_noise /= np.std(colored_noise)

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

def gaussian_noise_gs(mean=0.0, sigma=0.1, image=np.ones((256, 256))):
    row,col= image.shape
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

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

def speckle_noise_gs(image=np.ones((256, 256))):
    row,col = image.shape
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)        
    noisy = image + image * gauss
    return noisy

if __name__ == '__main__':
    try:
        # Parameters
        a = 1.4
        b = 0.3
        x0 = 0.1
        y0 = 0.1
        iterations = 1000

        # Generate the 2D Hénon map data
        image_x, image_y = henon_map(a, b, x0, y0, iterations)

        # Plot the results
        plt.figure(1,figsize=(12, 8))
        plt.imshow(normalize_img_to_0_255(image_x), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='x Value')
        plt.title('Hénon Map 2D Data (x)')
        plt.xlabel('Image X')
        plt.ylabel('Image Y')
        # plt.show()
        # plt.clf()
        plt.figure(2,figsize=(12, 8))
        plt.imshow(normalize_img_to_0_255(image_y), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='y Value')
        plt.title('Hénon Map 2D Data (y)')
        plt.xlabel('Image X')
        plt.ylabel('Image Y')
        plt.show()
    except Exception as e:
        print("An exception occurred:\n\t\t{}".format(str(e)))