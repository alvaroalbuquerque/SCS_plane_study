import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

def normalize(img):
    img = img - np.min(img)
    return img / (np.max(img) + 1e-8)

# 1. f^-k Noise
def fk_noise_texture(k=1.0, image_size=(256, 256)):
    h, w = image_size
    noise = np.random.uniform(-0.5, 0.5, (h, w))
    spectrum = fft2(noise)

    # Generate frequency grid
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.fftfreq(w).reshape(1, -1)
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1e-6  # avoid division by zero

    # Apply f^-k power law
    filtered_spectrum = spectrum * (1 / f**(k / 2))
    image = np.real(ifft2(filtered_spectrum))
    return normalize(image).astype(np.float64)

# 2. Fractional Brownian Motion (FBM)
def fbm_texture(H=0.7, image_size=(256, 256)):
    h, w = image_size
    freq_y = np.fft.fftfreq(h).reshape(-1, 1)
    freq_x = np.fft.fftfreq(w).reshape(1, -1)
    f = np.sqrt(freq_x**2 + freq_y**2)
    f[0, 0] = 1e-6

    S_f = 1 / (f ** (2 * H + 1))  # PSD of FBM
    noise = np.random.normal(size=(h, w)) + 1j * np.random.normal(size=(h, w))
    spectrum = fft2(noise) * np.sqrt(S_f)
    image = np.real(ifft2(spectrum))
    return normalize(image).astype(np.float64)

# 3. Fractional Gaussian Noise (FGN)
def fgn_texture(H=0.7, image_size=(256, 256)):
    fbm_img = fbm_texture(H, (image_size[0] + 1, image_size[1] + 1))

    # Calculate differences
    diff_y = np.diff(fbm_img, axis=0)[:image_size[0], :image_size[1]]
    diff_x = np.diff(fbm_img, axis=1)[:image_size[0], :image_size[1]]

    fgn = diff_x + diff_y
    return normalize(fgn).astype(np.float64)

def fbm_texture_alpha(alpha=2.0, image_size=(256, 256)):
    """
    Generate 2D FBM using spectral exponent alpha in (1, 3)
    where H = (alpha - 1)/2
    """
    H = (alpha - 1) / 2
    h, w = image_size
    freq_y = np.fft.fftfreq(h).reshape(-1, 1)
    freq_x = np.fft.fftfreq(w).reshape(1, -1)
    f = np.sqrt(freq_x**2 + freq_y**2)
    f[0, 0] = 1e-6

    S_f = 1 / (f ** alpha)
    noise = np.random.normal(size=(h, w)) + 1j * np.random.normal(size=(h, w))
    spectrum = fft2(noise) * np.sqrt(S_f)
    image = np.real(ifft2(spectrum))
    return normalize(image).astype(np.float64)


def fgn_texture_alpha(alpha=0.0, image_size=(256, 256)):
    """
    Generate 2D FGN using spectral exponent alpha in (-1, 1)
    where H = (alpha + 1)/2
    """
    H = (alpha + 1) / 2
    fbm_img = fbm_texture_alpha(alpha=2 * H + 1, image_size=(image_size[0] + 1, image_size[1] + 1))
    diff_y = np.diff(fbm_img, axis=0)[:image_size[0], :image_size[1]]
    diff_x = np.diff(fbm_img, axis=1)[:image_size[0], :image_size[1]]
    fgn = diff_x + diff_y
    return normalize(fgn).astype(np.float64)



if __name__ == '__main__':
    try:
        # Generate all stochastic images
        fk_img = fk_noise_texture(k=1.0, image_size=(256, 256))
        fbm_img = fbm_texture(H=0.7, image_size=(256, 256))
        fgn_img = fgn_texture(H=0.7, image_size=(256, 256))
        print(fk_img.shape, fbm_img.shape, fgn_img.shape)
    except Exception as e:
        print("An exception occurred:\n\t\t{}".format(str(e)))