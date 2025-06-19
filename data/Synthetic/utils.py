import numpy as np

def uniform_image(image_size=(256, 256), intensity=0.5):
    """
    Generate a uniform image with a specified intensity.
    """
    # Check if image_size is a tuple of two positive integers
    if not (isinstance(image_size, tuple) and len(image_size) == 2 and
            all(isinstance(dim, int) and dim > 0 for dim in image_size)):
        raise ValueError("Parameter image_size must be a tuple of two positive integers (width, height)")
    # Check if intensity is a valid float in range [0, 1]
    if not (0 <= intensity <= 1):
        raise ValueError("Parameter intensity must be a float in the range [0, 1]")
    return np.ones(image_size, dtype=np.float64)*intensity

def colored_noise(image_size=(256, 256), k=6):
    """
    Generate colored noise using the Fourier transform method.
    """
    # Check if image_size is a tuple of two positive integers
    if not (isinstance(image_size, tuple) and len(image_size) == 2 and
            all(isinstance(dim, int) and dim > 0 for dim in image_size)):
        raise ValueError("Parameter image_size must be a tuple of two positive integers (width, height)")
    # Check if k is a valid integer
    if k not in [1, 2, 3, 4, 5, 6]:
        raise ValueError("Parameter k must be an integer in the range [1, 6]")
    
    # Uniform white noise
    white_noise = np.random.uniform(-0.5, 0.5, image_size)
    
    # Applies the Fourier transform and shifts low frequencies to the center
    y = np.fft.fftshift(np.fft.fft2(white_noise))
    
    _x, _y = np.mgrid[0:y.shape[0], 0:y.shape[1]]
    f = np.hypot(_x - y.shape[0] / 2, _y - y.shape[1] / 2)

    # Modified noise
    y_2 = y / f**(k/2)

    colored_noise = np.nan_to_num(y_2, nan=0, posinf=0, neginf=0)

    # Unshifts and applies the inverse Fourier transform 
    colored_noise = np.fft.ifft2(np.fft.ifftshift(colored_noise)).real

    
    # Normalize to [0, 1] range
    min_val = colored_noise.min()
    max_val = colored_noise.max()
    normalized_noise = (colored_noise - min_val) / (max_val - min_val)    

    return normalized_noise

def fractal_surface(image_size=(256, 256), hurst=0.7, delta0=1.0, iterations=None):
    """
    Generate a fractal surface using the random midpoint displacement algorithm with Hurst exponent.
    """
    
    # Check if image_size is a tuple of two positive integers
    if not (isinstance(image_size, tuple) and len(image_size) == 2 and
            all(isinstance(dim, int) and dim > 0 for dim in image_size)):
        raise ValueError("Parameter image_size must be a tuple of two positive integers (width, height)")
    # Check if hurst is a valid float in range [0, 1]
    if not (0 <= hurst <= 1):
        raise ValueError("Parameter hurst must be a float in the range [0, 1]")
    # Check if delta0 is a valid positive float
    if not (isinstance(delta0, float) and delta0 > 0):
        raise ValueError("Parameter delta0 must be a positive float")
    # Check if iterations is a valid integer or None
    if iterations is not None and (not isinstance(iterations, int) or iterations < 0):
        raise ValueError("Parameter iterations must be a non-negative integer or None")
    
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