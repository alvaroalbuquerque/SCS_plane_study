import numpy as np

def henon_texture(a=1.4, b=0.3, x0=0.1, y0=0.3, image_size=(256, 256), burn_in=1000):
    total_values = image_size[0] * image_size[1]
    x, y = x0, y0

    for _ in range(burn_in):
        x, y = 1 - a * x**2 + y, b * x

    values = []
    for _ in range(total_values):
        x, y = 1 - a * x**2 + y, b * x
        values.append(x)

    values = np.array(values)
    values -= values.min()
    values /= values.max()
    return values.reshape(image_size).astype(np.float64)


def logistic_texture(r=3.9, x0=0.5, image_size=(256, 256), burn_in=1000):
    total_values = image_size[0] * image_size[1]
    x = x0

    for _ in range(burn_in):
        x = r * x * (1 - x)

    values = []
    for _ in range(total_values):
        x = r * x * (1 - x)
        values.append(x)

    values = np.array(values)
    values -= values.min()
    values /= values.max()
    return values.reshape(image_size).astype(np.float64)


def tent_map_texture(mu=2.0, x0=0.1, image_size=(256, 256), burn_in=1000):
    """
    Generate a texture using the tent map chaotic system.
    
    Parameters:
    mu (float): Bifurcation parameter, controls chaotic behavior (ideally 0 < mu ≤ 2)
    x0 (float): Initial value (should be between 0 and 1)
    image_size (tuple): Dimensions of the output image
    burn_in (int): Number of initial iterations to discard
    
    Returns:
    numpy.ndarray: Generated texture as a 2D array
    """
    # Parameter validation
    if not (0 < mu <= 2.0):
        print(f"Warning: mu={mu} is outside the stable range (0,2]. Using mu=1.9")
        mu = 1.9
    
    # Ensure x0 is in (0,1)
    x0 = max(0.001, min(0.999, x0))
    
    total_values = image_size[0] * image_size[1]
    x = x0
    
    # Burn-in phase with stability checks
    for _ in range(burn_in):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)
        
        # Check for instability
        if not (0 <= x <= 1):
            print(f"Warning: Unstable value during burn-in. Resetting x to 0.5")
            x = 0.5
    
    # Generate values with stability checks
    values = []
    for _ in range(total_values):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)
            
        # Check for instability
        if not (0 <= x <= 1) or np.isnan(x):
            print(f"Warning: Generated unstable value. Resetting x to 0.5")
            x = 0.5
            
        values.append(x)
    
    values = np.array(values)
    
    # Check if all values are the same (would cause division by zero)
    if np.max(values) == np.min(values):
        print("Warning: All generated values are identical. Adding noise for variation.")
        values = values + np.random.uniform(0, 0.01, size=values.shape)
    
    # Normalize to [0,1] range
    values -= values.min()
    max_val = values.max()
    if max_val > 0:  # Avoid division by zero
        values /= max_val
    
    return values.reshape(image_size).astype(np.float64)


def lozi_map_texture(a=1.7, b=0.5, x0=0.0, y0=0.0, image_size=(256, 256), burn_in=1000):
    total_values = image_size[0] * image_size[1]
    x, y = x0, y0

    for _ in range(burn_in):
        x, y = 1 - a * abs(x) + y, b * x

    values = []
    for _ in range(total_values):
        x, y = 1 - a * abs(x) + y, b * x
        values.append(x)

    values = np.array(values)
    values -= values.min()
    values /= values.max()
    return values.reshape(image_size).astype(np.float64)


def ikeda_map_texture(u=0.918, x0=0.1, y0=0.1, image_size=(256, 256), burn_in=1000):
    """
    Generate a texture using the Ikeda map chaotic system.
    
    Parameters:
    u (float): Control parameter, typically between 0.7 and 1.0
    x0, y0 (float): Initial coordinates
    image_size (tuple): Dimensions of the output image
    burn_in (int): Number of initial iterations to discard
    
    Returns:
    numpy.ndarray: Generated texture as a 2D array with values in [0,1]
    """
    # Validate parameters
    if not (0.7 <= u <= 1.0):
        print(f"Warning: u={u} may be outside stable range. Using u=0.918")
        u = 0.918
    
    total_values = image_size[0] * image_size[1]
    x, y = x0, y0
    
    # Function to safely update coordinates
    def safe_update(x, y):
        try:
            # Prevent extreme values
            x = np.clip(x, -1e6, 1e6)
            y = np.clip(y, -1e6, 1e6)
            
            # Calculate t with safety check for division by zero
            denominator = 1 + x**2 + y**2
            if denominator < 1e-10:  # Avoid extremely small denominators
                t = 0.4 - 6
            else:
                t = 0.4 - 6 / denominator
                
            # Update coordinates using Ikeda map
            new_x = 1 + u * (x * np.cos(t) - y * np.sin(t))
            new_y = u * (x * np.sin(t) + y * np.cos(t))
            
            # Check for NaN values
            if np.isnan(new_x) or np.isnan(new_y):
                raise ValueError("NaN value detected")
                
            return new_x, new_y
            
        except (ValueError, ZeroDivisionError, OverflowError):
            # Reset to reasonable values if error occurs
            print("Warning: Numerical instability detected. Resetting coordinates.")
            return 0.1, 0.1
    
    # Burn-in phase
    for _ in range(burn_in):
        x, y = safe_update(x, y)
    
    # Generate values for texture
    values = []
    failures = 0
    i = 0
    
    while i < total_values:
        x, y = safe_update(x, y)
        
        # Check if the update was successful (not a reset)
        if not (x == 0.1 and y == 0.1):
            values.append(x)
            i += 1
        else:
            failures += 1
            if failures > 100:
                # If too many failures, fill remaining values with random numbers
                print(f"Warning: Exceeded failure threshold. Adding random values for remaining {total_values - i} pixels.")
                values.extend(np.random.uniform(-1, 1, total_values - i).tolist())
                break
    
    values = np.array(values)
    
    # Check if normalization is possible
    if np.max(values) == np.min(values):
        print("Warning: All values are identical. Adding noise for variation.")
        values = values + np.random.uniform(0, 0.01, size=values.shape)
    
    # Normalize to [0,1] range
    values -= np.min(values)
    max_val = np.max(values)
    if max_val > 0:  # Avoid division by zero
        values /= max_val
    
    return values.reshape(image_size).astype(np.float64)


def arnolds_cat_map_texture(image_size=(256, 256), steps=100):
    n = image_size[0]
    assert image_size[0] == image_size[1], "Arnold's Cat Map requires a square image."

    x, y = np.meshgrid(np.arange(n), np.arange(n))
    values = np.sin(2 * np.pi * x / n) * np.cos(2 * np.pi * y / n)

    for _ in range(steps):
        x_new = (x + y) % n
        y_new = (x + 2 * y) % n
        x, y = x_new, y_new
        values = np.sin(2 * np.pi * x / n) * np.cos(2 * np.pi * y / n)

    values -= values.min()
    values /= values.max()
    return values.astype(np.float64)


if __name__ == '__main__':
    try:
        # Generate all six chaotic textures
        henon_img = henon_texture()
        logistic_img = logistic_texture()
        tent_img = tent_map_texture()
        lozi_img = lozi_map_texture()
        ikeda_img = ikeda_map_texture()
        arnolds_cat_img = arnolds_cat_map_texture()
        print(henon_img.shape, logistic_img.shape, tent_img.shape, lozi_img.shape, ikeda_img.shape, arnolds_cat_img.shape)
    except Exception as e:
        print("An exception occurred:\n\t\t{}".format(str(e)))
