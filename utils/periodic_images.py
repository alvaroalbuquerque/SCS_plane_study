import numpy as np

def normalize(img):
    img = img - np.min(img)
    return img / (np.max(img) + 1e-8)

def get_levels_array(levels):
    """Return array of equally spaced values between 0 and 1."""
    return np.linspace(0, 1, levels, endpoint=True)

def horizontal_stripes_texture(period=16, levels=2, image_size=(256, 256)):
    rows = np.indices(image_size)[0]
    indices = (rows // period) % levels
    values = get_levels_array(levels)
    return values[indices].astype(np.float64)

def vertical_stripes_texture(period=16, levels=2, image_size=(256, 256)):
    cols = np.indices(image_size)[1]
    indices = (cols // period) % levels
    values = get_levels_array(levels)
    return values[indices].astype(np.float64)

def checkerboard_texture(period=16, levels=2, image_size=(256, 256)):
    rows, cols = np.indices(image_size)
    indices = ((rows // period + cols // period) % levels)
    values = get_levels_array(levels)
    return values[indices].astype(np.float64)

def diagonal_lines_texture(period=16, levels=2, image_size=(256, 256)):
    rows, cols = np.indices(image_size)
    indices = ((rows + cols) // period) % levels
    values = get_levels_array(levels)
    return values[indices].astype(np.float64)




if __name__ == '__main__':
    try:
        # Generate all six chaotic textures
        vertical_stripes_image = vertical_stripes_texture()
        horizontal_stripes_image = horizontal_stripes_texture()
        checkerboard_image = checkerboard_texture()
        diagonal_lines_image = diagonal_lines_texture()
        print(vertical_stripes_image.shape, horizontal_stripes_image.shape, checkerboard_image.shape, diagonal_lines_image.shape)
    except Exception as e:
        print("An exception occurred:\n\t\t{}".format(str(e)))
