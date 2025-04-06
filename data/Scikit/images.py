from skimage import data
from skimage.color import rgb2gray
from .utils import normalize_uint8_to_float64

images = {
    'astronaut': rgb2gray(data.astronaut()),
    'cat': rgb2gray(data.cat()),
    'coins': normalize_uint8_to_float64(data.coins())
}