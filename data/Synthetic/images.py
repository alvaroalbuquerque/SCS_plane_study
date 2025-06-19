from .utils import uniform_image, colored_noise, fractal_surface

images = {
    'constant-0.5': uniform_image((256, 256), 0.5),
    'colored-noise-k6': colored_noise((256, 256), k=6),
    'fractal-surface-h0.5-delta0.5': fractal_surface((256, 256), hurst=0.5, delta0=0.5)
}