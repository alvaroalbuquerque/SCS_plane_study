import numpy as np
import cv2
import pandas as pd

def normalize_img_to_int(img):
    '''
        Normalizes between 0 and 255 (np.uint8)
    '''
    return cv2.normalize(img, None,  0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def normalize_uint8_to_float64(img):
    '''
        Normalizes from uint8 [0, 255] to np.float64 [0.0, 1.0]
    '''
    # check if input is a numpy array
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # check if dtype is uint8
    if img.dtype != np.uint8:
        raise ValueError("Image must be of dtype uint8")
    
    # check if pixel values are in range [0, 1]
    if img.min() < 0 or img.max() > 255:
        raise ValueError("Pixel values must be in the range [0, 255]")
    
    # Set the min and max of a uint8 image
    uint8_min = 0
    uint8_max = 255

    # normalizes
    normalized = ((img - uint8_min) / (uint8_max - uint8_min)).astype(np.float64)
    return normalized

def normalize_outbound_to_float64(img):
    '''
        Normalizes image float64 with undefined bounds to np.float64 [0.0, 1.0]
    '''
    # check if input is a numpy array
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    # check if dtype is uint8
    if img.dtype != np.float64:
        raise ValueError("Image must be of dtype float64")
    
    # Set the min and max of the image
    image_min = img.min()
    image_max = img.max()

    # normalizes
    normalized = ((img - image_min) / (image_max - image_min)).astype(np.float64)
    return normalized

def read_entropy_complexity_limits(folder_path, N=24):
    '''
        Reads the entropy and complexity limits from files.
        Returns two dataframes: df_cont and df_troz.
    '''
    df_cont = pd.read_csv(folder_path +'continua-N'+str(N)+'.q1', skiprows=20, sep = '  ')
    df_cont.columns = ['HT', 'CJT']

    df_troz = pd.read_csv(folder_path +'trozos-N'+str(N)+'.q1', skiprows=20, sep = '  ')
    df_troz.columns = ['HT', 'CJT']
    
    return df_cont, df_troz