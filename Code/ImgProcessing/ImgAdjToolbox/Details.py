import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as Interp

def sharpen_USM(img, size, den):
    blur_img = cv2.GaussianBlur(img, (0, 0), size)
    usm = cv2.addWeighted(img, 1+den, blur_img, -den, 0)
    usm = np.clip(usm, 0, 1)
    return usm