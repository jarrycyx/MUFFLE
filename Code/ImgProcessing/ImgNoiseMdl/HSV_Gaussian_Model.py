import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
from ImgNoiseMdl import Utils
import GetTestImg


def pepper_sault(img, p_pepper=0, p_sault=0):
    size = img.shape
    pepper = np.random.uniform(0,1,size=size) < p_pepper
    sault = np.random.uniform(0,1,size=size) < p_sault
    img[np.where(pepper==True)] = 0
    img[np.where(sault==True)] = 255
    return img

def add_noise(img, l_sigma, a_sigma, b_sigma, kd, p_pepper=0, p_sault=0):
    labimg = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(int)
    lab_noisy = labimg
    imsize = labimg.shape[:2]
    
    l_noise = np.random.normal(0, l_sigma, imsize)
    a_noise = np.random.normal(0, a_sigma, imsize)
    b_noise = np.random.normal(0, b_sigma, imsize)
    
    lab_noisy[:,:,0] = np.clip(labimg[:,:,0] + l_noise, 0, 255)
    lab_noisy[:,:,1] = np.clip(labimg[:,:,1] + a_noise, 0, 255)
    lab_noisy[:,:,2] = np.clip(labimg[:,:,2] + b_noise, 0, 255)
    noisy = cv2.cvtColor(lab_noisy.astype(np.uint8), 
                         cv2.COLOR_YCrCb2RGB)
    
    noisy_quan = ((noisy / kd).astype(np.uint8) * kd).astype(np.uint8)
    noisy_peppersault = pepper_sault(noisy_quan, p_pepper=p_pepper, p_sault=p_sault)
    
    return noisy_peppersault

def add_noise_gray(img, sigma, kd, p_pepper=1e-6, p_sault=1e-4):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    noisy = np.clip(img + 
                    np.random.normal(0, sigma, img.shape), 
                    0, 255).astype(np.uint8)
    
    noisy_quan = ((noisy / kd).astype(np.uint8) * kd).astype(np.uint8)
    noisy_peppersault = pepper_sault(noisy_quan, p_pepper=p_pepper, p_sault=p_sault)
    
    return cv2.cvtColor(noisy_peppersault, cv2.COLOR_GRAY2RGB)


if __name__=="__main__":
    imgsource = GetTestImg.get_testimg(index=2) #  np.zeros([10,10,3])+10# 
    noisy = add_noise(imgsource, 15, 30, 30, 2)
    
    cv2.imwrite("./pic/noisy.bmp", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))