import sys, os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ImgProcessing.ImgAdjToolbox.Tone import *
from ImgProcessing.ImgAdjToolbox.Details import *



class ImgAdjPipeline(object):
    def __init__(self, show=False):
        self.adj_exposure = False
        self.adj_curve = False
        self.adj_contrast = False
        self.adj_sharpen = False
        self.adj_temp = False
        self.adj_saturation = False

        self.show = show

    def load_img(self, inimg):
        if self.show:
            print("Image data type is: " + str(inimg.dtype))
            
        self.imgtype = inimg.dtype
        if inimg.dtype == np.uint8:
            self.inimg = inimg.astype("float32") / 255.0
        elif inimg.dtype == "float32":
            self.inimg = inimg

    def set_curve(self, l_nodes, r_nodes=None, g_nodes=None, b_nodes=None):
        self.adj_curve = True
        self.curve_par = {"l":l_nodes, "r":r_nodes, "g":g_nodes, "b":b_nodes}

    def set_exposure(self, adj):
        self.adj_exposure = True
        self.exposure_par = adj

    def set_color_temp(self, adj):
        self.adj_temp = True
        self.temp_par = adj

    def set_saturation(self, adj):
        self.adj_saturation = True
        self.saturation_par = adj

    def set_contrast(self, adj):
        self.adj_contrast = True
        self.contrast_par = adj

    def set_sharpen(self, size, density):
        self.adj_sharpen = True
        self.sharpen_par = {"size":size, "den":density}

    def show_process(self, img, title):
        if self.show:
            plt.figure()
            plt.title(title)
            plt.imshow(img)
            # plt.show()
    
    def process_img(self, out_size=None):
        img = self.inimg
        self.show_process(img, "Original")

        if self.adj_exposure:
            img = exposure(img, self.exposure_par)
            self.show_process(img, "Exposure")

        if self.adj_temp:
            img = color_temp(img, self.temp_par)
            self.show_process(img, "Color Temprature")

        if self.adj_contrast:
            img = contrast(img, self.contrast_par)
            self.show_process(img, "Contrast")

        if self.adj_saturation:
            img = natural_saturation(img, self.saturation_par)
            self.show_process(img, "Saturation")

        if self.adj_curve:
            img = curve(img, l_nodes=self.curve_par["l"], show_curve=self.show, r_nodes=self.curve_par["r"], g_nodes=self.curve_par["g"], b_nodes=self.curve_par["b"])
            self.show_process(img, "Curve")

        if self.adj_sharpen:
            img = sharpen_USM(img, size=self.sharpen_par["size"], den=self.sharpen_par["den"])
            self.show_process(img, "Sharpen")

        if self.imgtype == np.uint8:
            img = np.clip(img*255, 0, 255).astype(np.uint8)
        
        if out_size is not None:
            img = cv2.resize(img, (out_size[1], out_size[0]), cv2.INTER_CUBIC)
        
        plt.show()
        return img


if __name__=="__main__":
    img = cv2.cvtColor(cv2.imread("/Users/cjarry/Desktop/1.bmp"), cv2.COLOR_BGR2RGB)
    imgPipeline = ImgAdjPipeline(show=True)

    imgPipeline.set_exposure(0.2)
    imgPipeline.set_curve([[0,0], [0.27,0.33], [0.55,0.82], [1,1]])
    imgPipeline.set_color_temp(-20)
    imgPipeline.set_saturation(50)
    imgPipeline.set_sharpen(1, 2)

    imgPipeline.load_img(img)
    img2 = imgPipeline.process_img()
