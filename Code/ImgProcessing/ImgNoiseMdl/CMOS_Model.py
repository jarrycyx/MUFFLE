import sys
import os

import numpy as np
import cv2



class CMOS_Camera(object):
    def __init__(self, pars):

        self.sigma_beta = pars['sigma_beta']
        self.nd_factor = pars['nd_factor']
        self.sigma_r = pars['sigma_r']

    # sigma_beta, kc should be a list of size 3 or a number
    # imgsize should be a list of size 2 or 3
    # other arg should be a number
    # channel number of imgsize should match imgsource
    def simu_noise(self, img_source=None,
                   sigma_beta=0.01, nd_factor=1, sigma_r=1, ka=2,
                   exp_time=1, imgsize=[400, 800, 3], illumination=1,
                   sault_p=1e-5, pepper_p=1e-6, kd=3, gamma=1):

        def beta_c_array(sigma_beta, size=[200, 400]):
            beta_c_row = np.random.normal(loc=1, scale=sigma_beta, size=[size[0], 1])
            return np.array(beta_c_row).repeat(size[1], axis=1)

        def shot_noise(ne_array):
            return np.random.poisson(lam=ne_array)

        def dark_current(nd, size=[200, 400]):
            # return np.random.poisson(lam=nd, size=size)
            # return np.clip(np.random.exponential(nd, size=size) - nd, 0, np.inf)
            return np.clip(np.random.poisson(lam=nd, size=size) - nd, 0, np.inf)

        def read_noise(sigma_r, size=[200, 400]):
            return np.random.normal(loc=0, scale=sigma_r, size=size)

        def pepper_sault(img, p_pepper=0, p_sault=0):
            size = img.shape
            pepper = np.random.uniform(0,1,size=size) < p_pepper
            sault = np.random.uniform(0,1,size=size) < p_sault
            img[np.where(pepper==True)] = 0
            img[np.where(sault==True)] = 255
            return img
        
        green_gain = 2
        if img_source is None:
            ne_array = np.zeros(imgsize) * exp_time
        else:
            img_source = cv2.resize(img_source, (imgsize[1], imgsize[0])).astype(float)
            ne_array = img_source
            ne_array[:,:,0] = img_source[:,:,0] * exp_time * illumination
            ne_array[:,:,1] = img_source[:,:,1] * exp_time * illumination * green_gain
            ne_array[:,:,2] = img_source[:,:,2] * exp_time * illumination

        if len(imgsize) == 3:
            ka = [ka, ka/green_gain, ka] if type(ka) != list else ka
            nd_factor = [nd_factor, nd_factor, nd_factor] if type(nd_factor) != list else nd_factor
            exp_time = [exp_time, exp_time, exp_time] if type(exp_time) != list else exp_time
            sigma_beta = [sigma_beta, sigma_beta, sigma_beta] if type(sigma_beta) != list else sigma_beta

            simu_img = np.zeros(imgsize)
            shot_noise_rgb = shot_noise(ne_array)
            for i in range(imgsize[2]):
                simu_img[:, :, i] = pepper_sault(ka[i] * beta_c_array(sigma_beta[i], size=imgsize) * (
                        shot_noise_rgb[:, :, i]
                        + dark_current(nd_factor[i] * exp_time[i], size=imgsize[:2])
                        + read_noise(sigma_r, size=imgsize[:2])),  
                        p_sault=ka[i]*sault_p, p_pepper=ka[i]*pepper_p)
        else:
            simu_img = pepper_sault(ka * beta_c_array(sigma_beta, size=imgsize) * (
                    shot_noise(ne_array)
                    + dark_current(nd_factor * exp_time, size=imgsize)
                    + read_noise(sigma_r, size=imgsize)),  
                    p_sault=ka*sault_p, p_pepper=ka*pepper_p)

        simu_img = simu_img.astype(int) * kd
        # Quantify and Digital Gain
        # kc = ka * kd
        
        simu_img = (np.clip(simu_img, 0, 255) / 255) ** gamma * 255
        simu_img = np.clip(simu_img, 0, 255).astype(int)
        return simu_img

    def take_photo_M(self, img_source=None, exp_time=1, iso=100, aperture=2,
                   illumination_factor=1, imgsize=None, ka=2, kd=3):

        if imgsize is None:
            imgsize = img_source.shape
        img = self.simu_noise(img_source=img_source,
                              sigma_beta=self.sigma_beta,
                              nd_factor=self.nd_factor,
                              sigma_r=self.sigma_r,
                              ka=(np.array(ka)).tolist(),
                              exp_time=exp_time,
                              illumination=(1 / aperture ** 2) * illumination_factor,
                              imgsize=imgsize,
                              kd=kd)

        img = np.clip(img, 0, 255)

        return img.astype(np.uint8)
    
    def take_photo_P(self, img_source=None, imgsize=None, ka=1, kd=3):

        if imgsize is None:
            imgsize = img_source.shape
        img = self.simu_noise(img_source=img_source,
                              sigma_beta=self.sigma_beta,
                              nd_factor=self.nd_factor,
                              sigma_r=self.sigma_r,
                              ka=(np.array(ka)).tolist(),
                              exp_time=1,
                              illumination=(1/np.array(ka*kd)).tolist(),
                              imgsize=imgsize,
                              kd=kd)

        img = np.clip(img, 0, 255)

        return img.astype(np.uint8)


if __name__ == "__main__":
    model = CMOS_Camera({"sigma_beta":0.03, "nd_factor":10, "sigma_r":1})
    
    k = 40
    cleanimg = (np.ones([100,100,3]) * 180).astype(np.uint8)
    model = model.take_photo_P(img_source=cleanimg, kd=k/6, ka=4)
    gaussian = np.clip(cleanimg + np.random.normal(0, (100*k)**0.5, size=cleanimg.shape), 0, 255).astype(np.uint8)
    poisson = np.clip(np.random.poisson(lam=cleanimg/k) * k, 0, 255).astype(np.uint8)
    print(np.mean(cleanimg), np.mean(cleanimg),cleanimg.shape,cleanimg.shape)
    # ImgUtils.show_bright_images([ [ imgsource / 255.0, img / 255.0, img2 / 255.0] ], channel_first=False)

    cv2.imwrite("./pic/model.bmp", cv2.resize(cv2.cvtColor(model, cv2.COLOR_RGB2BGR), [500,500], interpolation=cv2.INTER_NEAREST))
    cv2.imwrite("./pic/poisson.bmp", cv2.resize(cv2.cvtColor(poisson, cv2.COLOR_RGB2BGR), [500,500], interpolation=cv2.INTER_NEAREST))
    cv2.imwrite("./pic/gaussian.bmp", cv2.resize(cv2.cvtColor(gaussian, cv2.COLOR_RGB2BGR), [500,500], interpolation=cv2.INTER_NEAREST))
    cv2.imwrite("./pic/clean.bmp", cv2.resize(cv2.cvtColor(cleanimg, cv2.COLOR_RGB2BGR), [500,500], interpolation=cv2.INTER_NEAREST))