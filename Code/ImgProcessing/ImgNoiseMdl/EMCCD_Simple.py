import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def value(ptonpx, h, w):
    QE = 0.9
    # readout = 74.4
    readout = 139
    c = 0.00145 # spurious charge
    e_per_edu = 0.1866
    baseline = 50
    EMgain = 300
    temp = QE * ptonpx + c
    temp[temp<1e-6] = 1e-6
    n_ie = np.random.poisson(lam=temp, size=[h,w]) # random('Poisson',max(1e-6,(QE * ptonpx + c)),[h, w])
    temp2 = n_ie
    temp2[temp2<1e-6] = 1e-6
    temp2[temp2>1e5] = 1e5
    n_oe = np.random.gamma(shape=temp2, scale=EMgain, size=[h,w]) # random('Gamma', max(1e-6,n_ie), EMgain)
    n_oe = n_oe + np.random.normal(loc=0, scale=readout, size=[h,w]) # ('Normal', baseline, readout)
    ADU_out = np.dot(n_oe, e_per_edu) + baseline
    # saturation = 2 ** 16
    # dn = floor(min(saturation, max(0, ADU_out)))
    return ADU_out

def addnoise(img, ap):
    [h, w, c] = img.shape
    values = np.zeros([h,w,c])
    pnumav = ap # average photon num per pixel
    area = pnumav * w * h * 3 # total photon number
    # img_gray = 0.3 * img[:,:,2] + 0.59 * img[:,:,1] + 0.11 * img[:,:,0]
    total = img.sum(axis=0).sum(axis=0).sum(axis=0)
    ptonpx = area / total * img if  not total == 0 else img
    # print('ratio = %f' % (area / total * 50.382))
    values[:,:,0] = value(ptonpx[:,:,0], h, w)
    values[:,:,1] = value(ptonpx[:,:,1], h, w)
    values[:,:,2] = value(ptonpx[:,:,2], h, w)
    img_save = values.astype(np.uint16)
    # img_save = np.clip(img_save, 0, 2**16)
    img_save = np.clip(img_save, 0, 2*np.mean(img_save))
    img_save_brightness_normed = np.zeros(img.shape)
    img_save_brightness_normed[:, :, 0] = np.clip(img_save[:, :, 0] / np.mean(img_save[:, :, 0]) * np.mean(img[:, :, 0]), 0, 255)
    img_save_brightness_normed[:, :, 1] = np.clip(img_save[:, :, 1] / np.mean(img_save[:, :, 1]) * np.mean(img[:, :, 1]), 0, 255)
    img_save_brightness_normed[:, :, 2] = np.clip(img_save[:, :, 2] / np.mean(img_save[:, :, 2]) * np.mean(img[:, :, 2]), 0, 255)
    
    return img_save_brightness_normed

if __name__ == "__main__":
    img_path = './Backups/ILSVRC/000173.JPEG'
    save_path = './Backups/ILSVRC/000173_Noisy.JPEG'
    img = cv2.imread(img_path) # cv2读取为BGR格式
    plt.show()
    ap = 1
    img_new = addnoise(img, ap)
    # img_new = cv2.resize(img_new, (200, 100))
    cv2.imwrite(save_path, img_new)
    plt.imshow(np.clip(img_new, 0, 255))
    plt.show()