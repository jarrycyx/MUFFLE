import os
from multiprocessing.pool import Pool
from os.path import join as opj
import argparse
import numpy as np
import cv2
from scipy import interpolate
import sys
import scipy.io as scio
import time
from shutil import get_terminal_size


SOURCE_TRAIN_PATH = "/data/zb/proj/Flow-Guided-Feature-Aggregation/data/ILSVRC2015/Data/VID/train/"
SOURCE_VAL_PATH = "/data/zb/proj/Flow-Guided-Feature-Aggregation/data/ILSVRC2015/Data/VID/train/"
SOURCE_TEST_PATH = "/data/zb/proj/Flow-Guided-Feature-Aggregation/data/ILSVRC2015/Data/VID/train/"


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()
        
class LLDImgCreator(object):
    
    def __init__(self):
        r_path = opj(os.path.dirname(__file__),'LUT/r.mat')
        g_path = opj(os.path.dirname(__file__),'LUT/g.mat')
        b_path = opj(os.path.dirname(__file__),'LUT/b.mat')
        self.r, self.g, self.b, r_r, g_r, b_r = self.read_rgb_mat(r_path, g_path, b_path)
         
    def create_lld(self, img_normal, ap, avgNorm=True):
        img_16 = self.gama_trans(img_normal, self.r, self.g, self.b)  # 8-->16
        imgsave= self.setap_addnoise(img_16, ap)  # lowlight sim
        # TODO 到底是存16位还是8位？ 8位的话小的值都不行
        # png.from_array(img_16.astype(np.uint16), 'RGB;16').save(tarpath.replace('jpg','png'))
        # with open(tarpath.replace('jpg','png'), 'wb') as f:
        #     writer = png.Writer(width=img_16.shape[1], height=img_16.shape[0], bitdepth=16,)
        #     z2list = img_16.reshape(-1, img_16.shape[1] * img_16.shape[2]).tolist()
        #     writer.write(f, z2list)
        
        if avgNorm:
            # imgsave[:, :, 0] = imgsave[:, :, 0] / np.mean(imgsave[:, :, 0]) * np.mean(img_normal[:, :, 0])
            # imgsave[:, :, 1] = imgsave[:, :, 1] / np.mean(imgsave[:, :, 1]) * np.mean(img_normal[:, :, 1])
            # imgsave[:, :, 2] = imgsave[:, :, 2] / np.mean(imgsave[:, :, 2]) * np.mean(img_normal[:, :, 2])
            imgsave = imgsave / np.mean(imgsave) * np.mean(img_normal)
        
        return imgsave
        
    def read_rgb_mat(self, r_path,g_path,b_path):
        r = scio.loadmat(r_path)['r'][0]#+256*3
        g = scio.loadmat(g_path)['g'][0]#+256*3
        b = scio.loadmat(b_path)['b'][0]#+256*3
        funcb = interpolate.UnivariateSpline(np.arange(29, 100), b[29:100], s=0)
        b[5:29] = funcb(np.arange(5,29))
        r_r_my = np.interp(np.arange(65536), r, np.arange(256), left=0, right=255)
        g_r_my = np.interp(np.arange(65536), g, np.arange(256), left=0, right=255)
        b_r_my = np.interp(np.arange(65536), b, np.arange(256), left=0, right=255)
        return r,g,b,r_r_my,g_r_my,b_r_my
    
    def gama_trans(self, img_normal,r,g,b):
        img_trains = np.zeros(img_normal.shape).astype(np.float32)
        img_trains[:, :, 0] = np.take(b, img_normal[:, :, 0])
        img_trains[:, :, 1] = np.take(g, img_normal[:, :, 1])
        img_trains[:, :, 2] = np.take(r, img_normal[:, :, 2])
        return img_trains
    
    def setap_addnoise(self, img_value, ap):
        # img_value: in every pix it consist a value number(gray scale)
        # img_photon: in every pix it consist a photon number
        [H, W, C] = img_value.shape
        #### 1. value --> (fixed or manually set) photon
        total_photon = ap * W * H
        total_value = img_value.sum()
        photon_per_value = total_photon / total_value
        img_at_fixed_ap_photon = img_value * photon_per_value
        #### 2. add noise (related to photon)
        QE = 0.9
        readout = 139
        c = 0.00145 # spurious charge
        e_per_edu = 0.1866
        baseline = 500
        EMgain = 300
        img_at_fixed_ap_photon_electric = QE * img_at_fixed_ap_photon + c
        img_at_fixed_ap_photon_electric[img_at_fixed_ap_photon_electric<1e-6] = 1e-6
        n_ie = np.random.poisson(lam=img_at_fixed_ap_photon_electric, size=[H,W,C])
        n_oe = np.random.gamma(shape=n_ie, scale=EMgain, size=[H,W,C])
        #### 3. add noise (readout)
        n_oe = n_oe + np.random.normal(loc=0, scale=readout, size=[H,W,C])
        #### 4. photon --> value
        ADU_out =  np.dot(n_oe, e_per_edu) + baseline
        return ADU_out

if __name__ == "__main__":
    img = (np.random.random((5, 5, 3))*255).astype(np.uint8)
    print(img)
    
    print("------")
    
    myLLDImgCreator = LLDImgCreator()
    NoisyImg = myLLDImgCreator.create_lld(img, 1)
    print(NoisyImg)
    