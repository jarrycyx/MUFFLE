import sys
import os

from os.path import join as opj
from os.path import dirname as opd

import tifffile
import numpy as np
import time
from omegaconf import OmegaConf
import torch

from Test import DualChannelLowLight

from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
import time
import sys


class ReconstructThread(QThread):
    resultReady = pyqtSignal(tuple)
    updateProgress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, parent, opt_path, model_path, ch_a_numpy, ch_b_numpy, selected_device, enhance_progress, output_path):
        super().__init__(parent=parent)
        self.opt_path = opt_path
        self.model_path = model_path
        self.ch_a_numpy = ch_a_numpy
        self.ch_b_numpy = ch_b_numpy
        self.selected_device = selected_device
        self.enhance_progress = enhance_progress
        self.output_path = output_path
        

    def run(self):
        # if True:
        try:
            # 模拟一个耗时的计算操作
            print(f"Reconstruct videos with device: {self.selected_device}")
            self.updateProgress.emit(5)
            if not self.selected_device == "cpu":
                torch.cuda.set_device(self.selected_device)
            opt = OmegaConf.load(self.opt_path)
            model = DualChannelLowLight(opt, self.model_path, device=self.selected_device, imgsize=self.ch_a_numpy.shape[1:])
            self.updateProgress.emit(10)
            in_a, in_b = self.ch_a_numpy[None, :, None, :, :], self.ch_b_numpy[None, :, None, :, :]
            print(f"in_a.shape: {in_a.shape}. in_a mean: {np.mean(in_a)}, in_b.shape: {in_b.shape}. in_b mean: {np.mean(in_b)}")
            model.load_low_light_video(in_a, in_b)
            out_a, out_b = model.start_nn(progress_recall=lambda x: self.updateProgress.emit(int(x)))
            # out_a, out_b = model.start_nn(progress_recall=None)
            print(out_a.shape, out_b.shape)
            ch_a_out_numpy = out_a[:, 0, :, :]
            ch_b_out_numpy = out_b[:, 0, :, :]
            self.updateProgress.emit(100)
            
            time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            result_dir = opj(self.output_path, f"MUFFLE_result_{time_str}")
            os.makedirs(result_dir, exist_ok=True)
            ch_a_save = np.clip(ch_a_out_numpy * 65535, 0, 65535).astype(np.uint16)
            ch_b_save = np.clip(ch_b_out_numpy * 65535, 0, 65535).astype(np.uint16)
            tifffile.imwrite(opj(result_dir, f"Acceptor_enhanced.tif"), ch_a_save)
            tifffile.imwrite(opj(result_dir, f"Donor_enhanced.tif"), ch_b_save)
            
            self.resultReady.emit((ch_a_out_numpy, ch_b_out_numpy, result_dir))
        except Exception as e:
            traceback_info = sys.exc_info()
            self.error.emit(str(e) + "\n" + str(traceback_info))
            print(e)




def load_tiff_seq(path):
    """Load a sequence of TIFF files from a given directory path.

    Args:
        path (str): Path to the directory containing the TIFF files.

    Returns:
        np.ndarray: A 3D array containing the TIFF files.
    """
    if os.path.isdir(path):
        files = os.listdir(path)
        files.sort()
        files = [f for f in files if f.endswith(".tif")]
        files = [opj(path, f) for f in files]
        img = tifffile.imread(files)
    else:
        img = tifffile.imread(path)
        
    # 标准化到0-1，根据数据类型确定最大值
    if img.dtype == np.uint8:
        img = img / 255
    elif img.dtype == np.uint16:
        img = img / 65535
    else:
        raise ValueError("Unknown data type: {}".format(img.dtype))
    
    if len(img.shape) == 4:
        if img.shape[3] == 1:
            img = img[:, :, :, 0]
        else:
            raise ValueError("The input image is not a 3D image.")
    
    return img