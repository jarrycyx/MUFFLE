import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import skimage.metrics as Metrics
import tifffile
import tqdm
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops


FILE_PREFIX = ""

def get_time_stamp():
    return str(time.strftime("%Y-%m-%d-%H%M%S", time.localtime()))

def luckyprint(data, p=1):
    seed = np.random.random()
    if seed < p:
        print(data)

def simu_simple_data(f=20, h=100, w=200):
    # f, w, h = 10, 200, 100
    data = np.zeros([f, h, w])
    for frame_index in range(f):
        data_1d = [0.45*(np.sin(5*np.pi*x/w+2*np.pi*frame_index/f) + 1) for x in range(w)]
        data_2d = np.array([data_1d for y in range(h)])
        data[frame_index] = data_2d
    return data

def simu_random_data(f=20, h=100, w=200):
    # f, w, h = 10, 200, 100
    data = np.zeros([f, h, w])
    for frame_index in range(f):
        data_2d = (np.random.rand(h, w)*0.1).tolist()
        data[frame_index] = data_2d
    return data

def show_video(data):
    plt.cla()
    for frame_idx in range(data.shape[1]):
        plt.imshow(data[0][frame_idx])
        plt.pause(0.05)
    plt.cla()
    

def read_tiff_image(path, type=".tif", crop=None, size=None, color="gray"):
    # 8 bit image
    # luckyprint(path, p=0.02)
    if (os.path.splitext(path)[-1] == type):
        if color=="gray" or color=="GRAY":
            img = tifffile.imread(path) 
        else:
            img = tifffile.imread(path) 
        if crop:
            img = img[crop[0]:crop[1],crop[2]:crop[3]]
        if size != None:
            img = cv2.resize(img, (size[1], size[0]))
        
        return img
    else:
        raise FileNotFoundError
    


def get_all_image(path, start_idx=0, max_num=100000, imgsize=(0, 0)):
    list_dir = sorted(os.listdir(path))
    list_dir = list_dir[start_idx:min(start_idx+max_num, len(list_dir))]
    img_list = [read_tiff_image(path+img_path, size=imgsize) for img_path in list_dir]
    return np.array(img_list)

def get_low_light_data():
    pass

# [images] should be 2-d list of numpy images
def show_bright_images(images, channel_first=True, figwidth=40, 
                       save_path=None, 
                       filename_prefix=FILE_PREFIX,
                       BGR=False, closeFig=True, norm=False):
    
    if channel_first:
        figheight = figwidth*len(images)/len(images[0])*images[0][0].shape[1]/images[0][0].shape[2]
    else:
        figheight = figwidth*len(images)/len(images[0])*images[0][0].shape[0]/images[0][0].shape[1]
    
    range_min = np.inf
    range_max = 0
    for image_row in images:
        for image in image_row:
            range_min = min(range_min, np.min(image))
            range_max = max(range_max, np.max(image))
    
    fig = plt.figure(figsize=(figwidth, figheight))
    for row_idx in range(len(images)):
        for col_idx in range(len(images[row_idx])):
            image = images[row_idx][col_idx]
            fig.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.subplot(len(images), len(images[0]), row_idx*len(images[0])+col_idx+1)
            if channel_first and len(image.shape)==3: # only color image has 3 channels
                image = image.transpose(1, 2, 0)
            if BGR:
                b,g,r = cv2.split(image)
                image = cv2.merge([r,g,b])
            if norm and np.mean(image)>0:
                image = image / np.mean(image) * 0.5
            if len(image.shape)==2:
                plt.imshow(np.clip(image, 0, 1), vmin=range_min, vmax=range_max, cmap='gray')
            else:
                plt.imshow(np.clip(image, 0, 1), vmin=range_min, vmax=range_max)
                
            plt.axis('off')
    
    if save_path is None:
        save_path = opj("pic", filename_prefix+get_time_stamp()+".png")
        
    if not os.path.exists(opd(save_path)):
        os.makedirs(opd(save_path))
        
    plt.colorbar()
    plt.savefig(save_path)
    if closeFig:
        plt.close()
    # plt.show()
    return save_path
    
def show_bright_image(image,channel_first=True):
    image = image/np.mean(image)*0.5
    plt.imshow(np.clip(image.transpose(1, 2, 0) if channel_first else image, 0, 1))
    plt.show()

def img_resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def compare_imgs_rgb(vid, gtr, normalized=True):
    if len(vid.shape) == 3:
        vid = [vid]
        gtr = [gtr]
    
    vid = np.array(np.clip(vid, 0, 1)).astype(float)
    gtr = np.array(np.clip(gtr, 0, 1)).astype(float)
    if normalized:
        vid = vid / np.mean(vid) * np.mean(gtr)
        vid = np.clip(vid, 0, 1)
    
    frame_num = vid.shape[0]
    psnr = []
    ssim = []
    if vid.shape[1]==3 or vid.shape[1]==1:
        vid = np.transpose(vid, (0, 2, 3, 1))
    if gtr.shape[1]==3 or gtr.shape[1]==1:
        gtr = np.transpose(gtr, (0, 2, 3, 1))
    
    for i in range(frame_num):
        psnr.append(Metrics.peak_signal_noise_ratio(gtr[i], vid[i]))
        ssim.append(Metrics.structural_similarity(gtr[i], vid[i], channel_axis=2, multichannel=True))

    return np.mean(psnr), np.mean(ssim)


def compare_imgs_gray(vid, gtr, normalized=True):
    vid = np.array(np.clip(vid, 0, 1))
    gtr = np.array(np.clip(gtr, 0, 1))
    if normalized:
        vid = vid / np.mean(vid) * np.mean(gtr)
        vid = np.clip(vid, 0, 1)
    
    frame_num = vid.shape[0]
    psnr = []
    ssim = []
    
    for i in range(frame_num):
        psnr.append(Metrics.peak_signal_noise_ratio(gtr[i], vid[i], data_range=1))
        ssim.append(Metrics.structural_similarity(gtr[i], vid[i], data_range=1))

    return np.mean(psnr), np.mean(ssim)


def white_balance(img):
    '''
    第一种简单的求均值白平衡法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    # 读取图像
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img

def burst2vid(file_dir, vid_save_dir, fr_size=[800, 400], use_tqdm=False):
    # file_dir = '/Volumes/WD-SN550/Dual_Channel_Low_Light/HIK_IMG_DIR/Afternoon/Afternoon_0.010/USB[0]00F58881811/2021_05_07_17_52_49/'
    # vid_save_dir = '/Volumes/WD-SN550/Dual_Channel_Low_Light/HIK_IMG_DIR/Afternoon/Afternoon_0.010/vid.avi'
    list = []
    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            list.append(file)

    video = cv2.VideoWriter(vid_save_dir, cv2.VideoWriter_fourcc(*'MJPG'), 24, tuple(fr_size))  # 定义保存视频目录名称及压缩格式，fps=24,像素为1280*720
    
    range_method = tqdm.trange if use_tqdm else range
    for i in range_method(1, len(list)):
        img = cv2.imread(file_dir+list[i-1])  # 读取图片
        img = cv2.resize(img, tuple(fr_size))  # 将图片转换为1280*720
        img_enhanced = img #.astype(float)*70/np.mean(img)
        video.write(np.clip(img_enhanced, 0, 255).astype(np.uint8))  # 写入视频

    video.release()
    
    
if __name__=="__main__":
    burst2vid("D:/rgb_nir_low_light_dataset_v6_1116/0002/rgb/", "D:/rgb_nir_low_light_dataset_v6_1116/0002/rgb_raw.avi", fr_size=[640,360])
