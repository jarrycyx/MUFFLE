from Net.DCMAN import Dual_Channel_Attention_Net
import torch
import cv2
from Net.SOTA import fastdvdnet, fastdvdnet_map
from Net.SCMAN import Single_Channel_Attention_Net
from Net.SOTA import TOFlow
from Net.Loss import WeightedLoss
from torch import nn
import logging as L
import random
import numpy as np

def reproduc(Random_seed):
    """Make experiments reproducible
    """
    random.seed(Random_seed)
    np.random.seed(Random_seed)
    torch.manual_seed(Random_seed)
    torch.cuda.manual_seed_all(Random_seed)
    
    
def initLogging(logFilename):
    """Init for L
    """
    file = open(logFilename, encoding="utf-8", mode="w")
    L.basicConfig(
        level=L.INFO,
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S',
        stream=file)
    console = L.StreamHandler()
    console.setLevel(L.INFO)
    formatter = L.Formatter('%(asctime)s-%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    L.getLogger('').addHandler(console)

def build_network(opt, device):
    if opt.Network_name == "DCMAN":
        net = Dual_Channel_Attention_Net(
            frame_num=opt.Input_frame_num,
            Channel_merge=opt.Channel_merge,
            input_size=list(opt.Img_size), batchNorm=False)
    elif opt.Network_name == "DCMAN_Early":
        from Net.Ablation.DCMAN_Early import Dual_Channel_Attention_Net_Early
        net = Dual_Channel_Attention_Net_Early(
            frame_num=opt.Input_frame_num,
            Channel_merge=opt.Channel_merge,
            input_size=list(opt.Img_size), batchNorm=False)
    elif opt.Network_name == "DCMAN_Early_old":
        from Net.Ablation.DCMAN_Early_old import Dual_Channel_Attention_Net_Early
        net = Dual_Channel_Attention_Net_Early(
            frame_num=opt.Input_frame_num,
            Channel_merge=opt.Channel_merge,
            input_size=list(opt.Img_size), batchNorm=False)
    elif opt.Network_name == "DCMAN_Init":
        from Net.Ablation.DCMAN_Init import Dual_Channel_Attention_Net_Init
        net = Dual_Channel_Attention_Net_Init(
            frame_num=opt.Input_frame_num,
            input_size=list(opt.Img_size), batchNorm=False)
    elif opt.Network_name == "DCMAN_NoLSTM":
        from Net.Ablation.DCMAN_NoLSTM import Dual_Channel_Attention_Net_NoLSTM
        net = Dual_Channel_Attention_Net_NoLSTM(
            frame_num=opt.Input_frame_num,
            input_size=list(opt.Img_size), batchNorm=False)
    elif opt.Network_name == "SCMAN":
        net = Single_Channel_Attention_Net(
            frame_num=opt.Input_frame_num,
            input_size=list(opt.Img_size), batchNorm=False)
    elif "FastDVDnet" in opt.Network_name:
        if "map" in opt.Network_name: # FastDVDnet_map
            net = fastdvdnet_map.FastDVDnet()
        else:
            net = fastdvdnet.FastDVDnet_Fn5()
    elif "TOFlow" in opt.Network_name:
        net = TOFlow.TOFlow(h=opt.Img_size[0], 
                                    w=opt.Img_size[1], 
                                    task="denoising", cuda_flag=True)
    else:
        raise NotImplementedError
    if not device == "cpu":
        net = net.to(device)
    
    return net

def build_loss(opt):
    L.info("Using Loss: " + opt.name)
    if opt.name == "L1":
        loss_func = nn.L1Loss()
    elif opt.name == "L2":
        loss_func = nn.MSELoss()
    elif opt.name == "L2_Weighted":
        loss_func = WeightedLoss(basicLossFunc=nn.MSELoss(), **opt.params)
    else:
        raise NotImplementedError
    
    return loss_func