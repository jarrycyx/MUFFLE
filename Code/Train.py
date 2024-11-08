import sys
import os

from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

import random
from datetime import datetime
import logging as L
import argparse
import skimage.measure as Measure
from TrainUtils.MinibatchGenerater import MinibatchGenerater
import numpy as np
import ImgProcessing.ImgUtils as ImgUtils
import torch
import cv2
from copy import deepcopy

from tensorboardX import SummaryWriter

from omegaconf import OmegaConf
from TrainUtils.TrainUtils import *
from TrainUtils.OptType import TrainOpt


class Train(object):

    def __init__(self, opt: TrainOpt, save_path, 
                 save_model_every=5, valid_every_n_batch=20, gpu=[0]):
        
        if not os.path.exists(save_path):
            os.makedirs(opj(save_path, "pklmodels"))
            os.makedirs(opj(save_path, "pic"))
        torch.cuda.set_device(gpu[0])
        initLogging(opj(save_path, "train.log"))
        
        OmegaConf.save(opt, opj(save_path, "opt.yaml"))
        self.opt = opt
        self.PID = os.getpid()
        self.gpu = gpu
        self.save_model_every = save_model_every
        self.save_path = save_path
        self.writer = SummaryWriter(logdir=opj(save_path))
        self.VALID_EVERY = valid_every_n_batch

        self.dual_channel = ("DCMAN" in self.opt.Network_name)
        self.net = build_network(self.opt, f"cuda:{self.gpu[0]}")
        self.loss_func = build_loss(self.opt.Loss)
        self.net = torch.nn.DataParallel(self.net, device_ids=self.gpu).cuda()
        L.info("Net Loaded")
        
        self.start_iter_idx = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.LR)
        
        if hasattr(self.opt, "LR_Scheduler"):
            if self.opt.LR_Scheduler == "StepLR":
                self.variableLR = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.opt.LR_Scheduler_params)
                L.info("Using " + self.opt.LR_Scheduler + str(self.opt.LR_Scheduler_params))
            elif self.opt.LR_Scheduler == "MultiStepLR":
                self.variableLR = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **self.opt.LR_Scheduler_params)
                L.info("Using " + self.opt.LR_Scheduler + str(self.opt.LR_Scheduler_params))
        else:        
            self.variableLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
            L.warning("Using default LR scheduler")
            
    
    def resume_training(self, iter_idx, pkl_path):
        if iter_idx > 0:
            device = torch.device(
                self.gpu[0] if torch.cuda.is_available() else "cpu")
            self.net.load_state_dict(torch.load(pkl_path, map_location=device))
            L.info("Resume training")
        self.start_iter_idx = iter_idx
    
    def valid(self, valid_index):
        
        def compare_brightness(image, target, perc=99):
            image_brightness = np.percentile(image, perc)
            target_brightness = np.percentile(target, perc)
            return image_brightness / target_brightness
        
        self.net.eval()
        # frames number before or after this current frame
        half_num = (self.opt.Input_frame_num-1)//2
        
        generater = MinibatchGenerater(Img_size=self.opt.Img_size, dataset_opt=self.opt.Dataset,
                                       down_sample=1)
        ch_a_noisy, ch_b_noisy, ch_a_label, ch_b_label = generater.generate_test(self.opt.Input_frame_num,
                                                                                 file_index=valid_index,
                                                                                 gain="random")
        
        tensor_a = torch.from_numpy(ch_a_noisy).cuda().float()
        tensor_b = torch.from_numpy(ch_b_noisy).cuda().float()
        
        if self.dual_channel:
            a_out_list, b_out_list = self.net(tensor_a, tensor_b)
        else:
            a_out_list = self.net(tensor_a)
            b_out_list = a_out_list
            
        a_pred = a_out_list.detach().cpu().numpy()
        b_pred = b_out_list.detach().cpu().numpy()
        
        fname = ImgUtils.show_bright_images(
            [[ ch_a_noisy[0][half_num],  a_pred[0],  ch_a_label[0][half_num]  ],
             [ ch_b_noisy[0][half_num],  b_pred[0],  ch_b_label[0][half_num]  ]],
            save_path=opj(self.save_path, "pic", "epoch_{:03d}".format(self.epoch_idx), 
                          "valid_step{:d}".format(self.global_step)),
            norm=False)

        L.info("Save figure: " + fname)
        score_ch_a_pred2label = ImgUtils.compare_imgs_gray(a_pred[0], ch_a_label[0][half_num])
        score_ch_a_noisy2label = ImgUtils.compare_imgs_gray(ch_a_noisy[0][half_num], ch_a_label[0][half_num])
        score_ch_b_pred2label = ImgUtils.compare_imgs_gray(b_pred[0], ch_b_label[0][half_num])
        score_ch_b_noisy2label = ImgUtils.compare_imgs_gray(ch_b_noisy[0][half_num], ch_b_label[0][half_num])
        
        self.writer.add_scalar(tag="psnr_ch_a/noisy2label", scalar_value=score_ch_a_noisy2label[0], global_step=self.global_step)
        self.writer.add_scalar(tag="ssim_ch_a/noisy2label", scalar_value=score_ch_a_noisy2label[1], global_step=self.global_step)
        self.writer.add_scalar(tag="psnr_ch_a/pred2label", scalar_value=score_ch_a_pred2label[0], global_step=self.global_step)
        self.writer.add_scalar(tag="ssim_ch_a/pred2label", scalar_value=score_ch_a_pred2label[1], global_step=self.global_step)
        
        self.writer.add_scalar(tag="psnr_ch_b/noisy2label", scalar_value=score_ch_b_noisy2label[0], global_step=self.global_step)
        self.writer.add_scalar(tag="ssim_ch_b/noisy2label", scalar_value=score_ch_b_noisy2label[1], global_step=self.global_step)
        self.writer.add_scalar(tag="psnr_ch_b/pred2label", scalar_value=score_ch_b_pred2label[0], global_step=self.global_step)
        self.writer.add_scalar(tag="ssim_ch_b/pred2label", scalar_value=score_ch_b_pred2label[1], global_step=self.global_step)
        
        self.writer.add_scalar(tag="contrast/ch_a_brightness_per99", scalar_value=compare_brightness(a_pred[0], ch_a_label[0][half_num]), global_step=self.global_step)
        self.writer.add_scalar(tag="contrast/ch_b_brightness_per99", scalar_value=compare_brightness(b_pred[0], ch_b_label[0][half_num]), global_step=self.global_step)
        
        return [score_ch_a_pred2label, score_ch_a_noisy2label, score_ch_b_pred2label, score_ch_b_noisy2label]

    def train_step(self):
        self.net.train()
        half_num = (self.opt.Input_frame_num-1)//2
        # half_num = 3 if LSTM net get 7 frames
        
        dsample = np.random.uniform(low=self.opt.Simulation_dsample[0], high=self.opt.Simulation_dsample[1])
        
        generater = MinibatchGenerater(Img_size=self.opt.Img_size, dataset_opt=self.opt.Dataset,
                                       down_sample=dsample)
        ch_a_noisy, ch_b_noisy, ch_a_label, ch_b_label = generater.generate_batch(
            self.opt.Batch_size, 
            self.opt.Input_frame_num,
            dataset_type="train",
            gain="random"
        )
        
        self.optimizer.zero_grad()
            
        tensor_a = torch.from_numpy(ch_a_noisy).cuda().float()
        tensor_b = torch.from_numpy(ch_b_noisy).cuda().float()
        tensor_a_label = torch.from_numpy(ch_a_label).cuda().float()
        tensor_b_label = torch.from_numpy(ch_b_label).cuda().float()
        
        if self.dual_channel:
            a_out_list, b_out_list = self.net(tensor_a, tensor_b)
            # b_weight = torch.mean(tensor_a) / torch.mean(tensor_b)
            
            loss_a = self.loss_func(a_out_list, tensor_a_label[:, half_num])
            loss_b = self.loss_func(b_out_list, tensor_b_label[:, half_num])
                
            loss = loss_a + loss_b #  * b_weight
            self.writer.add_scalar(tag="loss_train/all", scalar_value=loss.item(), global_step=self.global_step)
            self.writer.add_scalar(tag="loss_train/ch_a", scalar_value=loss_a.item(), global_step=self.global_step)
            self.writer.add_scalar(tag="loss_train/ch_b", scalar_value=loss_b.item(), global_step=self.global_step)
        else:
            a_out_list = self.net(tensor_a)
            b_out_list = a_out_list
            loss = self.loss_func(a_out_list, tensor_a_label[:, half_num])
            self.writer.add_scalar(tag="loss_train/ch_a", scalar_value=loss.item(), global_step=self.global_step)

        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self):
        L.info("Current PID: " + str(os.getpid()))
        torch.cuda.empty_cache()
        for epoch_idx in range(self.start_iter_idx):
            self.variableLR.step()
            L.info("Skip epoch %d" % (epoch_idx+1))
            
        for epoch_idx in range(self.start_iter_idx, self.opt.Iter_num):
            self.epoch_idx = epoch_idx
            ch_a_psnr_list = []
            ch_b_psnr_list = []
            ch_a_ssim_list = []
            ch_b_ssim_list = []
            for i in range(self.opt.Epoch_size):
                self.global_step = (epoch_idx*self.opt.Epoch_size + i) * self.opt.Batch_size
                loss = self.train_step()
                this_LR = self.optimizer.param_groups[0]['lr']
                
                if i % self.VALID_EVERY == 0 and i != 0:
                    # score = self.valid(i // self.VALID_EVERY)
                    score = self.valid(np.random.randint(0, 1000))
                    
                    score_ch_a_pred2label, score_ch_a_noisy2label, score_ch_b_pred2label, score_ch_b_noisy2label = tuple(score)
                    ch_a_psnr_list.append(score_ch_a_pred2label[0])
                    ch_b_psnr_list.append(score_ch_b_pred2label[0])
                    ch_a_ssim_list.append(score_ch_a_pred2label[1])
                    ch_b_ssim_list.append(score_ch_b_pred2label[1])

                    valstr_ch_a = ('ch_a Channel Score: {:.2f}/{:.4} - {:.2f}/{:.4}').format(
                        score_ch_a_pred2label[0], score_ch_a_pred2label[1],
                        score_ch_a_noisy2label[0], score_ch_a_noisy2label[1])
                    
                    valstr_ch_b = ('ch_b Channel Score: {:.2f}/{:.4} - {:.2f}/{:.4}').format(
                        score_ch_b_pred2label[0], score_ch_b_pred2label[1],
                        score_ch_b_noisy2label[0], score_ch_b_noisy2label[1])
                    L.info('Train Evaluation: (Pred PSNR / Pred SSIM - Noisy PSNR / Noisy SSIM)')
                    L.info(valstr_ch_a)
                    L.info(valstr_ch_b)

                logstr = 'iter: [{:d}/{:d}], batch: [{:d}/{:d}], loss: {:0.6f}, lr: {:.7f}'.format(
                    epoch_idx + 1, self.opt.Iter_num,
                    (i + 1), self.opt.Epoch_size,
                    loss, this_LR)
                
                if i % 10 == 0 and i != 0:
                    L.info(logstr)
                    
                # torch.save(self.net.state_dict(), opj(self.save_path, "pklmodels", "train_iter_{}".format(str(epoch_idx+1)+".pkl")))
                
            ##########################################################
            # epoch finished
                    
            if epoch_idx % self.save_model_every == self.save_model_every - 1:
                model_path = opj(self.save_path, "pklmodels", "train_iter_{}".format(str(epoch_idx+1)+".pkl"))
                torch.save(self.net.state_dict(), model_path)
                L.info("Saving Model...")
            
            L.info('Epoch Evaluation: Ave ch_a PSNR = {:.2f}, Ave ch_a SSIM={:.4f}'.format(
                np.mean(ch_a_psnr_list), np.mean(ch_a_ssim_list)
            ))
            L.info('Epoch Evaluation: Ave ch_b PSNR = {:.2f}, Ave ch_b SSIM={:.4f}'.format(
                np.mean(ch_b_psnr_list), np.mean(ch_b_ssim_list)
            ))
            
            self.variableLR.step()
            torch.cuda.empty_cache()
        return model_path


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # clear previous output files/images
    # save figure to file every x epoch
    parser.add_argument('-valid_every_n_batch', type=int, default=1)
    parser.add_argument('-load_checkpoint_pkl', type=str, default="")
    parser.add_argument('-resume_iter', type=int, default=0)
    parser.add_argument('-save_model_every_n_iter', type=int, default=5)
    parser.add_argument('-opt', type=str, default="Exp/Data0511/DCMAN_Concat_Fn15_HJHS_ALL.yaml")
    parser.add_argument('-gpu', type=str, default="0,1")
    
    args = parser.parse_args()
    gpu_list = [int(gpu) for gpu in args.gpu.split(",")]
    
    opt: TrainOpt = OmegaConf.load(args.opt)
    timetag = datetime.now().strftime("%Y_%m%d_%H%M%S_%f")
    
    reproduc(opt.Random_seed)
    save_dir = "./outputs/train_process/{}_{}".format(opt.Proj_name, timetag)
    my_train = Train(opt=opt, save_model_every=args.save_model_every_n_iter,
                     save_path=save_dir, 
                     valid_every_n_batch=args.valid_every_n_batch, 
                     gpu=gpu_list)
    if args.load_checkpoint_pkl != "":
        my_train.resume_training(args.resume_iter, args.load_checkpoint_pkl)

    final_model_path = my_train.train()
    # final_model_path = "1"
    
    
    # Test
    import subprocess
    command = ["python", "Test.py", 
               "-opt", args.opt, "-load_pkl", final_model_path, "-gpu", args.gpu]
    subprocess.run("gpustat")
    subprocess.run(command)
