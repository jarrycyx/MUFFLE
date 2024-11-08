import ImgProcessing.ImgUtils as ImgUtils
from TrainUtils.MinibatchGenerater import MinibatchGenerater

import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

import tifffile
from TrainUtils.TrainUtils import *
from TrainUtils.OptType import TrainOpt
from omegaconf import OmegaConf
import argparse
import tqdm
import cv2
import numpy as np
import shutil

import logging as L


class DualChannelLowLight(object):
    def __init__(self, opt, pklpath, device="cuda", imgsize=[200, 400]):
        self.imgsize = imgsize
        self.opt = opt
        self.opt.Img_size = imgsize
        L.info("CUDA Device Count: " + str(torch.cuda.device_count()))
        # L.info("Primary device %d: %s" % (devices[0], torch.cuda.get_device_name(devices[0])))

        
        # torch.cuda.set_device(devices[0])
        self.device = device
        self.net = build_network(self.opt, device)
        if "cpu" in device:
            model_file = torch.load(pklpath, map_location=self.device)
            self.net.load_state_dict({k.replace("module.", ""): v for k, v in model_file.items()})
        else:
            self.net = torch.nn.DataParallel(self.net, device_ids=[self.device])
            self.net.load_state_dict(torch.load(pklpath, map_location=self.device))


        self.net.eval()
        self.dual_channel = "DCMAN" in self.opt.Network_name
        L.info("Model file loaded!")

    def load_low_light_video(self, ch_a, ch_b):
        # print(ch_a.shape, ch_b.shape)
        self.ch_a_noisy = ch_a
        self.ch_b_noisy = ch_b

    def gen_test_batch(self, frames, fidx, N):
        half_num = int((N - 1) / 2)
        fnum = frames.shape[0]
        if fidx < half_num:
            # use frame 0,0,1,2,3,4,5 if fidx==2
            res = [frames[0:1]] * (half_num - fidx)
            res += [frames[: fidx + half_num + 1]]
        elif fidx > fnum - half_num - 1:
            # use frame 11,12,13,14,15,15,15 if fidx==14 (fnum=15)
            res = [frames[fidx - half_num :]]
            res += [frames[fnum - 1 : fnum]] * (fidx - fnum + half_num + 1)
        else:
            # use frame 3,4,5,6,7,8,9 if fidx==6 (fnum=15)
            res = [frames[fidx - half_num : fidx + half_num + 1]]

        return np.concatenate(res, axis=0)

    def start_nn(self, savepath=None, gain=1, progress_recall=None):
        def denoise_one_frame(channel_a, channel_b):
            tensor_a = torch.from_numpy(channel_a).to(self.device).float().unsqueeze(0)
            tensor_b = torch.from_numpy(channel_b).to(self.device).float().unsqueeze(0)

            if torch.mean(tensor_a) > 1e-5 and torch.mean(tensor_b) > 1e-5:
                a_gain = torch.mean(tensor_b) / torch.mean(tensor_a)
            else:
                a_gain = 1
            # print(f"Gain of channel a: {a_gain:.2f}")
            tensor_a = torch.clip(tensor_a * a_gain * gain, 0, 1)
            tensor_b = torch.clip(tensor_b * gain, 0, 1)
            # tensor_b = torch.clip(tensor_b * torch.mean(tensor_a) / torch.mean(tensor_b), 0, 1)
            with torch.no_grad():
                if self.dual_channel:
                    out_a, out_b = self.net(tensor_a, tensor_b)
                else:
                    out_a = self.net(tensor_a)
                    out_b = out_a
            out_a /= a_gain
            preds_a = out_a.detach().cpu().numpy()[0]
            preds_b = out_b.detach().cpu().numpy()[0]
            return preds_a, preds_b

        vid_fnum = self.ch_b_noisy.shape[1]
        half_num = (self.opt.Input_frame_num - 1) // 2

        ch_a_preds = []
        ch_b_preds = []
        for fidx in range(vid_fnum):
            channel_ch_a = self.gen_test_batch(self.ch_a_noisy[0], fidx, self.opt.Input_frame_num)
            channel_ch_b = self.gen_test_batch(self.ch_b_noisy[0], fidx, self.opt.Input_frame_num)
            preds_ch_a, preds_ch_b = denoise_one_frame(channel_ch_a, channel_ch_b)
            ch_a_preds.append(preds_ch_a)
            ch_b_preds.append(preds_ch_b)
            if fidx % 300 == 0:
                print(f"Frame {fidx} done.")
            if progress_recall is not None:
                progress_recall(10 + 90 * fidx / vid_fnum)

        ch_a_preds = np.array(ch_a_preds)
        ch_b_preds = np.array(ch_b_preds)
        return ch_a_preds, ch_b_preds


def make_tif(img):
    img = img[3:-3].transpose(0, 2, 3, 1)
    img = np.clip(img, 0, 1)
    return (img * 65535).astype(np.uint16)


def make_mip(img, norm=True):
    img = img[3:-3].transpose(0, 2, 3, 1)
    img = np.clip(img, 0, 1)
    img = (img * 65535).astype(np.uint16)
    mip = np.max(img, axis=0)
    if norm:
        l, r = np.percentile(mip, [1, 99.9])
        print(f"mip: {l:.2f} - {r:.2f}")
        mip = np.clip((mip - l) / (r - l) * 255, 0, 255).astype(np.uint8)
    return mip


def test_simu_video(
    opt: TrainOpt,
    model_path,
    dataset="test",
    dir_name=None,
    index=0,
    vid_path="./outputs/test_res/simu/",
    imgsize=[720, 1280],
    max_n_frame=20000,
):

    VID_FRAME_NUM = max_n_frame
    print("Generating data...")
    generater = MinibatchGenerater(Img_size=imgsize, dataset_opt=opt.Dataset, allow_fewer_frame_num=True)
    ch_a_noisy, ch_b_noisy, ch_a_label, ch_b_label = generater.generate_batch(
        Batch_size=1,
        # file_index=index,
        dir_name=dir_name,
        frame_num=VID_FRAME_NUM,
        crop="none",
        dataset_type=dataset,
        norm=False,
        gain="norm"
    )

    gain = np.mean(ch_b_label) / np.mean(ch_b_noisy)
    print("Gain: ", gain)
    # ch_a_noisy = ch_a_noisy * gain
    # ch_b_noisy = ch_b_noisy * gain

    if os.path.exists(vid_path):
        shutil.rmtree(vid_path)

    os.makedirs(vid_path, exist_ok=True)
    os.makedirs(opj(vid_path, "3d"), exist_ok=True)
    os.makedirs(opj(vid_path, "mip"), exist_ok=True)
    tifffile.imwrite(opj(vid_path, "3d", "ch_a_noisy.tif"), make_tif(ch_a_noisy[0]))
    tifffile.imwrite(opj(vid_path, "3d", "ch_b_noisy.tif"), make_tif(ch_b_noisy[0]))
    tifffile.imwrite(opj(vid_path, "3d", "ch_a_label.tif"), make_tif(ch_a_label[0]))
    tifffile.imwrite(opj(vid_path, "3d", "ch_b_label.tif"), make_tif(ch_b_label[0]))
    tifffile.imwrite(opj(vid_path, "mip", "ch_a_noisy.tif"), make_mip(ch_a_noisy[0]))
    tifffile.imwrite(opj(vid_path, "mip", "ch_b_noisy.tif"), make_mip(ch_b_noisy[0]))
    tifffile.imwrite(opj(vid_path, "mip", "ch_a_label.tif"), make_mip(ch_a_label[0]))
    tifffile.imwrite(opj(vid_path, "mip", "ch_b_label.tif"), make_mip(ch_b_label[0]))

    DCLL = DualChannelLowLight(opt, model_path, device="cuda", imgsize=imgsize)
    DCLL.load_low_light_video(ch_a_noisy, ch_b_noisy)
    ch_a_preds, ch_b_preds = DCLL.start_nn(savepath=vid_path, gain=gain)
    tifffile.imwrite(opj(vid_path, "3d", "ch_a_preds.tif"), make_tif(ch_a_preds))
    tifffile.imwrite(opj(vid_path, "3d", "ch_b_preds.tif"), make_tif(ch_b_preds))
    tifffile.imwrite(opj(vid_path, "mip", "ch_a_preds.tif"), make_mip(ch_a_preds))
    tifffile.imwrite(opj(vid_path, "mip", "ch_b_preds.tif"), make_mip(ch_b_preds))

    # compare does not include first and last 3 frames
    ch_a_pred_psnr, ch_a_pred_ssim = ImgUtils.compare_imgs_gray(ch_a_preds[3:-3, 0], ch_a_label[0, 3:-3, 0])
    ch_a_noisy_psnr, ch_a_noisy_ssim = ImgUtils.compare_imgs_gray(ch_a_noisy[0, 3:-3, 0], ch_a_label[0, 3:-3, 0])
    ch_b_pred_psnr, ch_b_pred_ssim = ImgUtils.compare_imgs_gray(ch_b_preds[3:-3, 0], ch_b_label[0, 3:-3, 0])
    ch_b_noisy_psnr, ch_b_noisy_ssim = ImgUtils.compare_imgs_gray(ch_b_noisy[0, 3:-3, 0], ch_b_label[0, 3:-3, 0])

    valstr_ch_a_ = ("ch_a_ Channel Score: {:.2f}/{:.4} - {:.2f}/{:.4} \n").format(ch_a_pred_psnr, ch_a_pred_ssim, ch_a_noisy_psnr, ch_a_noisy_ssim)
    valstr_ch_b_ = ("ch_b_ Channel Score: {:.2f}/{:.4} - {:.2f}/{:.4} \n").format(ch_b_pred_psnr, ch_b_pred_ssim, ch_b_noisy_psnr, ch_b_noisy_ssim)
    info_str = "PATH: %s \n Train Evaluation: (Pred PSNR / Pred SSIM - Noisy PSNR / Noisy SSIM)\n" % vid_path + valstr_ch_a_ + valstr_ch_b_

    L.info(info_str)
    return (
        ch_a_pred_psnr,
        ch_a_noisy_psnr,
        ch_a_pred_ssim,
        ch_a_noisy_ssim,
        ch_b_pred_psnr,
        ch_b_noisy_psnr,
        ch_b_pred_ssim,
        ch_b_noisy_ssim,
    )


def test_real_video(
    opt: TrainOpt,
    model_path,
    dataset="test",
    dir_name=None,
    vid_path="./outputs/test_res/simu/",
    imgsize=[720, 1280],
    gain=1,
    max_n_frame=20000,
):

    VID_FRAME_NUM = max_n_frame
    print("Generating data...")
    generater = MinibatchGenerater(Img_size=imgsize, dataset_opt=opt.Dataset, allow_fewer_frame_num=True)
    ch_a_noisy, ch_b_noisy = generater.generate_batch(
        Batch_size=1,
        dir_name=dir_name,
        frame_num=VID_FRAME_NUM,
        crop="none",
        dataset_type=dataset,
        norm=False,
        gain=None
    )
    
    if ch_a_noisy.shape[-2] > 800 or ch_a_noisy.shape[-1] > 800:
        print(f"Image size too large: {ch_a_noisy.shape}")
        ch_a_noisy = ch_a_noisy[:, :, :800, :800]
        ch_b_noisy = ch_b_noisy[:, :, :800, :800]

    print("Gain: ", gain)
    # ch_a_noisy = ch_a_noisy * gain
    # ch_b_noisy = ch_b_noisy * gain

    if os.path.exists(vid_path):
        shutil.rmtree(vid_path)

    os.makedirs(vid_path, exist_ok=True)
    os.makedirs(opj(vid_path, "3d"), exist_ok=True)
    os.makedirs(opj(vid_path, "mip"), exist_ok=True)

    tifffile.imwrite(opj(vid_path, "3d", "ch_a_noisy.tif"), make_tif(ch_a_noisy[0]))
    tifffile.imwrite(opj(vid_path, "3d", "ch_b_noisy.tif"), make_tif(ch_b_noisy[0]))
    tifffile.imwrite(opj(vid_path, "mip", "ch_a_noisy.tif"), make_mip(ch_a_noisy[0]))
    tifffile.imwrite(opj(vid_path, "mip", "ch_b_noisy.tif"), make_mip(ch_b_noisy[0]))

    origimgsize = ch_a_noisy.shape[-2:]
    if ch_a_noisy.shape[-2] < 64:
        print(f"Padding image from {ch_a_noisy.shape} to >64")
        rep_n = 64 // ch_a_noisy.shape[-2] + 1
        ch_a_noisy = np.tile(ch_a_noisy, [1, 1, 1, rep_n, 1])
        ch_b_noisy = np.tile(ch_b_noisy, [1, 1, 1, rep_n, 1])
    if ch_a_noisy.shape[-1] < 64:
        print(f"Padding image from {ch_a_noisy.shape} to >64")
        rep_n = 64 // ch_a_noisy.shape[-1] + 1
        ch_a_noisy = np.tile(ch_a_noisy, [1, 1, 1, 1, rep_n])
        ch_b_noisy = np.tile(ch_b_noisy, [1, 1, 1, 1, rep_n])

    DCLL = DualChannelLowLight(opt, model_path, device="cuda", imgsize=ch_a_noisy.shape[-2:])
    DCLL.load_low_light_video(ch_a_noisy, ch_b_noisy)
    ch_a_preds, ch_b_preds = DCLL.start_nn(savepath=vid_path, gain=gain)

    print(f"Cropping from {ch_a_preds.shape} to {origimgsize}")
    ch_a_preds = ch_a_preds[:, :, : origimgsize[0], : origimgsize[1]]
    ch_b_preds = ch_b_preds[:, :, : origimgsize[0], : origimgsize[1]]

    tifffile.imwrite(opj(vid_path, "3d", "ch_a_preds.tif"), make_tif(ch_a_preds))
    tifffile.imwrite(opj(vid_path, "3d", "ch_b_preds.tif"), make_tif(ch_b_preds))
    tifffile.imwrite(opj(vid_path, "mip", "ch_a_preds.tif"), make_mip(ch_a_preds))
    tifffile.imwrite(opj(vid_path, "mip", "ch_b_preds.tif"), make_mip(ch_b_preds))


def parse_gain(argsgain, dir_name, dataset_dict):
    # Gain is set as parameter
    if argsgain == "search_note":
        print("Gain is str, treat as path to look up gain.")
        note_path = dataset_dict["note"]
        with open(note_path, "r") as f:
            for line in f:
                f_name = line.split(" ")[0]
                if f_name in dir_name:
                    # print(line, line.split("%")[0])
                    gain = line.split("%")[0].split(" ")[-1]
                    gain = 100 / float(gain)
                    print(f'Line: "{line:s}", gain: {gain:.2f}')
                    break
    elif argsgain == "search_dir_name":
        gain = dir_name.split("_S")[0]
        if "ms" in gain:
            gain = float(gain.split("ms")[0])
            gain = 33 / gain
        elif "10-noim" in gain:
            gain = 10
        else:
            gain = 100 / float(gain)
        print(f"Dir name: {dir_name:s}, gain: {gain:.2f}")
    else:
        print("Parsing gain as numbers.")
        gain = float(argsgain)
    return gain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # clear previous output files/images
    # save figure to file every x epoch
    parser.add_argument(
        "-load_pkl",
        type=str,
        default="Exp/experiments_outputs/train/DCMAN_Concat_Fn15_0320_2023_0515_170206_076694/pklmodels/train_iter_100.pkl",
    )
    parser.add_argument("-opt", type=str, default="Exp/Data0511/DCMAN_Concat_Fn15_HJHS_ALL.yaml")
    parser.add_argument("-gpu", type=str, default="0")
    parser.add_argument("-im_resize", type=str, default="None")  # no resize: None   180,160
    parser.add_argument("-dataset", type=str, default="test_180x160")
    parser.add_argument("-gain", type=str, default="search_dir_name")
    parser.add_argument("-max_n_frame", type=float, default=int(30)) #
    parser.add_argument("-continue_idx", type=int, default=0)

    args = parser.parse_args()

    # gpu_list = [int(gpu) for gpu in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opt: TrainOpt = OmegaConf.load(args.opt)

    reproduc(opt.Random_seed)

    if args.im_resize == "None":
        imsize = [0, 0]
    else:
        imsize = [int(num) for num in args.im_resize.split(",")]
    opt.Img_size = imsize

    timetag = datetime.now().strftime("%Y_%m%d_%H%M%S_%f")
    run_name = (
        opb(os.path.abspath(opj(args.load_pkl, "../", "../")))
        + "_"
        + opb(args.load_pkl)
        + "_"
        + opb(args.opt)
        + f"_{args.dataset}"
    )
    if args.continue_idx > 0:
        run_name += f"_Cont{args.continue_idx:03d}"
    if not os.path.exists("./outputs/test_res/{:s}".format(run_name)):
        os.makedirs("./outputs/test_res/{:s}".format(run_name))
    initLogging(opj("./outputs/test_res/{:s}/{:s}_testlog.txt".format(run_name, run_name)))

    test_dir = "./outputs/test_res/{:s}".format(run_name)
    # if os.path.exists(test_dir):
    #     shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    ch_a_pred_psnr_list = []
    ch_a_noisy_psnr_list = []
    ch_a_pred_ssim_list = []
    ch_a_noisy_ssim_list = []
    ch_b_pred_psnr_list = []
    ch_b_noisy_psnr_list = []
    ch_b_pred_ssim_list = []
    ch_b_noisy_ssim_list = []

    print(OmegaConf.to_container(opt.Dataset))
    datasets = OmegaConf.to_container(opt.Dataset)[args.dataset]
    for dataset_i, dataset_dict in enumerate(datasets):
        dataset_path = dataset_dict["path"]
        print(f"Dataset {dataset_i:d}/{len(datasets):d} path: {dataset_path:s}")

        dir_list = sorted(os.listdir(dataset_path))
        # print(dir_list)
        # dir_list.remove("note.txt")
        dir_list = dir_list[args.continue_idx :]
        for i in range(len(dir_list)):
            print(f"Processing {i:d}th / {len(dir_list):d} video: {opj(dataset_path, dir_list[i]):s}")
            dir_name = dir_list[i]
            if not os.path.isdir(opj(dataset_path, dir_name)):
                print("Not a directory, skip.")
                continue

            ### 测试真实数据，需要提供gain，并且只有弱光两个通道
            if "real" in args.dataset:

                gain = parse_gain(args.gain, dir_name, dataset_dict)

                save_name = "_".join(dataset_path.split("/")[-1:]) + "_" + dir_name
                test_real_video(
                    opt,
                    args.load_pkl,
                    imgsize=imsize,
                    dir_name=opj(dataset_path, dir_name),
                    dataset=args.dataset,
                    vid_path=opj(test_dir, save_name),
                    gain=gain,
                    max_n_frame=args.max_n_frame,
                )

            # 测试模拟数据，不需要提供gain，有弱光和强光四个通道
            else:
                # Gain is calculated according to label
                metrics = test_simu_video(
                    opt,
                    args.load_pkl,
                    imgsize=imsize,
                    dir_name=opj(dataset_path, dir_name),
                    index=i,
                    dataset=args.dataset,
                    vid_path=opj(test_dir, f"{dir_name:s}"),
                    max_n_frame=args.max_n_frame,
                )

                (
                    ch_a_pred_psnr,
                    ch_a_noisy_psnr,
                    ch_a_pred_ssim,
                    ch_a_noisy_ssim,
                    ch_b_pred_psnr,
                    ch_b_noisy_psnr,
                    ch_b_pred_ssim,
                    ch_b_noisy_ssim,
                ) = metrics

                ch_a_pred_psnr_list.append(ch_a_pred_psnr)
                ch_a_noisy_psnr_list.append(ch_a_noisy_psnr)
                ch_a_pred_ssim_list.append(ch_a_pred_ssim)
                ch_a_noisy_ssim_list.append(ch_a_noisy_ssim)
                ch_b_pred_psnr_list.append(ch_b_pred_psnr)
                ch_b_noisy_psnr_list.append(ch_b_noisy_psnr)
                ch_b_pred_ssim_list.append(ch_b_pred_ssim)
                ch_b_noisy_ssim_list.append(ch_b_noisy_ssim)

        L.info("Mean pred. ch_a_ PSNR: {:.3f}".format(np.mean(ch_a_pred_psnr_list)))
        L.info("Mean noisy ch_a_ PSNR: {:.3f}".format(np.mean(ch_a_noisy_psnr_list)))
        L.info("Mean pred. ch_a_ SSIM: {:.3f}".format(np.mean(ch_a_pred_ssim_list)))
        L.info("Mean noisy ch_a_ SSIM: {:.3f}".format(np.mean(ch_a_noisy_ssim_list)))
        L.info("Mean pred. ch_b_ PSNR: {:.3f}".format(np.mean(ch_b_pred_psnr_list)))
        L.info("Mean noisy ch_b_ PSNR: {:.3f}".format(np.mean(ch_b_noisy_psnr_list)))
        L.info("Mean pred. ch_b_ SSIM: {:.3f}".format(np.mean(ch_b_pred_ssim_list)))
        L.info("Mean noisy ch_b_ SSIM: {:.3f}".format(np.mean(ch_b_noisy_ssim_list)))

        L.info("Mean delta ch_a_ PSNR: {:.3f}".format(np.mean(ch_a_pred_psnr_list) - np.mean(ch_a_noisy_psnr_list)))
        L.info("Mean delta ch_a_ SSIM: {:.3f}".format(np.mean(ch_a_pred_ssim_list) - np.mean(ch_a_noisy_ssim_list)))
        L.info("Mean delta ch_a_ PSNR: {:.3f}".format(np.mean(ch_b_pred_psnr_list) - np.mean(ch_b_noisy_psnr_list)))
        L.info("Mean delta ch_a_ SSIM: {:.3f}".format(np.mean(ch_b_pred_ssim_list) - np.mean(ch_b_noisy_ssim_list)))
