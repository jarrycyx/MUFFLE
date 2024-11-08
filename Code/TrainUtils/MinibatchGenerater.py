"""
Generate noisy-clean ch_a-ch_b image Minibatches with Clean ch_a and ch_b Image. 
Can be used for training, validating or testing. Note that during our training process 
we first save the simulated noisy imgs to files and THEN load then while training.

Related files: 
    SimuDatasetGenerater.py: Save simulated noisy-clean ch_a-ch_b dataset to file.
    SimuDatasetLoader.py: Load the saved, simulated noisy-clean ch_a-ch_b dataset for training.

@author: Cheng Yuxiao (chengyx18@mails.tsinghua.edu.cn)
"""

import sys, os

from omegaconf import OmegaConf


import time
from multiprocessing import Pool
from ImgProcessing.ImgNoiseMdl import CMOS_Model
import ImgProcessing.ImgUtils as ImgUtils
import json
import cv2
import numpy as np
import tifffile


class MinibatchGenerater(object):
    """
    usage:
        generater = MinibatchGenerater(Img_size=(400, 800))
        if allow_fewer_frame_num is true, generated batches may have different frame num
            depending on the file numbers in the selected directory
        ch_a_noisy, ch_b_noisy, ch_a_label, ch_b_label = generater.generate_batch(1, 7)

    """

    def __init__(self, dataset_opt, Img_size=(100, 200), allow_fewer_frame_num=False, down_sample=2):
        self.allow_fewer_frame_num = allow_fewer_frame_num

        self.Img_size = Img_size

        self.Dataset_opt = OmegaConf.to_container(dataset_opt)
        self.down_sample = down_sample

    def get_random_dataset_path(self, dataset_type="train"):
        """Multiple datasets can be used for training
        a dataset path is selected randomly in this method

        Args:
            dataset_type: string, "train", "val" or "test"
        """
        paths = []
        for path in self.Dataset_opt[dataset_type]:
            for i in range(path["weight"]):
                paths.append(path)

        random_index = int(np.random.random() * len(paths))
        return paths[random_index]

    def get_filelist(self, frame_num, file_index=None, dir_name=None, dataset_type="train"):
        """Get a list of clean image files

        Args:
            frame_num: number of frames to generate
            dataset_type: string, "train", "val" or "test"
        """
        frame_num = int(frame_num)
        if dataset_type == "train":
            # Randomly select data when mode is "train"
            datapath_dict = self.get_random_dataset_path(dataset_type)
            thislayer_path = datapath_dict["path"]
            for i in range(datapath_dict["layer"]):
                thislayer_list = sorted(os.listdir(thislayer_path))
                thislayer_list = [n for n in thislayer_list if os.path.isdir(os.path.join(thislayer_path, n))]
                thislayer_path = os.path.join(
                    thislayer_path, thislayer_list[int(np.random.random() * len(thislayer_list))]
                )

            dir_path = thislayer_path

            ch_a_data_frames = [
                os.path.join(dir_path, "chah", img_path)
                for img_path in sorted(os.listdir(os.path.join(dir_path, "chah")))
                if img_path[0] != "."
            ]
            ch_b_data_frames = [
                os.path.join(dir_path, "chbh", img_path)
                for img_path in sorted(os.listdir(os.path.join(dir_path, "chah")))
                if img_path[0] != "."
            ]
            total_frame_amount = len(ch_a_data_frames)
            start_frame = max(0, int(np.random.random() * (total_frame_amount - frame_num - 1)))
            end_frame = int(start_frame + frame_num)

            return (
                ch_a_data_frames[start_frame:end_frame],
                ch_b_data_frames[start_frame:end_frame],
                datapath_dict["type"],
            )
        else:
            # Consequently select data when mode is not "train"
            # datapath_dict = self.get_random_dataset_path(dataset_type)
            dir_list = []
            for datapath_dict in self.Dataset_opt[dataset_type]:
                this_dataset_dir_list = sorted(os.listdir(datapath_dict["path"]))
                dir_list += [n for n in this_dataset_dir_list if os.path.join(datapath_dict["path"], n)]
            # print(f"Num of dirs: {len(dir_list)}")
                
            dir_list = [n for n in dir_list if os.path.isdir(os.path.join(datapath_dict["path"], n))]
            if file_index is not None:
                dir_path = os.path.join(datapath_dict["path"], dir_list[file_index % len(dir_list)])
            elif dir_name is not None:
                dir_path = dir_name
            else:
                raise ValueError("file_index and dir_name cannot be both None.")
            print(f"Generating {frame_num} frames from {dir_path}.")

            ch_a_data_frames = [
                os.path.join(dir_path, "chal", img_path)
                for img_path in sorted(os.listdir(os.path.join(dir_path, "chal")))
                if img_path[0] != "."
            ]
            ch_b_data_frames = [
                os.path.join(dir_path, "chbl", img_path)
                for img_path in sorted(os.listdir(os.path.join(dir_path, "chal")))
                if img_path[0] != "."
            ]
        
        return ch_a_data_frames[:frame_num], ch_b_data_frames[:frame_num], datapath_dict["type"]

    def generate_vid(
        self, frame_num, file_index=None, dir_name=None, dataset_type="train", printPath=False, crop="random"
    ):
        """Get a list of simulated noisy images

        Args:
            frame_num: number of frames to generate
            dataset_type: string, "train", "val" or "test"
            print_path: wether to pring img file path or not
            randomCrop: True - crop randomly to target size
                        False - resize to target size

        Returns:
            4-D numpy array of shape (frame_num, channel_num, height, width)
            image scale 0.0-1.0
        """

        ch_a_frames_path = []
        ch_a_frames_path, ch_b_frames_path, filetype = self.get_filelist(
            frame_num, file_index=file_index, dir_name=dir_name, dataset_type=dataset_type
        )
        # TO-DO: waiting for real ch_b images

        actual_shape = ImgUtils.read_tiff_image(ch_a_frames_path[0], type=filetype).shape
        if self.Img_size[0] == 0 or self.Img_size[1]==0:
            self.Img_size = actual_shape

        if len(ch_a_frames_path) < frame_num:
            print("Not enough video frames", ch_a_frames_path[0])
            if self.allow_fewer_frame_num:
                frame_num = len(ch_a_frames_path)
                print("Generating {:d} frames...".format(frame_num))
            else:
                zero_arr = np.zeros(
                    [frame_num, actual_shape[2] if len(actual_shape) == 3 else 1, self.Img_size[0], self.Img_size[1]]
                )
                print("Generating all zero frames")
                return zero_arr, zero_arr, zero_arr, zero_arr

        if printPath:
            print(ch_a_frames_path[0])

        crop_size = (self.Img_size[0] * self.down_sample, self.Img_size[1] * self.down_sample)

        if crop == "none":
            crop_area = None
            if actual_shape[0] == self.Img_size[0] and actual_shape[1] == self.Img_size[1]:
                resize = None
            else:
                resize = self.Img_size
        else:
            if crop == "random" and actual_shape[0] > crop_size[0] and actual_shape[1] > crop_size[1]:
                # random patch coordinates
                ys = int(np.random.random() * (actual_shape[0] - crop_size[0]))
                ye = int(ys + crop_size[0])
                xs = int(np.random.random() * (actual_shape[1] - crop_size[1]))
                xe = int(xs + crop_size[1])
            else:
                ys = 0
                xs = 0
                ye = int(ys + crop_size[0])
                xe = int(xs + crop_size[1])

            crop_area = [ys, ye, xs, xe]
            resize = self.Img_size

        if dataset_type == "real":
            ch_a_noisy_img_list = np.array(
                [
                    ImgUtils.read_tiff_image(frame_path, type=filetype, crop=crop_area, size=resize, color="gray")
                    for frame_path in ch_a_frames_path
                ]
            )
            ch_b_noisy_img_list = np.array(
                [
                    ImgUtils.read_tiff_image(frame_path, type=filetype, crop=crop_area, size=resize, color="gray")
                    for frame_path in ch_b_frames_path
                ]
            )

            ch_a_noisy_list = []
            ch_b_noisy_list = []
            for ch_a_noisy_img, ch_b_noisy_img in zip(ch_a_noisy_img_list, ch_b_noisy_img_list):
                max_range = {"uint8": 255, "uint16": 65535, "float": 1}[str(ch_a_noisy_img.dtype)]
                ch_a_noisy_list.append(ch_a_noisy_img[None, :, :] / max_range)
                ch_b_noisy_list.append(ch_b_noisy_img[None, :, :] / max_range)

            return np.array(ch_a_noisy_list), np.array(ch_b_noisy_list)

        else:
            ch_a_label_img_list = np.array(
                [
                    ImgUtils.read_tiff_image(frame_path.replace("chal", "chah"), type=filetype, crop=crop_area, size=resize, color="gray")
                    for frame_path in ch_a_frames_path
                ]
            )
            ch_b_label_img_list = np.array(
                [
                    ImgUtils.read_tiff_image(frame_path.replace("chbl", "chbh"), type=filetype, crop=crop_area, size=resize, color="gray")
                    for frame_path in ch_b_frames_path
                ]
            )

            ch_a_noisy_img_list = np.array(
                [
                    ImgUtils.read_tiff_image(
                        frame_path.replace("chah", "chal"), type=filetype, crop=crop_area, size=resize, color="gray"
                    )
                    for frame_path in ch_a_frames_path
                ]
            )
            ch_b_noisy_img_list = np.array(
                [
                    ImgUtils.read_tiff_image(
                        frame_path.replace("chbh", "chbl"), type=filetype, crop=crop_area, size=resize, color="gray"
                    )
                    for frame_path in ch_b_frames_path
                ]
            )

            ch_a_noisy_list = []
            ch_b_noisy_list = []
            ch_a_label_list = []
            ch_b_label_list = []
            for ch_a_label_img, ch_b_label_img, ch_a_noisy_img, ch_b_noisy_img in zip(
                ch_a_label_img_list, ch_b_label_img_list, ch_a_noisy_img_list, ch_b_noisy_img_list
            ):
                max_range = {"uint8": 255, "uint16": 65535, "float": 1}[str(ch_a_label_img.dtype)]
                ch_a_label_list.append(ch_a_label_img[None, :, :] / max_range)
                ch_b_label_list.append(ch_b_label_img[None, :, :] / max_range)
                ch_a_noisy_list.append(ch_a_noisy_img[None, :, :] / max_range)
                ch_b_noisy_list.append(ch_b_noisy_img[None, :, :] / max_range)

            return (
                np.array(ch_a_noisy_list),
                np.array(ch_b_noisy_list),
                np.array(ch_a_label_list),
                np.array(ch_b_label_list),
            )

    def generate_batch(
        self,
        Batch_size,
        frame_num,
        worker=10,
        norm=True,
        dir_name=None,
        file_index=None,
        crop="random",
        dataset_type="train",
        gain="random",
    ):
        """Get a mini-batch of simulated noisy images for training

        Args:
            Batch_size
            frame_num: number of frames to generate
            dataset_type: string, "train", "val" or "test"
            print_path: wether to pring img file path or not
            randomCrop: True - crop randomly to target size
                        False - resize to target size

        Returns:
            5-D numpy array of shape (Batch_size, frame_num, channel_num, height, width)
            image scale 0.0-1.0
        """

        if dataset_type == "real":
            epoch_ch_a_noisy = []
            epoch_ch_b_noisy = []
            for i in range(Batch_size):
                ch_a_noisy, ch_b_noisy = self.generate_vid(
                    frame_num, file_index=file_index, dir_name=dir_name, crop=crop, dataset_type=dataset_type
                )
                epoch_ch_a_noisy.append(ch_a_noisy)
                epoch_ch_b_noisy.append(ch_b_noisy)

            epoch_ch_a_noisy = np.array(epoch_ch_a_noisy)
            epoch_ch_b_noisy = np.array(epoch_ch_b_noisy)

            gain = 1

            if norm:
                a_gain = np.mean(epoch_ch_b_label) / np.mean(epoch_ch_a_label)
            else:
                a_gain = 1

            return (epoch_ch_a_noisy * a_gain * gain, epoch_ch_b_noisy * gain)
        else:
            epoch_ch_a_noisy = []
            epoch_ch_b_noisy = []
            epoch_ch_a_label = []
            epoch_ch_b_label = []
            for i in range(Batch_size):
                ch_a_noisy, ch_b_noisy, ch_a_label, ch_b_label = self.generate_vid(
                    frame_num, file_index=file_index, crop=crop, dataset_type=dataset_type, dir_name=dir_name
                )
                epoch_ch_a_noisy.append(ch_a_noisy)
                epoch_ch_b_noisy.append(ch_b_noisy)
                epoch_ch_a_label.append(ch_a_label)
                epoch_ch_b_label.append(ch_b_label)

            epoch_ch_a_noisy = np.array(epoch_ch_a_noisy)
            epoch_ch_b_noisy = np.array(epoch_ch_b_noisy)
            epoch_ch_a_label = np.array(epoch_ch_a_label)
            epoch_ch_b_label = np.array(epoch_ch_b_label)

            # a_gain is the gain to make the mean of ch_a_label equal to the mean of ch_a_noisy
            # noisy_gain is the gain to make the mean of ch_b_label equal to the mean of ch_b_noisy
            if norm:
                a_gain = np.mean(epoch_ch_b_label) / np.mean(epoch_ch_a_label)
                noisy_gain = np.mean(epoch_ch_b_label) / np.mean(epoch_ch_b_noisy)
            else:
                a_gain = 1
                noisy_gain = 1

            # Final gain
            if gain == "random":
                gain = np.random.uniform(
                    1, np.max([1, 0.7 / np.percentile([epoch_ch_a_label, epoch_ch_b_label], 99)]), size=1
                )
            elif gain == "norm":
                gain = 0.5 / np.percentile([epoch_ch_a_label, epoch_ch_b_label], 99)
            else:
                gain = 1

            return (
                epoch_ch_a_noisy * a_gain * noisy_gain * gain,
                epoch_ch_b_noisy * noisy_gain * gain,
                epoch_ch_a_label * a_gain * gain,
                epoch_ch_b_label * gain,
            )

    def generate_test(self, frame_num, file_index=None, gain="none"):
        """Get a mini-batch of simulated noisy images for testing
        image scale 0.0-1.0
        """
        return self.generate_batch(
            Batch_size=1, frame_num=frame_num, file_index=file_index, crop="none", dataset_type="test", gain=gain
        )


if __name__ == "__main__":
    generater = MinibatchGenerater(
        OmegaConf.load("opt\DCMAN_Concat.yaml").Dataset, Img_size=(200, 200), ch_a_kc=24, ch_b_kc=24
    )
    ch_a_noisy, ch_b_noisy, ch_a_label, ch_b_label = generater.generate_batch(1, 7)
    print(ch_a_noisy.dtype, ch_b_label.shape)
    print(ch_a_noisy.max(), ch_b_label.max())

    ImgUtils.show_bright_images(
        [[ch_a_noisy[0][1], ch_b_noisy[0][1]], [ch_a_label[0][1], ch_b_label[0][1]]], channel_first=True, norm=False
    )
