
import sys
import os
from os.path import join as opj
from os.path import dirname as opd
sys.path.append(opj(opd(__file__), ".."))

from Net.DualDecoder.DecoderAdd import Channel_Merge_Decoder_Add
from Net.DualDecoder.DecoderAdd import Channel_Merge_Decoder_None
from Net.DualDecoder.DecoderFusion import Channel_Merge_Decoder_Fusion
from Net.DualDecoder.DecoderConcat import Channel_Merge_Decoder_Concat
from Net.DualDecoder.DecoderConvLSTM import Channel_Merge_Decoder_LSTM

from Net.Utils import *
import numpy as np
from Net.LSTM.BiConvLSTM import BiConvLSTM
from torch.nn.init import kaiming_normal_
import torch.nn as nn
import torch
import sys


class Channel_Encoder(nn.Module):

    def __init__(self, batchNorm=False, input_size=[240, 416], frame_num=7,
                 mode='train', lstm_layer=1):
        super(Channel_Encoder, self).__init__()
        self.batchNorm = batchNorm
        self.mode = mode
        self.lstm_layer = lstm_layer

        self.frame_num = frame_num
        pre_convs = []
        for i in range(self.frame_num):
            preconv = nn.ModuleList([conv(self.batchNorm, 1, 64,  kernel_size=3,  stride=1),
                                     conv(self.batchNorm, 64, 64, kernel_size=3,  stride=1)])
            pre_convs.append(preconv)

        self.pre_convs = nn.ModuleList(pre_convs)

        self.biconvlstm = BiConvLSTM(input_size=(
            input_size[0], input_size[1]), input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=self.lstm_layer)

        self.LSTM_out = conv(self.batchNorm, 128, 64,
                             kernel_size=1,  stride=1)

        self.conv1 = conv(self.batchNorm,   64, 128,
                          kernel_size=7,  stride=2)  # 64
        self.conv1_1 = conv(self.batchNorm,   128, 128)  # 128*128 ->64*64
        self.conv2 = conv(self.batchNorm,   128, 256,
                          kernel_size=3,  stride=2)  # 64 ->32
        self.conv2_1 = conv(self.batchNorm,   256, 256)  # 128*128 ->64*64
        self.conv3 = conv(self.batchNorm,   256, 512,
                          kernel_size=3,  stride=2)  # 32->16
        self.conv3_1 = conv(self.batchNorm,   512, 512)
        self.conv4 = conv(self.batchNorm,   512, 1024,
                          kernel_size=3,  stride=2)  # 16->8
        self.conv4_1 = conv(self.batchNorm,   1024, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data, a=0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, frames):

        CNN_seq = []
        half_num = (self.frame_num-1)//2

        for i in range(self.frame_num):
            pre_conv_this_frame = self.pre_convs[i][0](frames[:, i])
            pre_conv_1_this_frame = self.pre_convs[i][1](pre_conv_this_frame)
            CNN_seq.append(pre_conv_1_this_frame)

        CNN_seq_out = torch.stack(CNN_seq, dim=1)
        CNN_seq_feature_maps = self.biconvlstm(CNN_seq_out)

        CNN_concat_input = torch.cat(
            [CNN_seq_out[:, half_num, ...], CNN_seq_feature_maps[:, half_num, ...]], dim=1)

        LSTM_out = self.LSTM_out(CNN_concat_input)  # 128*128*64
        
        # LSTM_out = CNN_seq_out

        out_conv1 = self.conv1_1(self.conv1(LSTM_out))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        return [LSTM_out, out_conv1, out_conv2, out_conv3, out_conv4]


class Dual_Channel_Attention_Net(nn.Module):

    def __init__(self, batchNorm=False, input_size=[240, 416], mode="train",
                 frame_num=7, lstm_layer=1, Channel_merge="concat"):
        super(Dual_Channel_Attention_Net, self).__init__()
        self.batchNorm = batchNorm
        self.mode = mode
        self.frame_num = frame_num
        self.lstm_layer = lstm_layer

        self.channel_a_encoder = Channel_Encoder(batchNorm=batchNorm, input_size=input_size,
                                                 frame_num=frame_num, lstm_layer=self.lstm_layer)
        self.channel_b_encoder = Channel_Encoder(batchNorm=batchNorm, input_size=input_size,
                                                 frame_num=frame_num, lstm_layer=self.lstm_layer)
        
        if Channel_merge == "add":
            Dual_Decoder = Channel_Merge_Decoder_Add
        if Channel_merge == "none":
            Dual_Decoder = Channel_Merge_Decoder_None
        elif Channel_merge == "concat":
            Dual_Decoder = Channel_Merge_Decoder_Concat
        elif Channel_merge == "fusion":
            Dual_Decoder = Channel_Merge_Decoder_Fusion
        elif Channel_merge == "lstm":
            Dual_Decoder = Channel_Merge_Decoder_LSTM
        else:
            raise NotImplementedError

        self.decoder_a = Dual_Decoder(batchNorm=batchNorm, input_size=input_size,
                                                      frame_num=frame_num, mode=mode)
        self.decoder_b = Dual_Decoder(batchNorm=batchNorm, input_size=input_size,
                                                      frame_num=frame_num, mode=mode)

    def forward(self, channel_a, channel_b):
        codes_a = self.channel_a_encoder(channel_a)
        codes_b = self.channel_b_encoder(channel_b)

        if self.mode == "debug":
            print("l0 size: ", codes_a[0].shape)
            print("l1 size: ", codes_a[1].shape)
            print("l2 size: ", codes_a[2].shape)
            print("l3 size: ", codes_a[3].shape)
            print("l4 size: ", codes_a[4].shape)

        output_a = self.decoder_a(codes_a, codes_b)
        output_b = self.decoder_b(codes_b, codes_a)
        return output_a, output_b


if __name__ == "__main__":
    from torchinfo import summary
    test_a = torch.zeros([1, 7, 1, 120, 240])
    test_b = torch.zeros([1, 7, 1, 120, 240])

    net = Dual_Channel_Attention_Net(input_size=[120, 240], mode='debug', Channel_merge="lstm").cuda()
    # net(test_a, test_b)
    summary(net, [(1, 7, 1, 120, 240), (1, 7, 1, 120, 240)])
    
    
    net = Dual_Channel_Attention_Net(input_size=[120, 240], mode='debug', Channel_merge="concat").cuda()
    # net(test_a, test_b)
    summary(net, [(1, 7, 1, 120, 240), (1, 7, 1, 120, 240)])
    
    
    net = Dual_Channel_Attention_Net(input_size=[120, 240], mode='debug', Channel_merge="none").cuda()
    # net(test_a, test_b)
    summary(net, [(1, 7, 1, 120, 240), (1, 7, 1, 120, 240)])
    
    # # 统计参数个数
    # num_params = 0
    # for param in net.parameters():
    #     num_params += param.numel()
    # print(f"Number of parameters: {num_params/1e6:.2f}M")