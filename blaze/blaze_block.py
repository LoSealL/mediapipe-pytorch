"""
Copyright: 2021 Intel Corporation
Author: Wenyi Tang
Email: wenyi.tang@intel.com

Common blaze block
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConvTF(nn.Conv2d):
    """Tensorflow convolution has a different padding than PyTorch.
    When using padding='SAME' in tf, it will pad right bottom first, while
    pt pads left top first.
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='same',
                 dilation=1,
                 groups=1,
                 bias=True):
        super(ConvTF, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        if padding.lower() not in ('valid', 'same'):
            raise ValueError("padding must be 'same' or 'valid'")
        self.pad = padding

    def compute_valid_shape(self, in_shape):
        in_shape = np.asarray(in_shape).astype('int32')
        stride = np.asarray(self.stride).astype('int32')
        kernel_size = np.asarray(self.kernel_size).astype('int32')
        stride = np.concatenate([[1, 1], stride])
        kernel_size = np.concatenate([[1, 1], kernel_size])
        dilation = np.asarray(self.dilation).astype('int32')
        dilation = np.concatenate([[1, 1], dilation])
        if self.pad == 'same':
            out_shape = (in_shape + stride - 1) // stride
        else:
            out_shape = (in_shape - dilation * (kernel_size - 1) - 1) // stride + 1
        valid_input_shape = (out_shape - 1) * stride + 1 + dilation * (kernel_size - 1)
        return valid_input_shape

    def forward(self, input):
        in_shape = np.asarray(input.shape).astype('int32')
        valid_shape = self.compute_valid_shape(in_shape)
        pad = []
        for x in valid_shape - in_shape:
            if x == 0:
                continue
            pad_left = x // 2
            pad_right = x - pad_left
            # pad right should be larger than pad left
            pad.extend((pad_left, pad_right))
        if np.not_equal(pad, 0).any():
            padded_input = F.pad(input, pad)
        else:
            padded_input = input
        return super(ConvTF, self).forward(padded_input)


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride > 1:
            self.max_pool = nn.MaxPool2d(stride)

        self.convs = nn.Sequential(
            ConvTF(in_channels=in_channels, out_channels=in_channels,
                   kernel_size=kernel_size, stride=stride, padding='same',
                   groups=in_channels, bias=True),
            ConvTF(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=1, stride=1, padding='same', bias=True),
        )

    def forward(self, x):
        if self.stride > 1:
            h = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            h = F.pad(h, (0, 0, 0, 0, 0, self.channel_pad))

        return self.convs(x) + h


class DualBlazeBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1):
        super(DualBlazeBlock, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride > 1:
            self.max_pool = nn.MaxPool2d(stride)

        self.convs = nn.Sequential(
            ConvTF(in_channels=in_channels, out_channels=in_channels,
                   kernel_size=kernel_size, stride=stride, padding='same',
                   groups=in_channels, bias=True),
            ConvTF(in_channels=in_channels, out_channels=mid_channels,
                   kernel_size=1, stride=1, padding='same', bias=True),
            nn.ReLU(True),
            ConvTF(in_channels=mid_channels, out_channels=mid_channels,
                   kernel_size=kernel_size, stride=1, padding='same',
                   groups=mid_channels, bias=True),
            ConvTF(in_channels=mid_channels, out_channels=out_channels,
                   kernel_size=1, stride=1, padding='same', bias=True),
        )

    def forward(self, x):
        if self.stride > 1:
            h = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            h = F.pad(h, (0, 0, 0, 0, 0, self.channel_pad))

        return self.convs(x) + h
