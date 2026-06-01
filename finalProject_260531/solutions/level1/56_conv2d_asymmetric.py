"""
KernelBench Level 1 / Problem 56 — Conv2D with asymmetric input/kernel.
cuDNN-dominated; falls back.
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 groups=1, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv2d(x)
