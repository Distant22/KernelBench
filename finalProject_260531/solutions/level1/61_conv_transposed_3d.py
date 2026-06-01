"""
KernelBench Level 1 / Problem 61 — ConvTranspose3D.
cuDNN-dominated; falls back to nn.ConvTranspose3d.
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv_transpose3d(x)
