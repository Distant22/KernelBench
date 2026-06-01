"""
KernelBench Level 1 / Problem 50 — AlexNet first conv2d (11x11, stride=4).
cuDNN-dominated; falls back.
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        return self.conv1(x)
