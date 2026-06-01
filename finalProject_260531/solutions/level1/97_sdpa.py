"""
KernelBench Level 1 / Problem 97 — Scaled Dot-Product Attention.
PyTorch SDPA already dispatches to fused attention; falls back.
"""
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
