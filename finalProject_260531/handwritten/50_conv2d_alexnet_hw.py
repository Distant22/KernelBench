"""
HAND-WRITTEN attempt — KernelBench Level 1 / Problem 50: AlexNet first conv2d.

Honest measurement vs cuDNN (fallback scored ~0.999x).

Shape: B=256, Cin=3, Cout=96, K=11, stride=4, pad=2, input 224x224. Out 55x55.
Cin=3 is tiny -> implicit-GEMM K dim per (kh,kw) is only 3; padded to BLOCK_K=16.
Same implicit-GEMM formulation as 56_conv2d_asymmetric_hw.py.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, Cin, H, W, Cout, Hout, Wout,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_hw = tl.program_id(2)

    n_wtile = tl.cdiv(Wout, BLOCK_N)
    oh = pid_hw // n_wtile
    wt = pid_hw % n_wtile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ow = wt * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < Cout
    mask_ow = offs_ow < Wout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_b = x_ptr + pid_b * Cin * H * W
    for kh in range(0, KH):
        ih = oh * SH - PH + kh
        ih_ok = (ih >= 0) & (ih < H)
        for kw in range(0, KW):
            iw = offs_ow * SW - PW + kw
            iw_ok = (iw >= 0) & (iw < W) & mask_ow
            for ci0 in range(0, Cin, BLOCK_K):
                cin = ci0 + tl.arange(0, BLOCK_K)
                mask_c = cin < Cin
                w_off = (offs_m[:, None] * (Cin * KH * KW)
                         + cin[None, :] * (KH * KW) + kh * KW + kw)
                w_tile = tl.load(w_ptr + w_off,
                                 mask=mask_m[:, None] & mask_c[None, :], other=0.0)
                x_off = cin[:, None] * (H * W) + ih * W + iw[None, :]
                x_tile = tl.load(x_b + x_off,
                                 mask=(mask_c[:, None] & (iw_ok[None, :] & ih_ok)),
                                 other=0.0)
                acc += tl.dot(w_tile, x_tile, allow_tf32=False)

    y_off = (pid_b * Cout * Hout * Wout + offs_m[:, None] * (Hout * Wout)
             + oh * Wout + offs_ow[None, :])
    tl.store(y_ptr + y_off, acc, mask=mask_m[:, None] & mask_ow[None, :])


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        B, Cin, H, W = x.shape
        conv = self.conv1
        Cout = conv.out_channels
        KH = KW = 11
        SH = SW = 4
        PH = PW = 2
        Hout = (H + 2 * PH - KH) // SH + 1
        Wout = (W + 2 * PW - KW) // SW + 1

        x = x.contiguous()
        w = conv.weight.contiguous()
        y = torch.empty((B, Cout, Hout, Wout), device=x.device, dtype=x.dtype)

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 16
        grid = (B, triton.cdiv(Cout, BLOCK_M), Hout * triton.cdiv(Wout, BLOCK_N))
        _conv2d_kernel[grid](
            x, w, y,
            B, Cin, H, W, Cout, Hout, Wout,
            KH=KH, KW=KW, SH=SH, SW=SW, PH=PH, PW=PW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        if conv.bias is not None:
            y += conv.bias.view(1, Cout, 1, 1)
        return y
