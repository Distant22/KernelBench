"""
HAND-WRITTEN attempt — KernelBench Level 1 / Problem 56: asymmetric Conv2D.

Honest measurement of a from-scratch Triton implicit-GEMM conv2d vs cuDNN
(fallback scored ~0.995x).

Shape: B=8, Cin=64, Cout=128, K=(5,7), H=512, W=256, stride=1, pad=0, no bias.
Out H=508, W=250.

Formulation: loop over (kh,kw); each is a 1x1-style GEMM over Cin:
    y[:, oh, ow] += W[:, :, kh, kw] @ x[:, oh+kh, ow+kw]
Program owns (batch, Cout-tile, one output row oh, width-tile of ow).
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # output channels
    offs_ow = wt * BLOCK_N + tl.arange(0, BLOCK_N)        # output width positions
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
                # W tile (BLOCK_M, BLOCK_K)
                w_off = (offs_m[:, None] * (Cin * KH * KW)
                         + cin[None, :] * (KH * KW) + kh * KW + kw)
                w_tile = tl.load(w_ptr + w_off,
                                 mask=mask_m[:, None] & mask_c[None, :], other=0.0)
                # x tile (BLOCK_K, BLOCK_N)
                x_off = cin[:, None] * (H * W) + ih * W + iw[None, :]
                x_tile = tl.load(x_b + x_off,
                                 mask=(mask_c[:, None] & (iw_ok[None, :] & ih_ok)),
                                 other=0.0)
                acc += tl.dot(w_tile, x_tile, allow_tf32=False)

    y_off = (pid_b * Cout * Hout * Wout + offs_m[:, None] * (Hout * Wout)
             + oh * Wout + offs_ow[None, :])
    tl.store(y_ptr + y_off, acc, mask=mask_m[:, None] & mask_ow[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 groups=1, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self.KH, self.KW = kernel_size
        self.SH, self.SW = (stride if isinstance(stride, tuple) else (stride, stride))
        self.PH, self.PW = (padding if isinstance(padding, tuple) else (padding, padding))

    def forward(self, x):
        B, Cin, H, W = x.shape
        Cout = self.conv2d.out_channels
        KH, KW = self.KH, self.KW
        Hout = (H + 2 * self.PH - KH) // self.SH + 1
        Wout = (W + 2 * self.PW - KW) // self.SW + 1

        x = x.contiguous()
        w = self.conv2d.weight.contiguous()
        y = torch.empty((B, Cout, Hout, Wout), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64 if Cin >= 64 else triton.next_power_of_2(Cin)
        grid = (B, triton.cdiv(Cout, BLOCK_M), Hout * triton.cdiv(Wout, BLOCK_N))
        _conv2d_kernel[grid](
            x, w, y,
            B, Cin, H, W, Cout, Hout, Wout,
            KH=KH, KW=KW, SH=self.SH, SW=self.SW, PH=self.PH, PW=self.PW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return y
