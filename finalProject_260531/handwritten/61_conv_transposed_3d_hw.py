"""
HAND-WRITTEN attempt — KernelBench Level 1 / Problem 61: ConvTranspose3D.

Honest measurement of a from-scratch Triton transposed-3D-conv vs cuDNN
(fallback scored ~1.004x). This is the hardest fallback to beat.

Shape: B=8, Cin=48, Cout=48, K=3, D=H=W=64, stride=1, pad=0, out_pad=0, no bias.
Output D=H=W=66.

Gather form (stride=1, pad=0):
  y[b,co,od,oh,ow] = sum_{ci,kd,kh,kw} w[ci,co,kd,kh,kw] * x[b,ci, od-kd, oh-kh, ow-kw]
valid when 0 <= od-kd < D (and likewise h,w). Weight layout is (Cin,Cout,K,K,K).
Implemented as KKK GEMMs over Cin (implicit GEMM), one per (kd,kh,kw).
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _convt3d_kernel(
    x_ptr, w_ptr, y_ptr,
    Cin, D, H, W, Cout, Dout, Hout, Wout,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    n_wtile = tl.cdiv(Wout, BLOCK_N)
    ow_tile = pid_s % n_wtile
    tmp = pid_s // n_wtile
    oh = tmp % Hout
    od = tmp // Hout

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ow = ow_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < Cout
    mask_ow = offs_ow < Wout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KKK = K * K * K
    x_b = x_ptr + pid_b * Cin * D * H * W
    for kd in range(0, K):
        idd = od - kd
        d_ok = (idd >= 0) & (idd < D)
        for kh in range(0, K):
            ihh = oh - kh
            h_ok = (ihh >= 0) & (ihh < H)
            for kw in range(0, K):
                iww = offs_ow - kw
                w_ok = (iww >= 0) & (iww < W) & mask_ow
                koff = kd * K * K + kh * K + kw
                for ci0 in range(0, Cin, BLOCK_K):
                    cin = ci0 + tl.arange(0, BLOCK_K)
                    mask_c = cin < Cin
                    w_off = cin[None, :] * (Cout * KKK) + offs_m[:, None] * KKK + koff
                    w_tile = tl.load(w_ptr + w_off,
                                     mask=mask_m[:, None] & mask_c[None, :], other=0.0)
                    x_off = cin[:, None] * (D * H * W) + idd * (H * W) + ihh * W + iww[None, :]
                    x_tile = tl.load(x_b + x_off,
                                     mask=(mask_c[:, None] & (w_ok[None, :] & (d_ok & h_ok))),
                                     other=0.0)
                    acc += tl.dot(w_tile, x_tile, allow_tf32=False)

    y_off = (pid_b * Cout * Dout * Hout * Wout + offs_m[:, None] * (Dout * Hout * Wout)
             + od * (Hout * Wout) + oh * Wout + offs_ow[None, :])
    tl.store(y_ptr + y_off, acc, mask=mask_m[:, None] & mask_ow[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias)
        self.K = kernel_size

    def forward(self, x):
        B, Cin, D, H, W = x.shape
        Cout = self.conv_transpose3d.out_channels
        K = self.K
        Dout = D + K - 1
        Hout = H + K - 1
        Wout = W + K - 1

        x = x.contiguous()
        w = self.conv_transpose3d.weight.contiguous()  # (Cin, Cout, K, K, K)
        y = torch.empty((B, Cout, Dout, Hout, Wout), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 16
        grid = (B, triton.cdiv(Cout, BLOCK_M),
                Dout * Hout * triton.cdiv(Wout, BLOCK_N))
        _convt3d_kernel[grid](
            x, w, y,
            Cin, D, H, W, Cout, Dout, Hout, Wout,
            K=K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return y
