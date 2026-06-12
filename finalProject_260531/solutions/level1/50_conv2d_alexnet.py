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
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, Cin, H, W, Cout, Hout, Wout,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Single implicit-GEMM formulation.
    #   M = Cout                              (rows of weight)
    #   N = B * Hout * Wout                   (flattened output pixels)
    #   K = Cin * KH * KW                      (full contraction in one loop)
    # This removes the previous 121 tiny per-(kh,kw) dots with K=3 padded to 16
    # (which wasted ~81% of every dot and was 13x slower than cuDNN).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    HW = Hout * Wout
    K = Cin * KH * KW
    KHW = KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < Cout
    mask_n = offs_n < B * HW

    # Decode flattened output index -> (b, oh, ow).
    nb = offs_n // HW
    s = offs_n % HW
    oh = s // Wout
    ow = s % Wout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # Decode contraction index -> (cin, kh, kw).
        cin = offs_k // KHW
        r = offs_k % KHW
        kh = r // KW
        kw = r % KW

        # Weight is contiguous (Cout, Cin, KH, KW) == (Cout, K).
        w_off = offs_m[:, None] * K + offs_k[None, :]
        w_tile = tl.load(w_ptr + w_off,
                         mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Gather input pixels for this K block over the N tile.
        ih = oh[None, :] * SH - PH + kh[:, None]
        iw = ow[None, :] * SW - PW + kw[:, None]
        in_ok = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        x_off = (nb[None, :] * (Cin * H * W) + cin[:, None] * (H * W)
                 + ih * W + iw)
        x_tile = tl.load(x_ptr + x_off,
                         mask=mask_k[:, None] & mask_n[None, :] & in_ok,
                         other=0.0)
        acc += tl.dot(w_tile, x_tile, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
        acc += bias[:, None]

    # Store to y at (b, oc, oh, ow); flattened (b,oh,ow) == offs_n with stride HW per channel.
    y_off = nb[None, :] * (Cout * HW) + offs_m[:, None] * HW + s[None, :]
    tl.store(y_ptr + y_off, acc, mask=mask_m[:, None] & mask_n[None, :])


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

        bias = conv.bias
        has_bias = bias is not None
        b_ptr = bias.contiguous() if has_bias else x  # dummy ptr when no bias

        # Cout=96 fits in a single M-block; the input gather (x_off) is
        # independent of offs_m, so a multi-block M-grid re-gathers the same
        # scattered input 3x. One M-block removes that redundancy and relieves
        # the memory pipeline (was 82% busy at SM 20%).
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(Cout, BLOCK_M),
                triton.cdiv(B * Hout * Wout, BLOCK_N))
        _conv2d_kernel[grid](
            x, w, b_ptr, y,
            B, Cin, H, W, Cout, Hout, Wout,
            HAS_BIAS=has_bias,
            KH=KH, KW=KW, SH=SH, SW=SW, PH=PH, PW=PW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=2,
        )
        return y
