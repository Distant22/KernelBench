"""
KernelBench Level 1 / Problem 82 — Depthwise 2D Conv (3x3, stride=1, pad=0).

Shape: input (16, 64, 512, 512) FP32 ≈ 64 MB, output (16, 64, 510, 510).
Compute is light (~2.4 GFLOPs) ⇒ memory-bound.

V100 strategy:
- Each Triton program produces a BLOCK_Y x BLOCK_X output tile for one
  (batch, channel) plane. Grid = (B*C, ceil(H_out/BLOCK_Y), ceil(W_out/BLOCK_X)).
- Per (batch, channel) we load 9 scalar weights once from constant-like
  registers and reuse them for the entire tile.
- Inner 3x3 unrolled; L1 / read-only cache absorbs the 9 overlapping reads
  per tile naturally.

Constraints:
- We register ourselves as a regular `nn.Conv2d` so KernelBench's seeded
  weight init produces identical weights to the reference. forward()
  dispatches to the Triton kernel using `self.conv2d.weight`.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _depthwise_conv2d_3x3_kernel(
    X_ptr, W_ptr, OUT_ptr,
    BC, H_in, W_in, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(2)

    # Decode (batch, channel) from flat pid_bc using the channel stride layout
    # (we treat input as (B*C, H, W) when iterating, with row stride H_in*W_in).
    # OUT_ptr already accounts for batch+channel via its strides; precompute base.
    x_base = X_ptr + pid_bc * stride_xc  # treat as one (H_in, W_in) plane
    o_base = OUT_ptr + pid_bc * stride_oc

    # Weight pointer: weight is (BC, 3, 3) — pre-broadcast on host so a flat
    # pid_bc gives the correct channel weight directly.
    w_base = W_ptr + pid_bc * stride_wc

    y_offs = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    x_offs = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    y_mask = y_offs < H_out
    x_mask = x_offs < W_out
    o_mask = y_mask[:, None] & x_mask[None, :]

    # Output position (y, x) corresponds to input window starting at (y, x) for pad=0.
    in_y = y_offs[:, None]  # (BLOCK_Y, 1)
    in_x = x_offs[None, :]  # (1, BLOCK_X)

    acc = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
    for dy in tl.static_range(0, 3):
        for dx in tl.static_range(0, 3):
            ry = in_y + dy
            rx = in_x + dx
            ptr = x_base + ry * stride_xh + rx * stride_xw
            m = (ry < H_in) & (rx < W_in)
            v = tl.load(ptr, mask=m, other=0.0)
            w = tl.load(w_base + dy * stride_wkh + dx * stride_wkw)
            acc += v * w

    out_ptr = o_base + y_offs[:, None] * stride_oh + x_offs[None, :] * stride_ow
    tl.store(out_ptr, acc, mask=o_mask)


def _launch_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
    assert weight.is_cuda and weight.dtype == torch.float32
    x = x.contiguous()
    B, C, H_in, W_in = x.shape
    Cw, _, kH, kW = weight.shape
    assert Cw == C and kH == 3 and kW == 3
    H_out = H_in - 2
    W_out = W_in - 2

    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # nn.Conv2d(groups=C) stores weight as (C, 1, kH, kW). For each program we
    # need a per-(B*C) weight pointer; a tiny 36 KB repeat avoids modulo math
    # in the kernel.
    weight_bc = weight.view(C, kH, kW).repeat(B, 1, 1).contiguous()  # (B*C, 3, 3)

    x_flat = x.view(B * C, H_in, W_in)
    out_flat = out.view(B * C, H_out, W_out)

    BLOCK_Y = 4
    BLOCK_X = 512
    grid = (
        B * C,
        triton.cdiv(H_out, BLOCK_Y),
        triton.cdiv(W_out, BLOCK_X),
    )
    _depthwise_conv2d_3x3_kernel[grid](
        x_flat, weight_bc, out_flat,
        B * C, H_in, W_in, H_out, W_out,
        x_flat.stride(0), x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
        weight_bc.stride(0), weight_bc.stride(1), weight_bc.stride(2),
        out_flat.stride(0), out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        BLOCK_Y=BLOCK_Y,
        BLOCK_X=BLOCK_X,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super().__init__()
        # Match the reference Model exactly so the seeded weight init yields
        # identical parameters in correctness check.
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size,
                                stride=stride, padding=padding,
                                groups=in_channels, bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.kernel_size == 3 and self.stride == 1 and self.padding == 0
                and not self.bias and x.is_cuda and x.dtype == torch.float32):
            return _launch_depthwise_conv2d(x, self.conv2d.weight)
        return self.conv2d(x)
