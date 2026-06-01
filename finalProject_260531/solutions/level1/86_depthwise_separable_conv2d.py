"""
KernelBench Level 1 / Problem 86 — Depthwise-Separable 2D Conv.

Pipeline: depthwise (in=64, k=3, stride=1, pad=1) -> pointwise (1x1, in=64, out=128)
Input (16, 64, 512, 512) -> intermediate (16, 64, 512, 512) -> output (16, 128, 512, 512)

V100 strategy:
- Depthwise: reuse the 3x3 stride=1 kernel from problem 82, extended to support
  symmetric `padding` (border outputs read masked-out neighbours as 0).
- Pointwise (1x1, in=64, out=128): defer to PyTorch's nn.Conv2d, which dispatches
  to a well-tuned cuBLAS GEMM (1x1 conv is exactly a (BHW, in) @ (in, out) matmul).
  Beating cuBLAS on V100 FP32 is unlikely; this is the same lesson as Level 1 GEMM.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _depthwise_conv2d_3x3_pad_kernel(
    X_ptr, W_ptr, OUT_ptr,
    H_in, W_in, H_out, W_out,
    PAD,
    stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wkh, stride_wkw,
    stride_oc, stride_oh, stride_ow,
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(2)

    x_base = X_ptr + pid_bc * stride_xc
    o_base = OUT_ptr + pid_bc * stride_oc
    w_base = W_ptr + pid_bc * stride_wc

    y_offs = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    x_offs = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    y_mask = y_offs < H_out
    x_mask = x_offs < W_out
    o_mask = y_mask[:, None] & x_mask[None, :]

    in_y = y_offs[:, None] - PAD  # (BLOCK_Y, 1)
    in_x = x_offs[None, :] - PAD  # (1, BLOCK_X)

    acc = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)
    for dy in tl.static_range(0, 3):
        for dx in tl.static_range(0, 3):
            ry = in_y + dy
            rx = in_x + dx
            ptr = x_base + ry * stride_xh + rx * stride_xw
            m = (ry >= 0) & (ry < H_in) & (rx >= 0) & (rx < W_in)
            v = tl.load(ptr, mask=m, other=0.0)
            w = tl.load(w_base + dy * stride_wkh + dx * stride_wkw)
            acc += v * w

    out_ptr = o_base + y_offs[:, None] * stride_oh + x_offs[None, :] * stride_ow
    tl.store(out_ptr, acc, mask=o_mask)


def _launch_depthwise_3x3(x: torch.Tensor, weight: torch.Tensor,
                          padding: int) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32 and x.dim() == 4
    x = x.contiguous()
    B, C, H_in, W_in = x.shape
    Cw, _, kH, kW = weight.shape
    assert Cw == C and kH == 3 and kW == 3
    H_out = H_in + 2 * padding - 2
    W_out = W_in + 2 * padding - 2

    out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
    weight_bc = weight.view(C, kH, kW).repeat(B, 1, 1).contiguous()
    x_flat = x.view(B * C, H_in, W_in)
    out_flat = out.view(B * C, H_out, W_out)

    BLOCK_Y = 4
    BLOCK_X = 512
    grid = (B * C, triton.cdiv(H_out, BLOCK_Y), triton.cdiv(W_out, BLOCK_X))
    _depthwise_conv2d_3x3_pad_kernel[grid](
        x_flat, weight_bc, out_flat,
        H_in, W_in, H_out, W_out,
        padding,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
        weight_bc.stride(0), weight_bc.stride(1), weight_bc.stride(2),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        BLOCK_Y=BLOCK_Y,
        BLOCK_X=BLOCK_X,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 bias: bool = False):
        super().__init__()
        # Match reference Model exactly: same layer types/order so seeded init
        # produces identical params.
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   bias=bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.kernel_size == 3 and self.stride == 1 and self.dilation == 1
                and not self.bias and x.is_cuda and x.dtype == torch.float32):
            d = _launch_depthwise_3x3(x, self.depthwise.weight, self.padding)
        else:
            d = self.depthwise(x)
        return self.pointwise(d)
