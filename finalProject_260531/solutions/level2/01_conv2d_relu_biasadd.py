"""Level 2 task 1: Conv2D + ReLU + BiasAdd, V100 fused.

Strategy
--------
- Conv2D dominates (cuDNN ~ optimal on V100 FP32 NCHW); call F.conv2d.
- The post-conv steps `relu(x) + bias[c]` is a memory-bound elementwise op
  (~1 GB read + 1 GB write). PyTorch baseline runs relu and bias-add as two
  separate kernels => 2x memory traffic. We fuse them into a single Triton
  pass keyed by (n*c, h*w) so each output element is touched exactly once.

CoT (4-step)
------------
1. Operator: conv2d (compute-bound, cuDNN handles it). post-op: ReLU + per-channel bias add (memory-bound, elementwise).
2. Tiling: flatten output to (N*C, H*W). 1D grid over channels-rows with BLOCK_HW per program. bias is loaded once per program (single scalar per channel).
3. Hardware: contiguous loads/stores -> coalesced. No shared memory needed. No bank conflict. Branch-free relu via tl.maximum.
4. Implementation below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_relu_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    C, HW,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)         # n*C + c
    col = tl.program_id(1)
    c = row % C
    b = tl.load(bias_ptr + c)
    offs = col * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    base = row * HW + offs
    val = tl.load(x_ptr + base, mask=mask, other=0.0)
    val = tl.maximum(val, 0.0) + b
    tl.store(out_ptr + base, val, mask=mask)


def fused_relu_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    bias_flat = bias.contiguous().view(-1)
    assert bias_flat.numel() == C, f"bias must have {C} elements, got {bias_flat.numel()}"
    BLOCK = 1024
    grid = (N * C, triton.cdiv(HW, BLOCK))
    _fused_relu_bias_kernel[grid](x, bias_flat, out, C, HW, BLOCK=BLOCK, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        # Reuse PyTorch Conv2d for parameter init parity with baseline.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        y = F.conv2d(x, self.conv.weight, self.conv.bias)
        return fused_relu_bias(y, self.bias)
