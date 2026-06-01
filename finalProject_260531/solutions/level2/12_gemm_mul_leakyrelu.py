"""Level 2 task 12: Linear (GEMM) + Mul + LeakyReLU, V100 fused.

CoT
---
1. GEMM (1024×8192×8192) is compute-bound; cuBLAS FP32 ~ near peak. Post-ops
   (x * c) and LeakyReLU are memory-bound elementwise passes over 8 M elems
   (≈ 32 MB) — small absolute time but two passes in baseline.
2. Fuse `mul + leaky_relu` into one Triton pass: 1D flattened, BLOCK=1024.
3. Branch-free LeakyReLU via `tl.where`. Coalesced contiguous access; no
   shared memory; no bank conflict.
4. Implementation below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mul_leaky_kernel(
    x_ptr, y_ptr, mult, slope, n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x * mult
    x = tl.where(x >= 0.0, x, x * slope)
    tl.store(y_ptr + offs, x, mask=mask)


def fused_mul_leaky(x: torch.Tensor, mult: float, slope: float) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _mul_leaky_kernel[grid](x, out, mult, slope, n, BLOCK=BLOCK, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        y = F.linear(x, self.gemm.weight, self.gemm.bias)
        return fused_mul_leaky(y, self.multiplier, self.negative_slope)
