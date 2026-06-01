"""Level 2 task 40: Linear + (clone + scale + residual-add).

Observation: `y * s + y_clone == y * (s + 1)`. So the post-op collapses to a
single per-element scalar multiply. We fuse it into the linear's bias-add by
just calling F.linear and a single Triton scale kernel.

CoT
---
1. GEMM (16384×4096×4096) compute-bound, cuBLAS ~ peak.
2. Replace 3 elementwise kernels (clone / scale / add) with 1 fused multiply.
3. 1D flattened, BLOCK=4096, num_warps=8, coalesced access.
4. See code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _scale_kernel(x_ptr, y_ptr, coef, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, v * coef, mask=mask)


def fused_scale(x: torch.Tensor, coef: float) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _scale_kernel[grid](x, out, float(coef), n, BLOCK=BLOCK, num_warps=8)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        y = F.linear(x, self.matmul.weight, self.matmul.bias)
        return fused_scale(y, self.scaling_factor + 1.0)
