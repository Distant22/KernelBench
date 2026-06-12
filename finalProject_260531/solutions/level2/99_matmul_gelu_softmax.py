"""Level 2 task 99: Linear + GELU + Softmax(dim=1).

Pipeline: cuBLAS linear -> single Triton kernel that does row-wise GELU+softmax
in 3 passes over the same row (recomputing GELU inline; saves the 32 MB
GELU intermediate buffer). With BLOCK=2048, num_warps=8 each row of 8192
elements fits in 4 chunks. Bit-exact w.r.t. torch.softmax(F.gelu(...)) up
to fp32 tolerance.

CoT
---
1. GEMM 1024x8192x8192 compute-bound (cuBLAS ~10 ms ~ peak); GELU+softmax
   on [1024, 8192] (32 MB) memory-bound, baseline does 4 passes (~0.4 ms).
2. Fuse into 3-pass per-row kernel: pass1 max(gelu(x)), pass2 sumexp,
   pass3 write. Removes the GELU intermediate buffer.
3. Per-row program; coalesced row reads; register accumulators; bit-exact
   max-subtraction softmax.
4. See code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gelu(x):
    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))


@triton.jit
def _gelu_softmax_row_kernel(
    x_ptr, out_ptr, M, N, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N

    # Pass 1: online max + sumexp in a single read (flash-style rescaling).
    rmax = float("-inf")
    rsum = 0.0
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        v = tl.load(x_ptr + base + offs, mask=mask, other=float("-inf"))
        g = _gelu(v)
        g = tl.where(mask, g, float("-inf"))
        block_max = tl.max(g, axis=0)
        new_max = tl.maximum(rmax, block_max)
        rsum = rsum * tl.exp(rmax - new_max) + tl.sum(
            tl.where(mask, tl.exp(g - new_max), 0.0), axis=0
        )
        rmax = new_max
    inv = 1.0 / rsum

    # Pass 2: out = exp(gelu(x) - rmax) / rsum
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        g = _gelu(v)
        tl.store(out_ptr + base + offs, tl.exp(g - rmax) * inv, mask=mask)


def fused_gelu_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = 2048
    _gelu_softmax_row_kernel[(M,)](x, out, M, N, BLOCK=BLOCK, num_warps=8)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y = F.linear(x, self.linear.weight, self.linear.bias)
        return fused_gelu_softmax(y.contiguous())
