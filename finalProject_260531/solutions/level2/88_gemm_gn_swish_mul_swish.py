"""Level 2 task 88: Linear + GroupNorm + Swish + Multiply + Swish (V100 fused).

Pipeline
--------
1. Linear via cuBLAS (1024x8192x8192).
2. Single Triton kernel, **one program per row**: load the whole C=8192 row
   as a [G, CG] = [256, 32] tile, compute all 256 group mean/var reductions in
   registers (axis=1), then normalize + swish + multiply + swish, and write
   once. No intermediate mean/inv_std buffer, single global read + single
   global write.

CoT
---
1. GEMM compute-bound (cuBLAS ~10 ms ~ peak); the GroupNorm epilogue on the
   32 MB GEMM output is memory-bound.
2. Prior v1 launched N*G = 262144 tiny 1-warp programs *twice* (reduce + post),
   each reducing only 32 elements -> 756 us, 5x slower than torch.compile's
   single fused 140 us kernel. Root cause: launch/scheduling overhead of 262k
   tiny programs + an extra full read of x (the second pass) + an intermediate
   mean/inv_std round-trip.
3. Fix: fuse both passes into ONE kernel with one program per row (1024
   programs). Each loads its full row as a [G, CG] tile, reduces over the 32
   group channels in registers, applies all post-ops, writes once. Traffic
   drops from ~96 MB (read+read+write) to ~64 MB (read+write); 262144x2 program
   launches collapse to 1024.
4. See code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -- single fused pass: reduce + normalize + swish + mul + swish (per row) ----
@triton.jit
def _gn_fused_kernel(
    x_ptr, gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    C, eps,
    G: tl.constexpr, CG: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * C
    g_idx = tl.arange(0, G)                       # [G]
    c_idx = tl.arange(0, CG)                      # [CG]
    offs = g_idx[:, None] * CG + c_idx[None, :]   # [G, CG] channel indices

    x = tl.load(x_ptr + base + offs)              # [G, CG]
    inv_n = 1.0 / CG
    mean = tl.sum(x, axis=1) * inv_n              # [G]
    var = tl.sum(x * x, axis=1) * inv_n - mean * mean
    inv = 1.0 / tl.sqrt(var + eps)                # [G]

    gam = tl.load(gamma_ptr + offs)               # [G, CG]
    bet = tl.load(beta_ptr + offs)
    mw = tl.load(mw_ptr + offs)

    y = (x - mean[:, None]) * inv[:, None] * gam + bet   # GroupNorm
    y = y * (1.0 / (1.0 + tl.exp(-y)))                   # swish #1
    y = y * mw                                           # multiply
    y = y * (1.0 / (1.0 + tl.exp(-y)))                   # swish #2
    tl.store(out_ptr + base + offs, y)


def fused_gn_swish_mul_swish(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    mw: torch.Tensor, num_groups: int, eps: float,
) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous() and x.dim() == 2
    N, C = x.shape
    G = num_groups
    CG = C // G

    out = torch.empty_like(x)
    _gn_fused_kernel[(N,)](
        x, gamma, beta, mw, out,
        C, eps, G=G, CG=CG, num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups

    def forward(self, x):
        y = F.linear(x, self.gemm.weight, self.gemm.bias)
        return fused_gn_swish_mul_swish(
            y.contiguous(),
            self.group_norm.weight, self.group_norm.bias,
            self.multiply_weight, self.num_groups, self.group_norm.eps,
        )
