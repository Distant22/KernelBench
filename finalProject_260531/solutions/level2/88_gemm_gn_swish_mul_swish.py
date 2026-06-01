"""Level 2 task 88: Linear + GroupNorm + Swish + Multiply + Swish (V100 fused).

Pipeline
--------
1. Linear via cuBLAS (1024x8192x8192).
2. Triton pass-1: per-(sample, group) reduction over 32 elements gives
   sum / sumsq -> mean / inv_std on host.
3. Triton pass-2: apply (gamma, beta) normalize, then x*sigmoid(x), then
   *w[c], then x*sigmoid(x) — all fused in a single pass.

CoT
---
1. GEMM compute-bound (cuBLAS ~10 ms ~ peak); the rest is 5 elementwise/
   reduce kernels on a 32 MB tensor (~0.4 ms baseline).
2. Pass-1: launch (N*G,) programs, each does a 32-element reduction.
   Pass-2: tile the (CG*?? wait single-row group of 32) -> just per-group
   apply with all post-ops fused.
3. Each program loads a contiguous 32-element slab; perfectly coalesced.
4. See code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -- pass 1: per-group sum / sumsq -------------------------------------------
@triton.jit
def _gn_reduce_kernel(
    x_ptr, sum_ptr, sumsq_ptr,
    N, C, G, CG,
    BLOCK: tl.constexpr,                # next pow2 >= CG
):
    pid = tl.program_id(0)              # 0 .. N*G - 1
    n = pid // G
    g = pid % G
    base = n * C + g * CG
    offs = tl.arange(0, BLOCK)
    mask = offs < CG
    v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    s = tl.sum(tl.where(mask, v, 0.0), axis=0)
    sq = tl.sum(tl.where(mask, v * v, 0.0), axis=0)
    tl.store(sum_ptr + pid, s)
    tl.store(sumsq_ptr + pid, sq)


# -- pass 2: normalize + swish + mul + swish ---------------------------------
@triton.jit
def _gn_post_kernel(
    x_ptr, mean_ptr, inv_std_ptr,
    gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    N, C, G, CG,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    base = n * C + g * CG
    offs = tl.arange(0, BLOCK)
    mask = offs < CG
    ch = g * CG + offs                  # global channel index
    m = tl.load(mean_ptr + pid)
    inv = tl.load(inv_std_ptr + pid)
    gam = tl.load(gamma_ptr + ch, mask=mask, other=0.0)
    bet = tl.load(beta_ptr + ch, mask=mask, other=0.0)
    mw = tl.load(mw_ptr + ch, mask=mask, other=0.0)

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    y = (x - m) * inv * gam + bet                    # GroupNorm
    y = y * (1.0 / (1.0 + tl.exp(-y)))               # swish #1
    y = y * mw                                       # multiply
    y = y * (1.0 / (1.0 + tl.exp(-y)))               # swish #2
    tl.store(out_ptr + base + offs, y, mask=mask)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def fused_gn_swish_mul_swish(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    mw: torch.Tensor, num_groups: int, eps: float,
) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous() and x.dim() == 2
    N, C = x.shape
    G = num_groups
    CG = C // G
    BLOCK = _next_pow2(CG)

    sums = torch.empty(N * G, dtype=torch.float32, device=x.device)
    sumsq = torch.empty_like(sums)
    _gn_reduce_kernel[(N * G,)](
        x, sums, sumsq, N, C, G, CG, BLOCK=BLOCK, num_warps=1,
    )

    inv_n = 1.0 / float(CG)
    mean = sums * inv_n
    var = sumsq * inv_n - mean * mean
    inv_std = torch.rsqrt(var + eps)

    out = torch.empty_like(x)
    _gn_post_kernel[(N * G,)](
        x, mean, inv_std, gamma, beta, mw, out,
        N, C, G, CG, BLOCK=BLOCK, num_warps=1,
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
