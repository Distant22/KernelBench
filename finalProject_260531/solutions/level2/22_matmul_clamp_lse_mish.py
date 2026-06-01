"""Level 2 task 22: Linear + scale + residual + clamp + logsumexp(dim=1) + x*mish(x).

CoT
---
1. GEMM (1024×8192×8192) compute-bound (~10 ms cuBLAS, ~95% peak). The post-op
   sequence on the [1024, 8192] activation is memory-bound but tiny in
   absolute terms (~32 MB tensor). Final logsumexp produces [1024, 1] then
   `x * mish(x)` is element-wise on 1024 scalars.
2. Replace 5 baseline elementwise/reduction kernels with a single Triton
   kernel that, for each row: streams the row once, applies
   `clamp(4*y, lo, hi)`, computes `m = max(...)` and `l = log(sum(exp(...-m)))`,
   then writes `lse = m + l`, and finally `out = lse * mish(lse)` to a
   `[1024, 1]` tensor. (`y * scale + y = (2*scale)*y = 4y` here.)
3. One program per row, BLOCK iterates over 8192 columns. Online streaming
   max + sum-exp ensures fp32 numerical stability and no second pass.
4. Implementation below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_post_kernel(
    x_ptr, out_ptr,
    M, N,
    coef, lo, hi,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N

    # Online streaming logsumexp over the row, with elementwise transform
    # f(v) = clamp(coef * v, lo, hi) folded in.
    running_max = float("-inf")
    running_sum = 0.0
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        v = tl.load(x_ptr + base + offs, mask=mask, other=float("-inf"))
        v = v * coef
        v = tl.minimum(tl.maximum(v, lo), hi)
        # masked positions should not affect max/sum -> set to -inf
        v = tl.where(mask, v, float("-inf"))
        block_max = tl.max(v, axis=0)
        new_max = tl.maximum(running_max, block_max)
        # rescale running_sum to the new max
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(v - new_max), axis=0
        )
        running_max = new_max

    lse = running_max + tl.log(running_sum)
    # mish(x) = x * tanh(softplus(x)); out = lse * mish(lse)
    sp = tl.log(1.0 + tl.exp(lse))
    mish = lse * (tl.exp(sp) - tl.exp(-sp)) / (tl.exp(sp) + tl.exp(-sp))
    out = lse * mish
    tl.store(out_ptr + row, out)


def fused_post(x: torch.Tensor, coef: float, lo: float, hi: float) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous() and x.dim() == 2
    M, N = x.shape
    out = torch.empty((M, 1), dtype=x.dtype, device=x.device)
    BLOCK = 1024
    _fused_post_kernel[(M,)](
        x, out, M, N, float(coef), float(lo), float(hi),
        BLOCK=BLOCK, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        y = F.linear(x, self.matmul.weight, self.matmul.bias)
        # scale + (x+x) collapses to coef = 2 * scale_factor
        coef = 2.0 * self.scale_factor
        return fused_post(y, coef, self.clamp_min, self.clamp_max)
