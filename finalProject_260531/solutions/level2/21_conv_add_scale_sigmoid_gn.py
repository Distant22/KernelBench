"""Level 2 task 21: Conv2D + Bias + Scale + Sigmoid + GroupNorm, V100 fused.

Pipeline
--------
1. Conv2D via cuDNN (out shape [128, 32, 254, 254], ~1 GB).
2. Triton pass-1: fuse `y = sigmoid((conv + bias) * scale)`, write y to a
   temp buffer, AND simultaneously accumulate per-(sample, group) sum and
   sum-of-squares for the GroupNorm reduction. Each output element is
   touched exactly once instead of three times in baseline.
3. Compute mean/inv-std on host with one tiny launch on the (N*G, ) buffers.
4. Triton pass-2: `out = (y - mean) * inv_std * gamma + beta`.

CoT
---
1. Operator: conv compute-bound (negligible here, 8->32 ch); rest is memory-
   bound elementwise / groupwise reduction over ~1 GB.
2. Tiling: pass-1 launches `(N*G, num_tiles_per_group)`. Each program covers
   a chunk of the (channels-per-group * H * W) elements of one group. Atomic
   adds (or per-tile partial reductions reduced on host) accumulate sum and
   sum-of-squares. We use *per-program partials* + a host-side reduction
   to avoid `tl.atomic_add` on fp32 accumulators (atomics work on V100 but
   are slow). Implemented as: launch one program per group with a serial
   loop over tiles -> register accumulators, then store final sum/sumsq once.
3. Hardware: contiguous (C, H, W) chunks, perfect coalescing. Single-warp
   reduction at end uses `tl.sum`. No shared memory, no bank conflict.
4. See code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass-1: y = sigmoid((x + bias[c]) * scale[c]); accumulate sum, sumsq per group.
# Launch grid: (N * G,)  -- one program per (sample, group).
# Each program does a serial loop over its group's elements.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_pre_groupnorm_kernel(
    x_ptr, bias_ptr, scale_ptr, y_ptr,
    sum_ptr, sumsq_ptr,
    N, C, HW, G, CG,                       # CG = C // G  (channels per group)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)            # 0 .. N*G - 1
    n = pid // G
    g = pid % G
    c0 = g * CG                       # first channel of this group

    group_numel = CG * HW
    # base offset of this (n, g) block in the [N, C, HW] flattened tensor
    base = n * C * HW + c0 * HW

    # Per-program scalar accumulators (compiled into registers).
    s_sum = 0.0
    s_sumsq = 0.0

    # Iterate over all elements of the group in BLOCK-sized chunks.
    for off in range(0, group_numel, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < group_numel
        # Channel index inside the group: offs // HW   -> bias/scale index.
        local_c = offs // HW
        ch = c0 + local_c

        b = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        s = tl.load(scale_ptr + ch, mask=mask, other=0.0)
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        v = (x + b) * s
        # sigmoid in fp32
        y = 1.0 / (1.0 + tl.exp(-v))
        tl.store(y_ptr + base + offs, y, mask=mask)
        s_sum += tl.sum(tl.where(mask, y, 0.0), axis=0)
        s_sumsq += tl.sum(tl.where(mask, y * y, 0.0), axis=0)

    # Write per-group totals.
    tl.store(sum_ptr + pid, s_sum)
    tl.store(sumsq_ptr + pid, s_sumsq)


# ---------------------------------------------------------------------------
# Pass-2: out = (y - mean[n,g]) * inv_std[n,g] * gamma[c] + beta[c]
# Launch grid: (N*G, num_tiles).
# ---------------------------------------------------------------------------
@triton.jit
def _groupnorm_apply_kernel(
    y_ptr, mean_ptr, inv_std_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, C, HW, G, CG,
    BLOCK: tl.constexpr,
):
    pid_ng = tl.program_id(0)
    pid_t = tl.program_id(1)
    n = pid_ng // G
    g = pid_ng % G
    c0 = g * CG
    group_numel = CG * HW
    base = n * C * HW + c0 * HW

    mean = tl.load(mean_ptr + pid_ng)
    inv_std = tl.load(inv_std_ptr + pid_ng)

    offs = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < group_numel
    local_c = offs // HW
    ch = c0 + local_c
    gam = tl.load(gamma_ptr + ch, mask=mask, other=0.0)
    bet = tl.load(beta_ptr + ch, mask=mask, other=0.0)

    y = tl.load(y_ptr + base + offs, mask=mask, other=0.0)
    z = (y - mean) * inv_std * gam + bet
    tl.store(out_ptr + base + offs, z, mask=mask)


def fused_pre_groupnorm(
    x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor,
    gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, eps: float,
) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.float32
    N, C, H, W = x.shape
    HW = H * W
    G = num_groups
    CG = C // G
    group_numel = CG * HW

    bias_flat = bias.contiguous().view(-1)
    scale_flat = scale.contiguous().view(-1)
    gamma_flat = gamma.contiguous().view(-1)
    beta_flat = beta.contiguous().view(-1)

    y = torch.empty_like(x)
    sums = torch.empty(N * G, dtype=torch.float32, device=x.device)
    sumsq = torch.empty_like(sums)

    BLOCK1 = 1024
    _fused_pre_groupnorm_kernel[(N * G,)](
        x, bias_flat, scale_flat, y, sums, sumsq,
        N, C, HW, G, CG, BLOCK=BLOCK1, num_warps=4,
    )

    inv_n = 1.0 / float(group_numel)
    mean = sums * inv_n
    var = sumsq * inv_n - mean * mean
    inv_std = torch.rsqrt(var + eps)

    out = torch.empty_like(x)
    BLOCK2 = 1024
    grid2 = (N * G, triton.cdiv(group_numel, BLOCK2))
    _groupnorm_apply_kernel[grid2](
        y, mean, inv_std, gamma_flat, beta_flat, out,
        N, C, HW, G, CG, BLOCK=BLOCK2, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups

    def forward(self, x):
        y = F.conv2d(x, self.conv.weight, self.conv.bias)
        return fused_pre_groupnorm(
            y, self.bias, self.scale,
            self.group_norm.weight, self.group_norm.bias,
            self.num_groups, self.group_norm.eps,
        )
