"""
HAND-WRITTEN attempt v2 — KernelBench Level 1 / Problem 97: Scaled Dot-Product
Attention (honest from-scratch Triton flash-attention).

Fixes the v1 compile failure: v1 staged the D=1024 head dim as a *Python list*
of per-chunk register accumulators (`q_chunks = []; q_chunks.append(...)`) inside
`@triton.jit` across a `tl.static_range` loop, which Triton cannot trace
(`NameError: q_chunks is not defined`). v2 keeps a single full-width
`[BLOCK_M, HEAD_DIM]` accumulator (HEAD_DIM a constexpr), so there is no Python
container inside the kernel and it JIT-compiles cleanly.

Shape: B=32, H=32, S=512, D=1024, no mask, scale = 1/sqrt(D).
Strategy: standard flash-attention (online running-max / running-sum softmax),
one program per (batch*head, BLOCK_M rows). BLOCK_M is kept small (16) because
the D=1024 accumulator is wide; correctness is the goal, vendor SDPA is expected
to win on speed.
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    S, scale,
    stride_bh, stride_s, stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < S

    base = pid_bh * stride_bh

    # Load this block's Q rows once: (BLOCK_M, HEAD_DIM).
    q = tl.load(
        q_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
        mask=mask_m[:, None], other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < S

        # K tile: (BLOCK_N, HEAD_DIM); scores = scale * Q @ K^T -> (BLOCK_M, BLOCK_N)
        k = tl.load(
            k_ptr + base + cur_n[:, None] * stride_s + offs_d[None, :] * stride_d,
            mask=mask_n[:, None], other=0.0,
        )
        s = tl.dot(q, tl.trans(k), allow_tf32=False) * scale
        s = tl.where(mask_n[None, :], s, -float("inf"))

        # online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_ptr + base + cur_n[:, None] * stride_s + offs_d[None, :] * stride_d,
            mask=mask_n[:, None], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
        acc, mask=mask_m[:, None],
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        B, H, S, D = Q.shape
        scale = 1.0 / math.sqrt(D)
        q = Q.contiguous().view(B * H, S, D)
        k = K.contiguous().view(B * H, S, D)
        v = V.contiguous().view(B * H, S, D)
        o = torch.empty_like(q)

        BLOCK_M = 16
        BLOCK_N = 64
        grid = (B * H, triton.cdiv(S, BLOCK_M))
        _flash_attn_kernel[grid](
            q, k, v, o,
            S, scale,
            q.stride(0), q.stride(1), q.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
            num_warps=8, num_stages=2,
        )
        return o.view(B, H, S, D)
