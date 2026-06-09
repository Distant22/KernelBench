"""
HAND-WRITTEN attempt — KernelBench Level 1 / Problem 97: Scaled Dot-Product Attention.

Honest measurement of a from-scratch Triton flash-attention vs PyTorch's fused
SDPA (fallback scored ~1.000x). The hard part here is the head dim D=1024, which
is far larger than typical flash kernels assume (<=128). We handle it by splitting
D into NUM_D chunks of BD=128 and keeping a *Python list* of per-chunk register
accumulators (unrolled at trace time), so neither Q/K/V tiles nor the O
accumulator ever materialise the full 1024-wide row at once.

Shape: B=32, H=32, S=512, D=1024. No mask. scale = 1/sqrt(D).
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BD: tl.constexpr, NUM_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < S

    base = pid_bh * stride_bh

    # Preload Q chunks (one (BLOCK_M, BD) tile per head-dim chunk).
    q_chunks = []
    for di in tl.static_range(NUM_D):
        d = di * BD + tl.arange(0, BD)
        q = tl.load(q_ptr + base + offs_m[:, None] * stride_s + d[None, :] * stride_d,
                    mask=mask_m[:, None], other=0.0)
        q_chunks.append(q)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_chunks = [tl.zeros((BLOCK_M, BD), dtype=tl.float32) for _ in tl.static_range(NUM_D)]

    for start_n in range(0, S, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < S

        # scores = scale * Q @ K^T  (contract over full D via chunked dots)
        s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for di in tl.static_range(NUM_D):
            d = di * BD + tl.arange(0, BD)
            k = tl.load(k_ptr + base + cur_n[:, None] * stride_s + d[None, :] * stride_d,
                        mask=mask_n[:, None], other=0.0)            # (BLOCK_N, BD)
            s += tl.dot(q_chunks[di], tl.trans(k), allow_tf32=False)
        s = s * scale
        s = tl.where(mask_n[None, :], s, -float("inf"))

        # online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        for di in tl.static_range(NUM_D):
            d = di * BD + tl.arange(0, BD)
            v = tl.load(v_ptr + base + cur_n[:, None] * stride_s + d[None, :] * stride_d,
                        mask=mask_n[:, None], other=0.0)            # (BLOCK_N, BD)
            o_chunks[di] = o_chunks[di] * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
        m_i = m_new

    for di in tl.static_range(NUM_D):
        d = di * BD + tl.arange(0, BD)
        o = o_chunks[di] / l_i[:, None]
        tl.store(o_ptr + base + offs_m[:, None] * stride_s + d[None, :] * stride_d,
                 o, mask=mask_m[:, None])


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

        BD = 128
        NUM_D = D // BD
        BLOCK_M = 32
        BLOCK_N = 64
        grid = (B * H, triton.cdiv(S, BLOCK_M))
        _flash_attn_kernel[grid](
            q, k, v, o,
            S, scale,
            q.stride(0), q.stride(1), q.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BD=BD, NUM_D=NUM_D,
            num_warps=4, num_stages=2,
        )
        return o.view(B, H, S, D)
