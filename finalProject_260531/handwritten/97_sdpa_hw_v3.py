"""
HAND-WRITTEN attempt v3 — KernelBench Level 1 / Problem 97: Scaled Dot-Product
Attention (honest from-scratch Triton flash-attention), D=1024.

Why v1/v2 failed and how v3 fixes it
------------------------------------
* v1: staged the D=1024 head dim as a *Python list* of per-chunk register
  accumulators inside @triton.jit  ->  NameError, does not trace.
* v2: single full-width [BLOCK_M, HEAD_DIM=1024] accumulator. Compiles past the
  tracer but the tl.dot operands (K/V tiles of [BLOCK_N, 1024]) need ~256 KB of
  *shared memory*, far over Volta's 96 KB/SM  ->  OutOfResources(shared memory).

v3 keeps shared-memory tiles small by tiling the head dimension twice:
  (a) the OUTPUT head-dim is a GRID dimension: each program owns a BD=256-wide
      column slice of O, so the P@V tile V[BLOCK_N, BD] is only 64 KB SMEM;
  (b) the QK^T contraction is tiled in BK=256 chunks, so each Q/K tile is small.
The price is that the full S=Q@K^T softmax is recomputed once per output
column-block (HEAD_DIM/BD = 4x redundant QK work). This is the honest cost of a
portable from-scratch kernel on a 1024-wide head dim; vendor fused SDPA, which
computes QK once, is expected to win comfortably.

Shape: B=32, H=32, S=512, D=1024, no mask, scale = 1/sqrt(D).
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
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BD: tl.constexpr, BK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_do = pid_d * BD + tl.arange(0, BD)      # this program's output columns
    mask_m = offs_m < S

    base = pid_bh * stride_bh
    q_row = q_ptr + base + offs_m[:, None] * stride_s
    k_row = k_ptr + base
    v_row = v_ptr + base

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BD), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < S

        # scores[BLOCK_M, BLOCK_N] = scale * sum_d Q[m,d] K[n,d], tiled over d (BK)
        s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in range(0, HEAD_DIM, BK):
            offs_k = d0 + tl.arange(0, BK)
            qd = tl.load(q_row + offs_k[None, :] * stride_d,
                         mask=mask_m[:, None], other=0.0)            # (BLOCK_M, BK)
            kd = tl.load(k_row + cur_n[:, None] * stride_s + offs_k[None, :] * stride_d,
                         mask=mask_n[:, None], other=0.0)            # (BLOCK_N, BK)
            s += tl.dot(qd, tl.trans(kd), allow_tf32=False)
        s = s * scale
        s = tl.where(mask_n[None, :], s, -float("inf"))

        # online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # accumulate this program's output column slice: P @ V[:, offs_do]
        v = tl.load(v_row + cur_n[:, None] * stride_s + offs_do[None, :] * stride_d,
                    mask=mask_n[:, None], other=0.0)                 # (BLOCK_N, BD)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=False)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * stride_s + offs_do[None, :] * stride_d,
             acc, mask=mask_m[:, None])


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
        BD = 256          # output head-dim tile (SMEM-bound: V tile = BLOCK_N*BD)
        BK = 256          # QK contraction tile
        grid = (B * H, triton.cdiv(S, BLOCK_M), D // BD)
        _flash_attn_kernel[grid](
            q, k, v, o,
            S, scale,
            q.stride(0), q.stride(1), q.stride(2),
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BD=BD, BK=BK,
            num_warps=4, num_stages=2,
        )
        return o.view(B, H, S, D)
