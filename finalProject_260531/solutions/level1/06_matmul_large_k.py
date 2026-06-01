"""
KernelBench Level 1 / Problem 6 — Matmul with large K dimension.

Reference shape: A (256, 524288) @ B (524288, 256) -> C (256, 256).
Strategy on V100 (sm_70, FP32): Split-K matmul in Triton to saturate all 80 SMs,
since the (M,N)=(256,256) output is far too small for a vanilla GEMM grid.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _split_k_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Each split-K program owns a contiguous K range [k_start, k_end).
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    a_row_ptrs = A_ptr + offs_m[:, None] * stride_am
    b_col_ptrs = B_ptr + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = k_start
    while k < k_end:
        k_offs = k + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < k_end)
        b_mask = (k_offs[:, None] < k_end) & (offs_n[None, :] < N)

        a = tl.load(a_row_ptrs + k_offs[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_col_ptrs + k_offs[:, None] * stride_bk, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=False)
        k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


def _launch_split_k_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "FP32 only on V100."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dims mismatch: {K} vs {K2}"

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    SPLIT_K = 16  # 4*4*16 = 256 blocks >> 80 SMs on V100 -> good saturation

    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), SPLIT_K)
    _split_k_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        num_warps=4,
        num_stages=3,
    )
    return C


class ModelNew(nn.Module):
    """Triton split-K GEMM specialised for V100 + large-K skinny-MN matmul."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _launch_split_k_matmul(A, B)
