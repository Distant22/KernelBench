"""
KernelBench Level 1 / Problem 9 — Tall-skinny matmul (huge output, tiny K).

Shape: A (32768, 32) @ B (32, 32768) -> C (32768, 32768)  [FP32]

V100 strategy:
- K=32 fits in a single BLOCK_K, so each program computes the *entire* tile
  (BLOCK_M, BLOCK_N) of C in one tl.dot call -- no K-loop needed.
- Output dominates traffic (4 GB) -> memory-bound; we focus on coalesced stores
  and L2 reuse via super-grouped (GROUP_M) program ordering.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tall_skinny_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Super-grouped ordering: visit BLOCK rows in chunks of GROUP_M to maximise
    # L2 reuse of A across neighbouring (pid_m, pid_n) tiles.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    a = tl.load(
        A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
        mask=a_mask, other=0.0,
    )
    b = tl.load(
        B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=b_mask, other=0.0,
    )

    acc = tl.dot(a, b, allow_tf32=False)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _launch_tall_skinny_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _tall_skinny_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=4,
        num_stages=2,
    )
    return C


class ModelNew(nn.Module):
    """Triton tall-skinny matmul specialised for V100 + tiny K dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _launch_tall_skinny_matmul(A, B)
