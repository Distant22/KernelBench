"""
KernelBench Level 1 / Problem 16 — Matmul with transposed A: C = A.T @ B.

Shapes: A (K=8192, M=2048), B (K=8192, N=4096) -> C (M=2048, N=4096)  [FP32]

V100 strategy:
- A is stored as (K, M) row-major.  We load each A tile in its natural
  (BLOCK_K, BLOCK_M) layout (M is the contiguous dim -> coalesced) and then
  transpose with `tl.trans` before feeding `tl.dot`.  B loads are likewise
  coalesced along N.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _matmul_at_b_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ak, stride_am,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
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

    a_ptrs = A_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # M, N, K are exact multiples of the block sizes for this problem
    # (M=2048, N=4096, K=8192), so all bounds masks are statically true.
    # Dropping them removes per-iteration predicate overhead in the hot loop.
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)  # (BLOCK_K, BLOCK_M)
        b = tl.load(b_ptrs)  # (BLOCK_K, BLOCK_N)
        acc += tl.dot(tl.trans(a), b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _launch_at_b(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    A = A.contiguous()
    B = B.contiguous()

    K, M = A.shape
    K2, N = B.shape
    assert K == K2

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_at_b_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=3,
    )
    return C


class ModelNew(nn.Module):
    """Triton GEMM: C = A.T @ B specialised for V100 FP32."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _launch_at_b(A, B)
