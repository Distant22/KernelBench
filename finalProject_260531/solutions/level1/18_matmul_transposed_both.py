"""
KernelBench Level 1 / Problem 18 — Matmul with both operands transposed: C = A.T @ B.T.

Shapes: A (K=8192, M=2048), B (N=4096, K=8192) -> C (M=2048, N=4096)  [FP32]

V100 strategy:
- A is (K, M) row-major: load (BLOCK_K, BLOCK_M) coalesced along M, then tl.trans.
- B is (N, K) row-major: load (BLOCK_N, BLOCK_K) coalesced along K, then tl.trans.
- Standard tiled GEMM with super-grouped pid ordering for L2 reuse.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _matmul_at_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ak, stride_am,
    stride_bn, stride_bk,
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
    b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_k[:, None] < k_remaining) & (offs_m[None, :] < M)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_K, BLOCK_M)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # (BLOCK_N, BLOCK_K)
        acc += tl.dot(tl.trans(a), tl.trans(b), allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _launch_at_bt(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    A = A.contiguous()
    B = B.contiguous()

    K, M = A.shape
    N, K2 = B.shape
    assert K == K2

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_at_bt_kernel[grid](
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
        num_stages=3,
    )
    return C


class ModelNew(nn.Module):
    """Triton GEMM: C = A.T @ B.T specialised for V100 FP32."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _launch_at_bt(A, B)
