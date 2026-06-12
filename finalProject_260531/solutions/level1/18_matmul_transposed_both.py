"""
KernelBench Level 1 / Problem 18 — Matmul with both operands transposed: C = A.T @ B.T.

Shapes: A (K=8192, M=2048), B (N=4096, K=8192) -> C (M=2048, N=4096)  [FP32]

V100 strategy (reformulated):
- C = A.T @ B.T = (B @ A).T.  Compute D = B @ A directly: D[n, m] = sum_k B[n,k]*A[k,m] = C[m,n].
- B is (N, K) row-major: load (BLOCK_N, BLOCK_K) tile coalesced along K -> standard lhs.
- A is (K, M) row-major: load (BLOCK_K, BLOCK_M) tile coalesced along M -> standard rhs.
- tl.dot(b_tile, a_tile) needs NO in-loop transpose (the old double-trans spilled regs).
- Accumulate (BLOCK_N, BLOCK_M), transpose ONCE at the end, coalesced store to C.
- Super-grouped pid ordering for L2 reuse.
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

    # D = B @ A, D[n,m] = sum_k B[n,k] * A[k,m].
    # b_tile: (BLOCK_N, BLOCK_K) coalesced along K (stride_bk == 1).
    # a_tile: (BLOCK_K, BLOCK_M) coalesced along M (stride_am == 1).
    b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    a_ptrs = A_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    # M, N, K are exact multiples of the block sizes for this problem
    # (M=2048, N=4096, K=8192), so all bounds masks are statically true.
    # Dropping them removes per-iteration predicate overhead in the hot loop.
    for k in range(0, K, BLOCK_K):
        b = tl.load(b_ptrs)  # (BLOCK_N, BLOCK_K)
        a = tl.load(a_ptrs)  # (BLOCK_K, BLOCK_M)
        acc += tl.dot(b, a, allow_tf32=False)  # (BLOCK_N, BLOCK_M); no in-loop transpose
        b_ptrs += BLOCK_K * stride_bk
        a_ptrs += BLOCK_K * stride_ak

    # acc[n_local, m_local] = C[m, n]; transpose once -> (BLOCK_M, BLOCK_N), coalesced store.
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, tl.trans(acc))


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
        num_warps=8,
        num_stages=3,
    )
    return C


class ModelNew(nn.Module):
    """Triton GEMM: C = A.T @ B.T specialised for V100 FP32."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _launch_at_bt(A, B)
