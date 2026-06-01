"""Level 2 task 56: Linear + Sigmoid + Sum(dim=1, keepdim=True).

Problem is dominated by the GEMM (128×32768×32768, ~274 GFLOPs). Sigmoid+sum
on the [128, 32768] activation are memory-bound (~16 MB), trivially fusable.

CoT
---
1. GEMM compute-bound (cuBLAS ~peak); rest is memory-bound and small.
2. F.linear -> Triton single-pass row reduction that does sigmoid + sum.
3. Per-row program, online accumulate, BLOCK=1024 num_warps=8.
4. Implementation below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _row_sigmoid_sum_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * N
    acc = 0.0
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-v))
        acc += tl.sum(tl.where(mask, s, 0.0), axis=0)
    tl.store(out_ptr + row, acc)


def row_sigmoid_sum(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    M, N = x.shape
    out = torch.empty((M, 1), dtype=x.dtype, device=x.device)
    BLOCK = 2048
    _row_sigmoid_sum_kernel[(M,)](x, out, M, N, BLOCK=BLOCK, num_warps=8)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        y = F.linear(x, self.linear.weight, self.linear.bias)
        return row_sigmoid_sum(y.contiguous())
