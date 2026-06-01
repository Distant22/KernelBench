"""Level 2 task 45: Linear + Sigmoid + Linear + LogSumExp(dim=1).

Two large GEMMs (16384×2048×4096 and 16384×4096×1024) dominate runtime; both
are cuBLAS-optimal on V100. Sigmoid and LogSumExp are memory-bound, but only
LogSumExp can be safely "fused" into a single Triton row-reduction (sigmoid
sits between two GEMMs and feeds linear2 unchanged).

CoT
---
1. linear1, linear2 compute-bound (~30 ms cuBLAS); sigmoid 0.6 ms; lse 0.15 ms.
2. Use F.linear for both; replace `logsumexp` with a single Triton row-pass
   (online streaming max+sumexp) on [16384, 1024]. Saves a pass over 64 MB.
3. Per-row online streaming LSE -> coalesced row reads, register accumulators.
4. Implementation below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _row_lse_kernel(
    x_ptr, out_ptr, M, N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N

    running_max = float("-inf")
    running_sum = 0.0
    for off in range(0, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        v = tl.load(x_ptr + base + offs, mask=mask, other=float("-inf"))
        block_max = tl.max(v, axis=0)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(v - new_max), axis=0
        )
        running_max = new_max
    tl.store(out_ptr + row, running_max + tl.log(running_sum))


def row_logsumexp(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype, device=x.device)
    BLOCK = 1024
    _row_lse_kernel[(M,)](x, out, M, N, BLOCK=BLOCK, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.sigmoid(F.linear(x, self.linear1.weight, self.linear1.bias))
        y = F.linear(h, self.linear2.weight, self.linear2.bias)
        return row_logsumexp(y.contiguous())
