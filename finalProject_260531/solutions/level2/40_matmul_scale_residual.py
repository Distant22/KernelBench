"""Level 2 task 40: Linear + (clone + scale + residual-add).

Observation: `y * s + y_clone == y * (s + 1)`. So the post-op collapses to a
single per-element scalar multiply by `(s + 1)`. Because that factor is a
compile-time constant, we fold it directly into the Linear weight and bias in
`__init__`:

    y = (x @ (W*(s+1)).T) + b*(s+1)

so the whole forward is a single cuBLAS GEMM with NO epilogue kernel at all.
torch.compile still emits matmul + a fused scale epilogue (extra ~2 GB output
round-trip); pre-scaling the parameters removes that entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        coef = self.scaling_factor + 1.0
        with torch.no_grad():
            self.weight = nn.Parameter(self.matmul.weight * coef)
            self.bias = nn.Parameter(self.matmul.bias * coef)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
