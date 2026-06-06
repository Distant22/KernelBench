"""Level 3 task 1: MLP (3 Linear + 2 ReLU), V100 fused-epilogue.

CoT
---
1. **算子特性**: 3 GEMMs dominate.
   - L1: (128, 16384) × (16384, 16384) → (128, 16384)  ~6.87e10 FLOPs
   - L2: same shape                                    ~6.87e10 FLOPs
   - L3: (128, 16384) × (16384, 8192)  → (128, 8192)   ~3.44e10 FLOPs
   Total ~1.7e11 FLOPs. V100 FP32 peak ~14 TFLOPs → compute-bound, lower
   bound ~12 ms. Hidden activations only 8 MB each → ReLU passes are <20 µs.
2. **策略**: cuBLAS FP32 GEMM 已 ~95% peak，沒有理由用 Triton 重寫 GEMM
   (Level 1 task 1–4 已驗證 0.5–0.7×). 唯一可榨的就是 epilogue:
   - 嘗試 `torch._C._nn.linear_relu` / `torch._addmm_activation`
     (cuBLASLt RELU epilogue) 將 ReLU 融入 GEMM 收尾，省掉一個 8 MB 寫回。
   - Fallback：`F.linear` + 自行 in-place `relu_`。
3. **減少衝突**: 全部交給 cuBLAS / cuBLASLt，無 shared memory 議題。
4. **實作**: 下方。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _try_addmm_activation(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """cuBLASLt fused linear+relu via torch._addmm_activation if available.

    Computes  bias + x @ w.T  with ReLU epilogue.  Returns None if the op is
    unavailable on this build.
    """
    fn = getattr(torch, "_addmm_activation", None)
    if fn is None:
        return None
    try:
        return fn(bias, x, w.t(), use_gelu=False)
    except (RuntimeError, TypeError):
        return None


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        layers = []
        cur = input_size
        for sz in layer_sizes:
            layers.append(nn.Linear(cur, sz))
            cur = sz
        layers.append(nn.Linear(cur, output_size))
        # Keep weights as a ModuleList for parameter registration parity.
        self.linears = nn.ModuleList(layers)

        # Detect epilogue support once.
        probe_x = torch.zeros(1, 1, device="cpu")
        probe_w = torch.zeros(1, 1, device="cpu")
        probe_b = torch.zeros(1, device="cpu")
        self._has_addmm_activation = _try_addmm_activation(probe_b, probe_x, probe_w) is not None

    def forward(self, x):
        # Hidden layers with ReLU
        n_hidden = len(self.linears) - 1
        for i in range(n_hidden):
            lin = self.linears[i]
            out = None
            if self._has_addmm_activation:
                out = _try_addmm_activation(lin.bias, x, lin.weight)
            if out is None:
                out = F.linear(x, lin.weight, lin.bias)
                out = F.relu(out, inplace=True)
            x = out
        # Final linear, no activation
        return F.linear(x, self.linears[-1].weight, self.linears[-1].bias)


# Test harness parity (KernelBench imports these names from the file).
batch_size = 128
input_size = 16384
layer_sizes = [16384, 16384]
output_size = 8192


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, layer_sizes, output_size]
