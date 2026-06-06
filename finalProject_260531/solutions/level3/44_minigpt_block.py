"""Level 3 task 44: minGPT Block (LN + Causal Attn + LN + MLP), V100 SDPA-fused.

CoT
---
1. **算子特性**: 與 task 43 同尺度 (B=128, T=512, dim=768, heads=8) 加上
   兩個 LayerNorm 與 MLP (dim → 4·dim → dim) + NewGELU。Attention block
   依舊是最大開銷 (~25 ms manual)，MLP ~ 15 ms (兩個 GEMM)，LN/GELU < 1 ms。
   Test `attn_pdrop=resid_pdrop=0` → 無 RNG 議題。
2. **策略**:
   - Attention: 同 task 43，`F.scaled_dot_product_attention(is_causal=True)`。
   - 其餘 (LN/MLP/GELU/residual) 維持 PyTorch；它們已 cuBLAS / 內建 fused。
   - GELU 使用 `F.gelu(x, approximate='tanh')` 替代 manual `tanh(...)` 表達式
     (PyTorch 內建 fused，省 4 個 elementwise pass)。
3. **減少衝突**: 全交給 SDPA 與 cuBLAS。
4. **實作**: 下方。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
        )
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_pdrop = float(attn_pdrop)

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        hd = C // H
        qkv = self.c_attn(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dropout_p = self.attn_pdrop if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class _NewGELU(nn.Module):
    """Tanh-approx GELU; uses PyTorch's fused F.gelu(approximate='tanh')."""

    def forward(self, x):
        return F.gelu(x, approximate="tanh")


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = _CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=_NewGELU(),
            dropout=nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


# Test harness parity
batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0


def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]


def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
