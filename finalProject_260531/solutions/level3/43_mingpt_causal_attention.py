"""Level 3 task 43: minGPT Causal Self-Attention, V100 SDPA-fused.

CoT
---
1. **算子特性**: B=128, T=512, dim=768, heads=8, head_dim=96.
   Baseline 物化 (B,H,T,T)=128·8·512·512·4 B = 1 GB attention matrix，
   緊接 mask + softmax + matmul，全部走 HBM；且兩個 matmul 已是 GEMM-bound。
   Memory-bound 主要在 (B,H,T,T) 的 read/write 流量。
2. **策略**: 直接呼叫 `F.scaled_dot_product_attention(is_causal=True)`，
   走 V100 上的 mem-efficient SDPA：避免物化 attention matrix，融合
   scale + causal-mask + softmax + 第二個 matmul，省 ~3 GB 流量。
   QKV 分開 split + transpose 用 `permute(2,0,3,1,4)` 一次完成。
   Test 設定 `attn_pdrop=resid_pdrop=0` → 無 RNG 議題。
3. **減少衝突**: SDPA 內部已 tiled；外部僅一次 contiguous reshape。
4. **實作**: 下方。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
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

        # QKV in one GEMM, then split→reshape→permute to (3, B, H, T, hd) view.
        qkv = self.c_attn(x)                     # (B, T, 3C)
        qkv = qkv.view(B, T, 3, H, hd)           # (B, T, 3, H, hd)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SDPA: causal=True 在 V100 上走 mem-efficient kernel，避免物化 (B,H,T,T)
        dropout_p = self.attn_pdrop if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )                                        # (B, H, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


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
