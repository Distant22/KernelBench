"""Level 3 task 48: Mamba2 SSD ReturnY, V100.

CoT
---
1. **算子特性**: B=2048, T=128, H=8, hd=64, n=16, block_len=64 → c=2 chunks/seq.
   主要計算為 4 個 einsum (chunked SSD)，沒有 RNG / dropout。因 `A/B/C` 是
   `randn` parameter，`exp(cumsum)` 後值跨度可達 1e22，重排 contraction order
   (BMM 重寫) 會放大累積誤差超出 KernelBench fp32 容差 (atol/rtol = 1e-4)。
   故必須保留 PyTorch einsum 內部的 contraction order。
2. **策略**: 維持 baseline einsum 表達式；可做 micro-optim：
   - `torch.exp_` in-place 取代 out-of-place，省 segsum/decay 中介寫回。
   - `masked_fill_` in-place。
   - `view`/`permute` 取代 `einops.rearrange`，省 Python overhead。
3. **減少衝突**: 全交給 PyTorch einsum / cuBLAS。
4. **實作**: 下方。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        assert seq_length % block_len == 0
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def _segsum(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        return x_segsum.masked_fill_(~mask, float("-inf"))

    def forward(self, X, initial_states=None):
        Bsz, T, H, P = X.shape
        L_blk = self.block_len
        c = T // L_blk

        X_blocks = X.view(Bsz, c, L_blk, H, P)
        A_blocks = self.A.view(Bsz, c, L_blk, H).permute(0, 3, 1, 2).contiguous()
        B_blocks = self.B.view(Bsz, c, L_blk, H, self.d_state)
        C_blocks = self.C.view(Bsz, c, L_blk, H, self.d_state)

        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        # 1) Diagonal
        L = torch.exp_(self._segsum(A_blocks))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp",
                              C_blocks, B_blocks, L, X_blocks)

        # 2) Intra-chunk states
        decay_states = torch.exp_(A_cumsum[..., -1:] - A_cumsum)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn",
                              B_blocks, decay_states, X_blocks)

        # 3) Inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp_(self._segsum(F.pad(A_cumsum[..., -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4) State-to-output
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp",
                             C_blocks, states, state_decay_out)

        return (Y_diag + Y_off).reshape(Bsz, T, H, P)


# Test harness parity
batch_size = 2048
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64


def get_inputs():
    return [torch.rand(batch_size, seq_length, n_heads, d_head)]


def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]
