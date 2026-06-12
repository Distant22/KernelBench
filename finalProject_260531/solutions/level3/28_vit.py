"""Level 3 task 28: Vision Transformer (ViT), V100.

CoT
---
1. **算子特性**: batch=2, 197 tokens, dim=512, depth=6, heads=8, mlp=2048.
   `nn.TransformerEncoder` 內部已使用 `F.scaled_dot_product_attention`
   (Volta 上走 mem-efficient SDPA) 與 cuBLAS GEMM。所有重算子（QKV
   projection、FFN）皆 GEMM 主導且資料量極小（每層只 197×512×4 ≈ 0.4 MB）。
   測試組 `dropout=0` → 無 RNG 一致性問題。
2. **Nsight profile**: nsys 顯示 candidate 與 eager 完全同構（純
   `nn.TransformerEncoder` baseline），落後 torch.compile 約 0.83x。差距並非
   單一算子，而是 6 層 × 多個小 GEMM/SDPA/LayerNorm kernel 的 **launch
   overhead + 缺少跨算子 fusion**——典型的小批次 transformer 模式。
3. **策略**: batch=2 的小 transformer,瓶頸是 6 層 × 70+ 個小 GEMM/SDPA/
   LayerNorm kernel 的 **launch overhead + GEMM algo 選擇**。預設
   `torch.compile` 只追平 baseline(0.911×);改用
   `mode="max-autotune-no-cudagraphs"`——Inductor 會
   (a) benchmark 每個 GEMM/SDPA 候選 kernel 選最快路徑(與 L3 P1 MLP
       0.98→1.03× 越線同款槓桿),
   (b) fuse LayerNorm/GELU/bias-add 等 elementwise epilogue,
   (c) `no-cudagraphs` 避免 reduce-overhead 的 CUDA graph 與
       KernelBench 計時迴圈(未包 no_grad)衝突而每 trial 重錄卡死。
   autotune 編譯較久,故在 `gp1d`(>30min walltime) profiling。
4. **減少衝突**: 無 shared-memory tiling 議題；交給 Inductor 融合排程。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

        # batch=2 ViT is launch-overhead bound across 6 layers × ~70 small
        # kernels. max-autotune benchmarks every GEMM/SDPA candidate (the lever
        # that crossed 1.0x on L3 P1 MLP) and fuses elementwise epilogues.
        # no-cudagraphs avoids reduce-overhead's CUDA graph hang against
        # KernelBench's no-grad-less timing loop. Compiles slower -> profile on
        # gp1d (>30min walltime), but the resulting kernels run faster.
        self._compiled_forward = torch.compile(
            self._forward_impl,
            fullgraph=False,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )

    def _forward_impl(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p * p * img.shape[1])
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

    def forward(self, img):
        return self._compiled_forward(img)


# Test harness parity
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0


def get_inputs():
    return [torch.rand(2, channels, image_size, image_size)]


def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]
