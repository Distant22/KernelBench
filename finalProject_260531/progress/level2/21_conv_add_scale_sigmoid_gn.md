# Level 2 / Task 21 — Conv2D + Bias + Scale + Sigmoid + GroupNorm

- Solution: [21_conv_add_scale_sigmoid_gn.py](../../solutions/level2/21_conv_add_scale_sigmoid_gn.py)
- Baseline: `KernelBench/level2/21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py`

## Shape & Properties
- Conv: in 8 → out 32, 3×3, no pad → out `[128, 32, 254, 254]` ≈ 1.03 GB
- bias / scale: `(32, 1, 1)` per-channel
- GroupNorm: G=8, channels-per-group CG=4
- Group element count = 4·254·254 ≈ 258k

## CoT
1. **算子特性**：Conv2D 計算量低（8→32，3×3，總 ~ 0.6 GFLOPs，<1 ms）；其餘為 memory-bound。Baseline 共 5 個 kernel：conv / +bias / *scale / sigmoid / groupnorm（pass1+pass2）。
2. **融合策略**：把 `bias + scale + sigmoid` 與 GroupNorm 的 reduction pass 合併成單一 Triton kernel：每個 `(N, G)` 一個 program 序列掃描整組，邊算 elementwise、邊用 register accumulator 累加 sum / sumsq，最後寫回每組單一純量。Pass-2 再做 `(y - mean) * inv_std * γ + β`。Baseline 的 5 個 pass 變成 2 個。
3. **硬體衝突**：每 program 只負責同一個 `(n, g)`，連續位址 → 完美 coalesce。Mean/var 用 `tl.sum` 在 register / warp shuffle 完成，無 shared memory bank conflict。Sigmoid 直接 `1/(1+exp(-x))`，無分支發散。
4. 實作：`F.conv2d` → `_fused_pre_groupnorm_kernel` → host-side `mean / var / rsqrt` → `_groupnorm_apply_kernel`。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuDNN conv + 2-pass fused GN) | ✅ | 12.4 | **1.45×** | **1.15×** |

- Reference eager: 18.0 ms / compile: 14.3 ms
- 主要 speedup 來自 5 → 2 個 pass，省掉 ~3 GB read + 3 GB write 的中間流量。
- 已勝過 torch.compile inductor，代表融合 schedule 比 inductor 為 GroupNorm 排的更省。

## Notes / Next
- 還可進一步把 conv 換成 implicit-GEMM Triton 並把 epilogue 直接接到 pass-1 的 reduction，理論上能省 1 個 1 GB 寫回；但 cuDNN 8→32 3×3 已 < 1 ms，省的空間不大。
- 已記錄為 **fusion win**，不再迭代。
