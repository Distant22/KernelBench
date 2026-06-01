# Level 2 / Task 1 — Conv2D + ReLU + BiasAdd

- Solution: [01_conv2d_relu_biasadd.py](../../solutions/level2/01_conv2d_relu_biasadd.py)
- Baseline: `KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py`

## Shape & Properties
- Input: `[128, 64, 128, 128]` fp32, kernel `3×3`, no padding → conv out `[128, 128, 126, 126]`
- bias shape `(128, 1, 1)` (per-channel), broadcast added **after** ReLU
- Output: ~258 M elements ≈ 1.03 GB

## CoT
1. **算子特性**：Conv2D 是 compute-bound（cuDNN 已接近 peak）；後段 `relu(x) + bias[c]` 是 memory-bound 的 elementwise pass（~2 GB 讀寫）。
2. **Tiling / Memory**：把後段的 ReLU 與 bias-add 融成單一 Triton kernel，flatten 成 `(N*C, H*W)`，1D grid 沿 channel-row + HW tile。bias 每個 program 只需 1 純量。
3. **硬體衝突**：連續位址 → 完美 coalesce；無 shared memory / 無 bank conflict；`tl.maximum(x, 0)` 無分支發散；`BLOCK=1024, num_warps=4` 給 ILP。
4. **實作**：`F.conv2d` + `_fused_relu_bias_kernel`（單 pass 完成 ReLU + 加 bias）。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuDNN conv + Triton fused relu+bias) | ✅ | 22.7 | **1.08×** | **1.10×** |

- Reference eager: 24.5 ms / compile: 24.9 ms
- 主要 speedup 來自把 baseline 兩個 elementwise pass（relu、+bias）併為一次 → 省 ~1 GB read + 1 GB write。
- conv 本身 ~10–12 ms 已是 cuDNN peak，無法再壓。

## Notes
- 沿用 `nn.Conv2d` 做參數初始化以對齊 baseline 權重分佈。
- 保持 fp32，無 TF32 / BF16（V100 限制）。

## Next
- 可考慮把 `F.conv2d` 換成自訂 Triton implicit-GEMM conv 並把 epilogue 直接 fuse 進去；但 cuDNN 在 64→128 ch、3×3 已非常強，性價比低。
