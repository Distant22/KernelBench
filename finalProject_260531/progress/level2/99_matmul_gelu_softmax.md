# Level 2 / Task 99 — Linear + GELU + Softmax(dim=1)

- Solution: [99_matmul_gelu_softmax.py](../../solutions/level2/99_matmul_gelu_softmax.py)
- Baseline: `KernelBench/level2/99_Matmul_GELU_Softmax.py`

## Shape & Properties
- Linear: `[1024, 8192] @ [8192, 8192]^T` → `[1024, 8192]` (32 MB) → ~9.5 ms cuBLAS
- GELU + softmax dim=1

## CoT
1. **算子特性**：GEMM compute-bound (cuBLAS ~peak)；GELU+softmax memory-bound（baseline 4 passes ~0.4 ms）。
2. **融合**：cuBLAS GEMM + 單 Triton kernel 做 row softmax，inline 重算 GELU 免去 32 MB 中介 buffer。
3. **硬體**：每行一 program，BLOCK=2048 num_warps=8；連續 row read 完美 coalesce；exact GELU 用 `tl.erf`。
4. 實作見 solution。

## Result (V100)
| Version | Correct | kernel_ms | Speedup eager | Speedup compile | epilogue |
|---------|---------|-----------|---------------|-----------------|----------|
| v1 (3-pass: max → sumexp → write, 重讀 row 3 次) | ✅ | 10.0 | 0.99× | 0.96–0.99× | 170 µs |
| **v2 (2-pass online softmax：max+sumexp 同一趟 → write)** | ✅ 5/5 | **10.0** | **1.000×** | 0.96× | **132 µs** |

### v2 — online (flash-style) softmax，3-pass → 2-pass（profile-verified，JID 950116 / run 20260611_231357）
- **瓶頸歸因（v1 timeline）**：epilogue `_gelu_softmax_row_kernel` = 170 µs，重讀整列 GELU 三趟
  （pass1 求 max、pass2 求 sumexp、pass3 寫出），對 32 MB tensor 多一趟 read。
- **單一 scoped change**：把 pass1+pass2 合成 **online streaming** 一趟
  （flash-style：每個 block 更新 running max 並對 running sum 做 `exp(old_max-new_max)` rescale），
  epilogue 從 3 趟 read 降為 2 趟（1 讀算統計 + 1 讀寫出）。
- **實測結果**：epilogue **170 µs → 132 µs**（快 1.29×，且 **反超 compile 的 160 µs**），
  correctness 5/5（與 v1 max-subtraction softmax 在 fp32 tol 內等價）。
- **wall-clock 殘差**：`volta_sgemm` GEMM 占 ~98%（9.3–11 ms，100-trial event 量測本身有 run-to-run 抖動），
  `sc` 在 0.96–0.99 區間浮動屬 GEMM 量測噪音，非我的 kernel 退步——epilogue 本身已嚴格變快且反超 compile。
- **歸因**：GEMM 為 cuBLAS FP32 compute 天花板（同 P88 實測 96.5% SM peak），不可動；epilogue 已最佳化到位。

## Notes / Next
- epilogue 已比 compile 快（132 vs 160 µs），整體 wall-clock 受 GEMM 主導 ≈ 1.0× eager。
- 進一步要超越需自寫 GEMM；V100 FP32 不可行（cuBLAS ~95% peak）。
- 改動前備份：[_fallback_backup/99_matmul_gelu_softmax.before_online_softmax.py](../../solutions/level2/_fallback_backup/99_matmul_gelu_softmax.before_online_softmax.py)
