# Level 2 / Task 88 — Linear + GroupNorm + Swish + Multiply + Swish

- Solution: [88_gemm_gn_swish_mul_swish.py](../../solutions/level2/88_gemm_gn_swish_mul_swish.py)
- Baseline: `KernelBench/level2/88_Gemm_GroupNorm_Swish_Multiply_Swish.py`

## Shape & Properties
- Linear: `[1024, 8192] @ [8192, 8192]^T` → `[1024, 8192]` (32 MB) → ~10 ms cuBLAS
- GroupNorm: G=256, CG=32（每組 32 個元素，超小組）
- 後段：swish · multiply · swish

## CoT
1. **算子特性**：GEMM ~95% peak (cuBLAS dominates)；後段 5 個 elementwise/reduce kernel 在 32 MB tensor 上共 ~0.4 ms。
2. **融合**：早期版本用兩個 Triton pass，最終 v2 融成單一 per-row kernel（見下）。
3. **硬體**：見各 version。
4. 實作見 solution。

## Result (V100)
| Version | Correct | kernel_ms | Speedup eager | Speedup compile | epilogue kernels |
|---------|---------|-----------|---------------|-----------------|------------------|
| v1 (2-pass: `_gn_reduce` + `_gn_post`, N·G=262144 programs ×2) | ✅ | 10.6 | 1.009× | 0.940× | 378+378 = **756 µs** |
| **v2 (single fused per-row kernel, 1024 programs)** | ✅ 5/5 | **10.0** | **1.070×** | **0.993×** | **102 µs** |

### v2 — single-pass per-row fusion（profile-verified WIN，JID 950102 / run 20260611_230807）
- **瓶頸歸因（v1，實測 timeline）**：`_gn_reduce` 與 `_gn_post` 各 378 µs（合 756 µs），
  比 torch.compile 的單一融合 epilogue（142 µs）**慢 5.3×**。根因：
  1. 啟動 **262144 個 1-warp tiny program 兩次**（每個只 reduce 32 元素）→ 排程/啟動開銷主導；
  2. 第二趟 pass 重新整批讀 x（多一次 32 MB read）；
  3. `mean`/`inv_std` 中間張量的 global round-trip。
- **單一 scoped change**：兩 pass 融成 **一個 kernel、每列一個 program**（1024 個），
  把整列 C=8192 當成 `[G, CG] = [256, 32]` tile 載入，於 register 內對 axis=1 一次算完
  256 個 group 的 mean/var，接著 normalize + swish + mul + swish，單讀單寫，
  不再有中間張量。`num_warps=8`。
- **實測結果**：epilogue **756 µs → 102 µs**（快 7.4×，且 **反超 compile 的 142 µs**），
  correctness 5/5，`speedup_compile 0.940 → 0.993`、`speedup_eager 1.009 → 1.070`。
- **真實 HW counter（Nsight Compute，candidate 主導 kernel = `volta_sgemm_128x64_tn`）**：
  `sm__throughput = 96.47% of peak`（compute-bound，**cuBLAS FP32 天花板**）、
  `gpu__dram_throughput = 12.1%`（非 memory-bound）、GEMM 單獨 9.67 ms。
- **剩餘 0.7% (vs compile) 歸因**：GEMM 已達 96.5% SM peak，是不可動的 cuBLAS 上限；
  我的 epilogue 已比 compile 快，wall-clock 殘差落在 GEMM/量測噪音內。**epilogue 層面已最佳化到位**。

## Notes
- GEMM-bound 結構：對 eager **1.070× WIN ✅**，對 compile 0.993×（實質打平，epilogue 已反超 compile）。
- v1 → v2 教訓：**避免「每組一個 tiny program」的設計**；當每個 program 工作量 < 一個 warp、且 program 數量達數十萬時，啟動/排程開銷會壓垮 kernel。改用「每列一個 program + register 內多組 reduce」可同時消除第二趟讀取與中間張量。
- 改動前備份：[_fallback_backup/88_gemm_gn_swish_mul_swish.before_singlepass_fusion.py](../../solutions/level2/_fallback_backup/88_gemm_gn_swish_mul_swish.before_singlepass_fusion.py)
