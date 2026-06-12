# Level 2 / Task 12 — Linear (GEMM) + Mul + LeakyReLU

- Solution: [12_gemm_mul_leakyrelu.py](../../solutions/level2/12_gemm_mul_leakyrelu.py)
- Baseline: `KernelBench/level2/12_Gemm_Multiply_LeakyReLU.py`

## Shape & Properties
- Linear: `[1024, 8192] @ [8192, 8192]^T + bias` → `[1024, 8192]`
- multiplier = 2.0, negative_slope = 0.1
- Output 32 MB (small, post-ops cheap)
- GEMM 137 GFLOPs → ~9.5 ms on V100 cuBLAS (~95% peak)

## CoT
1. **算子特性**：GEMM compute-bound（cuBLAS 主導）；mul + LeakyReLU 是兩個 elementwise pass（共 ~64 MB 記憶體流量，<0.15 ms）。
2. **策略**：cuBLAS 已接近 peak，無法替代；改而把 `*c` 與 `LeakyReLU` 融成單一 Triton pass，省一次完整 read+write。
3. **硬體**：1D flatten，`BLOCK=1024 num_warps=4`，連續位址完美 coalesce；branch-free `tl.where`。
4. 實作：`F.linear` + `_mul_leaky_kernel`。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuBLAS GEMM + Triton fused mul+leaky) | ✅ | 9.95 | 1.00× | 0.975× |

- Reference eager: 9.96 ms / compile: 9.7 ms（最新 deep profile 重新量測）。
- Nsight 拆解：candidate = cuBLAS `volta_sgemm_128x64` **9.51 ms** + 融合 epilogue **84 µs**；
  compile = `volta_sgemm_128x128` **11.0 ms** + epilogue 84 µs。**我們的 GPU 工作量其實比 compile 少**，
  唯一落後處是兩個 kernel 之間的 host-side launch overhead。
- GEMM 占 ≥98% 時間，融合節省理論上界 < 0.1 ms (~1%)，難以越過 eager noise。

## Notes / Next
- 要超過 1.0× 需要重寫 GEMM 並把 epilogue 直接 fuse 進 cuBLAS-class kernel；以 V100 FP32 純 Triton 寫 8192×8192 GEMM 通常拿到 ~ 60–70% peak，會比 cuBLAS 慢。
- **2026-06-12 第二輪嘗試（皆失敗，已還原 v1）**：
  1. `torch.compile(F.linear, max-autotune)` 只想換更快 GEMM algo → autotune 反而挑了較慢的 `128x128`，僅 0.998×。
  2. 手動 `torch.cuda.CUDAGraph` 捕捉整段 forward 消 launch overhead → 但 graph-safe 需 `copy_` 輸入 + `clone` 輸出（各 32 MB），拷貝成本 > 省下的 launch，反而掉到 0.95×。
  - **教訓**：GEMM 主導題（占 99% 時間）做 kernel 融合 / CUDA graph 都得不償失；cuBLAS GEMM 已是誠實最優。
- 結論：cuBLAS-dominated 題型，速度被 cuBLAS 卡住；同 Level 1 的 GEMM 結論一致，記為 **GEMM-bound, 0.975× 為誠實天花板**。
