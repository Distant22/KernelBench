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
| v1 (cuBLAS GEMM + Triton fused mul+leaky) | ✅ | 9.97 | 0.99× | 0.95× |

- Reference eager: 9.9 ms / compile: 9.5 ms
- GEMM 占 ≥95% 時間，融合節省理論上界 < 0.1 ms (~1%)，難以越過 eager noise。
- torch.compile 也只有 ~4% 的提升（同樣是融合 epilogue）。

## Notes / Next
- 要超過 1.0× 需要重寫 GEMM 並把 epilogue 直接 fuse 進 cuBLAS-class kernel；以 V100 FP32 純 Triton 寫 8192×8192 GEMM 通常拿到 ~ 60–70% peak，會比 cuBLAS 慢。
- 結論：cuBLAS-dominated 題型，速度被 cuBLAS 卡住；同 Level 1 的 GEMM 結論一致，記為 **GEMM-bound, 1.0× 可接受**。
