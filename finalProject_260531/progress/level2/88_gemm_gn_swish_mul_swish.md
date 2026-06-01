# Level 2 / Task 88 — Linear + GroupNorm + Swish + Multiply + Swish

- Solution: [88_gemm_gn_swish_mul_swish.py](../../solutions/level2/88_gemm_gn_swish_mul_swish.py)
- Baseline: `KernelBench/level2/88_Gemm_GroupNorm_Swish_Multiply_Swish.py`

## Shape & Properties
- Linear: `[1024, 8192] @ [8192, 8192]^T` → `[1024, 8192]` (32 MB) → ~10 ms cuBLAS
- GroupNorm: G=256, CG=32（每組 32 個元素，超小組）
- 後段：swish · multiply · swish

## CoT
1. **算子特性**：GEMM ~95% peak (cuBLAS dominates)；後段 5 個 elementwise/reduce kernel 在 32 MB tensor 上共 ~0.4 ms。
2. **融合**：兩個 Triton pass：(a) per-group reduce sum/sumsq；(b) 同 program apply normalize + swish + mul + swish 全融。每 program 單一 32 元素 group。
3. **硬體**：BLOCK=`next_pow2(CG)=32`，num_warps=1，連續 32 元素完美 coalesce；無 bank conflict。
4. 實作見 solution。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuBLAS GEMM + 2-pass GN-fused-post) | ✅ | 10.5 | **1.01×** | 0.94× |

- Reference eager: 10.6 ms / compile: 9.83 ms
- GEMM 占 ~95% 時間，融合節省 ~0.3 ms。
- inductor 也做了類似 fusion，所以 compile 路徑略快。

## Notes
- GEMM-bound 結構，**1.01× over eager** ✅。
