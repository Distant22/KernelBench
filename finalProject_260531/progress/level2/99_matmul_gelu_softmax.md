# Level 2 / Task 99 — Linear + GELU + Softmax(dim=1)

- Solution: [99_matmul_gelu_softmax.py](../../solutions/level2/99_matmul_gelu_softmax.py)
- Baseline: `KernelBench/level2/99_Matmul_GELU_Softmax.py`

## Shape & Properties
- Linear: `[1024, 8192] @ [8192, 8192]^T` → `[1024, 8192]` (32 MB) → ~9.5 ms cuBLAS
- GELU + softmax dim=1

## CoT
1. **算子特性**：GEMM compute-bound (cuBLAS ~peak)；GELU+softmax memory-bound（baseline 4 passes ~0.4 ms）。
2. **融合**：cuBLAS GEMM + 單 Triton kernel 做 row 3-pass `max(gelu) → sumexp → write`，inline 重算 GELU 三次免去 32 MB 中介 buffer。
3. **硬體**：每行一 program，BLOCK=2048 num_warps=8；連續 row read 完美 coalesce；exact GELU 用 `tl.erf`。
4. 實作見 solution。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuBLAS GEMM + 3-pass row gelu+softmax) | ✅ | 10.0 | 0.99× | 0.96× |

- Reference eager: 9.87 ms / compile: 9.6 ms
- GEMM 占 ~96%；後段融合理論上界節省 ~0.2 ms 但 Triton overhead 抵消。
- 結論：**GEMM-bound, ~1.0×**。

## Notes / Next
- inductor compile 9.6 ms 為當前 fusion schedule 上限。本實作匹配 eager。
- 進一步要超越需自寫 GEMM；V100 FP32 不可行。
