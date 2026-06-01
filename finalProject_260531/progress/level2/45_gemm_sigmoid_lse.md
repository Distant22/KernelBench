# Level 2 / Task 45 — Linear + Sigmoid + Linear + LogSumExp

- Solution: [45_gemm_sigmoid_lse.py](../../solutions/level2/45_gemm_sigmoid_lse.py)
- Baseline: `KernelBench/level2/45_Gemm_Sigmoid_LogSumExp.py`

## Shape & Properties
- linear1: `[16384, 2048] @ [2048, 4096]^T` → `[16384, 4096]` (~275 GFLOPs)
- sigmoid 在 256 MB tensor 上
- linear2: `[16384, 4096] @ [4096, 1024]^T` → `[16384, 1024]` (~137 GFLOPs)
- logsumexp dim=1 → `[16384]`

## CoT
1. **算子特性**：兩個 GEMM compute-bound 共佔 ~30 ms (cuBLAS ~peak)；sigmoid 0.6 ms / lse 0.15 ms。
2. **融合**：sigmoid 夾在兩個 GEMM 中間，沒辦法繞過；lse 改用 Triton 單 pass 行 reduction 取代 baseline 的 `max + log + sum + exp` 多 kernel。
3. **硬體**：每行一個 program，online streaming max+sumexp，連續 row read 完美 coalesce。
4. 實作：`F.linear` ×2 + `torch.sigmoid` + `_row_lse_kernel`。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuBLAS ×2 + Triton row-LSE) | ✅ | 29.6 | **1.01×** | 0.98× |

- Reference eager: 30.0 ms / compile: 28.9 ms
- 兩個 GEMM 占 ~99% 時間，融合 lse 只省 ~0.1 ms。
- 已勝過 eager；inductor 微幅領先（compile 28.9）。

## Notes / Next
- GEMM-bound，純 Triton 寫 GEMM 在 V100 FP32 上輸 cuBLAS。維持 cuBLAS。
- 結論：**1.01× over eager** ✅。
