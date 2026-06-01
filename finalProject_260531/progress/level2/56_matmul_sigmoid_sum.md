# Level 2 / Task 56 — Linear + Sigmoid + Sum(dim=1)

- Solution: [56_matmul_sigmoid_sum.py](../../solutions/level2/56_matmul_sigmoid_sum.py)
- Baseline: `KernelBench/level2/56_Matmul_Sigmoid_Sum.py`

## Shape & Properties
- Linear: `[128, 32768] @ [32768, 32768]^T` → `[128, 32768]` (16 MB)
- Sigmoid + sum dim=1 → `[128, 1]`

## CoT
1. **算子特性**：GEMM 274 GFLOPs ≈ 20 ms (cuBLAS ~peak)；後段 16 MB tensor 上 sigmoid+sum 約 0.05 ms。
2. **融合**：F.linear → 單 Triton row-pass 同時做 sigmoid + sum（流式累加）。
3. **硬體**：每行一個 program，row read 完美 coalesce；register accumulator。
4. 實作：F.linear + `_row_sigmoid_sum_kernel`。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuBLAS GEMM + Triton fused sigmoid+sum) | ✅ | 20.1 | 1.00× | 1.00× |

- GEMM 占 ~99.7% 時間，融合空間 < 0.05 ms。
- 結論：**GEMM-bound, 1.0×**。

## Notes
- 進一步要超越需要把 GEMM 直接寫成 Triton 並把 sigmoid+sum 做成 epilogue，但 Triton FP32 GEMM 在 V100 上會輸 cuBLAS。維持現狀。
