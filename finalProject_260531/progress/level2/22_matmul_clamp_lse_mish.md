# Level 2 / Task 22 — Matmul + Scale + Residual + Clamp + LogSumExp + x·Mish(x)

- Solution: [22_matmul_clamp_lse_mish.py](../../solutions/level2/22_matmul_clamp_lse_mish.py)
- Baseline: `KernelBench/level2/22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish.py`

## Shape & Properties
- Linear: `[1024, 8192] @ [8192, 8192]^T + bias` → `[1024, 8192]`
- Activation: `y * scale + y` 等價於 `coef = 2*scale_factor` 倍縮放（觀察化簡）
- clamp[-10, 10]，logsumexp dim=1 → `[1024, 1]`，再 `x*mish(x)`
- 中介 tensor 32 MB，最終輸出 4 KB

## CoT
1. **算子特性**：GEMM compute-bound（cuBLAS ~10 ms，~95% peak）；後段 5 個 elementwise/reduce kernel 在 32 MB tensor 上一共 ~0.3 ms。
2. **融合**：cuBLAS GEMM + 一個 Triton kernel 完成 `clamp(coef·y) → logsumexp_row → x*mish(x)`。每行一個 program，online streaming max + exp-sum 確保 fp32 數值穩定且只讀一次。
3. **硬體**：row-major 連續讀，coalesced；單 program loop 內用 register accumulator；`tl.maximum/minimum` 取代分支 clamp；無 shared memory。
4. 實作見 solution。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuBLAS GEMM + 1-pass fused post) | ✅ | 9.96 | **1.03×** | 0.96× |

- Reference eager: 10.3 ms / compile: 9.56 ms
- GEMM 占 ~97% 時間，後段融合可省 ~0.3 ms。
- 已小幅勝過 eager；inductor torch.compile 也做了類似融合，所以 compile 路徑略快。

## Notes / Next
- GEMM-dominated，接近天花板。要再加速需自寫 Tensor-Core-free FP32 GEMM（V100 沒 FP32 TC），不可能贏 cuBLAS。
- 結論：**fusion 微 win，1.03× over eager** ✅。
