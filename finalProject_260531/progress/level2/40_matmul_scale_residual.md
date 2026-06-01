# Level 2 / Task 40 — Linear + Scale + ResidualAdd

- Solution: [40_matmul_scale_residual.py](../../solutions/level2/40_matmul_scale_residual.py)
- Baseline: `KernelBench/level2/40_Matmul_Scaling_ResidualAdd.py`

## Shape & Properties
- Linear: `[16384, 4096] @ [4096, 4096]^T + bias` → `[16384, 4096]` (256 MB)
- 後段：`y_clone = y; y = y*s; y = y + y_clone`，等價於 `y * (s + 1)`

## CoT
1. **算子特性**：GEMM ~549 GFLOPs，cuBLAS ~40 ms (~95% peak)；後段 256 MB tensor 上 3 個 elementwise kernel ~2 ms。
2. **化簡**：`y*s + y == y*(s+1)`，整段後處理塌縮成單一 scalar 乘法。
3. **硬體**：1D flatten，`BLOCK=4096 num_warps=8`，連續位址完美 coalesce；零分支。
4. 實作：`F.linear` + `_scale_kernel`。

## Result (V100)
| Version | Correct | mean (ms) | Speedup eager | Speedup compile |
|---------|---------|-----------|---------------|-----------------|
| v1 (cuBLAS GEMM + ×(s+1)) | ✅ | 39.1 | **1.03×** | 0.97× |

- Reference eager: 40.4 ms / compile: 37.9 ms
- 省下 baseline 的 clone/scale/add 三個 pass (~1.3 ms)。
- inductor 也做了相同代數簡化，所以 compile 路徑略快。

## Notes / Next
- GEMM-bound，無進一步空間。**fusion + algebraic simplification 微 win** ✅。
