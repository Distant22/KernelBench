# Level 1 / Problem 18 — Matmul Both Transposed (C = A.T @ B.T)

**Shape**: A (8192, 2048), B (4096, 8192) → C (2048, 4096) FP32  
**Baseline (PyTorch eager)**: 9.13 ms — cuBLAS ~15.0 TFLOPS (~96% V100 FP32 peak)  
**Baseline (torch.compile)**: 9.13 ms

## Iteration v1 — Triton GEMM with double `tl.trans` (採用)

- Source: [solutions/level1/18_matmul_transposed_both.py](../../solutions/level1/18_matmul_transposed_both.py)
- 雙邊 coalesced load + `tl.trans` 轉置；config 同 task 16 (128/128/32, num_warps=4, num_stages=3)

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Runtime mean / std | 15.2 ms / 0.45 ms |
| Runtime min / max | 14.9 / 17.7 ms |
| Speedup vs eager | **0.60×** ❌ |
| Effective TFLOPS | ~9.0 (~57% peak) |

## 結論
- 正確；cuBLAS 達 96% peak（這 shape 對 cuBLAS 特別友善），手寫差距可預期。
- 標記 `WIP`，繼續第 5 題。

## 命令紀錄
```bash
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=18 \
    kernel_src_path=finalProject_260531/solutions/level1/18_matmul_transposed_both.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
