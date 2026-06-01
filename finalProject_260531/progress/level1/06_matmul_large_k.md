# Level 1 / Problem 6 — Matmul with large K dimension

**Baseline (PyTorch eager)**: 4.63 ms (cuBLAS GEMM)  
**Baseline (torch.compile)**: 4.89 ms

## Iteration v1 — Triton split-K (BLOCK 64×64×32, SPLIT_K=16)

- Source: [solutions/level1/06_matmul_large_k.py](../../solutions/level1/06_matmul_large_k.py)
- Build flags: `num_warps=4`, `num_stages=3`, `allow_tf32=False`
- Grid: `(4, 4, 16)` = 256 blocks (≈3.2× SM count, OK occupancy)

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Runtime mean / std | 8.82 ms / 0.125 ms |
| Runtime min / max | 8.52 / 8.97 ms |
| Speedup vs eager | **0.52×** ❌ |
| Speedup vs torch.compile | 0.55× |
| Effective TFLOPS | ~7.8 (V100 FP32 peak ≈ 15.7) |

### 結論
- 正確性過關，但被 cuBLAS 完全壓制（cuBLAS 已達 ~95% peak）。
- 在 V100 FP32 下純手寫 GEMM 想超越 cuBLAS 極困難；split-K factor 16、64×64 tile 只達 50% peak。
- 下一輪迭代方向：
  1. 放大 tile 到 `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32`，配合 `SPLIT_K=20` 拿到 80 blocks 對齊 V100 80 SMs。
  2. 嘗試 `num_warps=8`、`num_stages=4` 增加 ILP 與 software pipelining。
  3. 改寫 reduction 為「private buffer + 一次 reduce kernel」避免 atomic add 競爭。

### 命令紀錄
```bash
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=6 \
    kernel_src_path=finalProject_260531/solutions/level1/06_matmul_large_k.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```

**狀態**：正確但未達加速目標，標記為 `WIP`，待之後若有空檔回頭優化。先繼續第 2 題以維持進度。
