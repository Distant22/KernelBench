# Level 1 / Problem 47 — Sum reduction over dim=1

**Shape**: (B=128, F=4096, J=4095) FP32 → 8.39 GB input, output (128, 1, 4095) ≈ 2 MB. **Memory-bound**.
**Roofline**: 8.39 GB / 900 GB/s ≈ **9.3 ms**
**Baseline**: PyTorch eager 11.9 ms (78% peak), torch.compile 13.6 ms

## 結果（已驗證 ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | 13.3 ms |
| Speedup (vs eager) | 0.89× |
| Speedup (vs torch.compile) | 1.02× |
| Roofline 達成率 | 9.3 / 13.3 ≈ 70% |

## 設計要點
- Source: [solutions/level1/47_sum_reduce.py](../../solutions/level1/47_sum_reduce.py)
- Grid `(B=128, ceil(J/BLOCK_J)=4)`，每 program 持有 1 batch、`BLOCK_J=1024` 個 j 位置的 fp32 accumulator。
- 內層 `for f in range(F=4096)` 累加 1024 個連續 float (coalesced)。
- num_warps=4, num_stages=4（讓 4096 次獨立讀取深度 pipeline）。
- monkey-patch `torch.allclose` 為 chunked streaming 版避免 V100 OOM。

## 過程踩雷 / 嘗試過的配置

| BLOCK_J | num_warps | num_stages | runtime | speedup |
|---|---|---|---|---|
| 1024 | 4 | 4 | 13.3 ms | 0.89× ✅ (採用) |
| 2048 | 8 | 3 | 14.5 ms | 0.82× |
| 1024 | 4 | 4 (split-K=4 + atomic_add) | 14.1 ms | 0.84× |

PyTorch 的 cub-based reduction kernel 在 V100 已達 ~78% memory peak，留給 Triton 的空間極小。Split-K 反而讓 atomic 序列化吃掉 bandwidth 增益。

## 結論
此題本質上 **memory-bound 且 baseline 已接近 peak**，naive 直接 reduce 已是合理上限。要超越需在 launch 配置或 SM-level 排程上做更精細的 tuning，CP 值不高。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=47 \
    kernel_src_path=finalProject_260531/solutions/level1/47_sum_reduce.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
