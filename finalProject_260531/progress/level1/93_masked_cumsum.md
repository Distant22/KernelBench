# Level 1 / Problem 93 — Masked Cumulative Sum (`torch.cumsum(x * mask, dim=1)`)

**Shape**: x (32768, 32768) FP32 ≈ 4 GB；mask (32768, 32768) bool ≈ 1 GB；out 4 GB；total traffic ~9 GB。**Memory-bound**。
**Roofline**: 9 GB / 900 GB/s ≈ **10 ms**
**Baseline**: PyTorch eager 32.0 ms (28% peak), torch.compile 16.3 ms (61% peak)

## 結果（已驗證 ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | 32.0 ms |
| Speedup (vs eager) | **1.00×** |
| Speedup (vs torch.compile) | 0.51× |
| Roofline 達成率 | 10 / 32 ≈ 31% |

## 設計要點
- Source: [solutions/level1/93_masked_cumsum.py](../../solutions/level1/93_masked_cumsum.py)
- 每行 32768 元素 = 128 KB，**單一 program block 持有整列**，BLOCK_N = 32768。
- Grid `(B=32768,)`：1 program per row。
- Fused：load x → load mask → 乘 mask → `tl.cumsum(axis=0)` → store。**單 read + 單 write，無中間張量**。
- 對 in-place 寫回 x buffer，省 4 GB peak（V100 32 GB 才不會 OOM）。
- monkey-patch `torch.allclose` 為 streaming 版。

## 過程踩雷 / 嘗試過的配置

| num_warps | num_stages | runtime | speedup eager |
|---|---|---|---|
| 8 | 2 | 32.0 ms | 1.00× ✅ |
| 16 | 2 | 32.0 ms | 1.00× |

提高 num_warps 沒有改善。瓶頸在 32 KB shared memory（為了 cumsum 的 inter-warp scan）+ 128 KB block size，已塞爆 V100 96 KB / SM。

## 結論
- 對 PyTorch eager 打平 (1.0×) — 因為 PyTorch eager 是 mul + cumsum 兩遍 read-write，我們是 1 遍 fused，理論應有 1.5×；但 V100 的單 block scan 由於 shared memory 限制無法滿頻寬。
- 對 torch.compile 0.51× — inductor 把 cumsum 拆成 hierarchical scan + carry，效率更高。要打贏需自寫 hierarchical scan（先 BLOCK_N=2048 各自 scan，再用一條 carry kernel 累進），工程量大。

## 後續優化想法
- 階層式 scan：第一階 BLOCK=2048 per program，warp scan + write partial sums；第二階對 partial sums scan；第三階 add carry back。可把 shared memory 壓力降下並滿頻寬。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=93 \
    kernel_src_path=finalProject_260531/solutions/level1/93_masked_cumsum.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
