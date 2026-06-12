# Level 1 / Problem 9 — Tall-skinny Matmul (huge output, tiny K)

**Shape**: A (32768, 32) @ B (32, 32768) → C (32768, 32768) FP32  
**Baseline (PyTorch eager)**: 6.22 – 6.30 ms  
**Baseline (torch.compile)**: 6.24 – 6.26 ms  
**Roofline下界 (4 GB write @ 900 GB/s)**: ≈4.4 ms  
**cuBLAS 估計 BW 利用率**: ~71%

## Iteration v1 — BLOCK 128×128×32, num_warps=4, GROUP_M=8 (採用)

- Source: [solutions/level1/09_tall_skinny_matmul.py](../../solutions/level1/09_tall_skinny_matmul.py)
- Grid: `(256 × 256, ) = 65,536 blocks`，super-grouped 排序提升 L2 reuse
- K=32 完全裝進一個 BLOCK_K → 無 K-loop，每 block 一次 `tl.dot` 完成
- shmem ≈ 32 KB / block → 3 blocks/SM 可駐留

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Runtime mean / std | 9.76 ms / 19.4 (含首輪 autotune outlier 203 ms) |
| Runtime min / max | 7.39 / 203 ms |
| Speedup vs eager | **0.64×** ❌ |
| Speedup vs torch.compile | 0.64× |
| Effective write BW (steady-state min) | ~540 GB/s (≈60% peak) |

## Iteration v2 — BLOCK 128×256, num_warps=8 (回退)

放大 N tile 想增加 store burst，但 shmem 48 KB → occupancy 降為 1 block/SM。

| Metric | Value |
|---|---|
| Correctness | ✅ |
| Runtime mean / min | 10.4 / 8.17 ms |
| Speedup vs eager | 0.61× ❌（更差）|

→ 已回退到 v1 設定。

## 結論
- 正確性 OK；達 cuBLAS 約 64% 速度。
- V100 FP32 GEMM 的 cuBLAS 實作已極接近 roofline（71% peak BW），手寫 Triton 想反超非常困難。
- 標記為 `WIP`，先繼續第 3 題。

## 後續可嘗試
1. 純 CUDA C++ 配 `__ldg` + `float4` vectorised store。
2. `num_stages=3` + pre-fetch A／B 到 register（K=32 全展開）。
3. 試 `BLOCK_M=64, BLOCK_N=128, num_warps=4` 看會不會降低 register pressure。

## 命令紀錄
```bash
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=9 \
    kernel_src_path=finalProject_260531/solutions/level1/09_tall_skinny_matmul.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```

---

## Profile-driven 調優 pass（補做）

A(32768,32)@B(32,32768) → C(32768,32768)，4GB 輸出、store-bound。

| 版本 | 設定 | kernel_ms | speedup_compile | 備註 |
|---|---|---|---|---|
| v1 | BLOCK_M=128, BLOCK_N=128, warps=4 | 7.83 | 0.812× | 255 regs spill，占用率 12.4% |
| v2 | **BLOCK_M=64, BLOCK_N=128, warps=8** | 7.32 | **0.88×** | 88 regs、占用率 24.3%、dram 64%，**採用** |

- **關鍵洞見**：輸出極大、K=32 極小 → 此題是 store-bound。縮小 BLOCK_M 到 64 並提高 warps=8
  讓暫存器壓力從 255（spill）降到 88，占用率翻倍至 24.3%，更能掩蓋寫出 4GB 的延遲。
- 最終 **0.812× → 0.88×**，已接近記憶體頻寬天花板（dram 64% 利用率）。

### 第二輪掃描（2026-06-12，確認天花板）

| 版本 | 設定 | kernel_ms | speedup_compile | 備註 |
|---|---|---|---|---|
| v3 | 同 v2 但移除靜態恆真 mask（M/N/K 皆整除 block）| 7.30 | 0.877× | 與 v2 持平、程式更乾淨，**採用** |
| v4 | BLOCK_M=32, warps=4（衝占用率）| 7.75 | 0.826× | 占用率未升（仍 24%，88 regs），變慢，捨棄 |

- **誠實結論**：`compile_ms = eager_ms = 6.4ms` → `torch.compile` 此題直接呼叫 cuBLAS 專用的 tall-skinny `volta_sgemm`。ncu 顯示 v3 sm__throughput 68–75%、占用率卡在 24%（88 regs，register-limited）。已掃過 tile / warps / mask / 占用率所有槓桿，純 Triton FP32 在 V100 上**無法超越 cuBLAS**。0.88× 為此 shape 的結構性天花板，誠實保留手寫 kernel。
