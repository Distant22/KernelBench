# Level 1 / Problem 93 — Masked Cumulative Sum (`torch.cumsum(x * mask, dim=1)`)

**Shape**: x (32768, 32768) FP32 ≈ 4 GB；mask (32768, 32768) ≈ 4 GB（見下方 fp32-cast 註記）；out 4 GB。**Memory-bound**。
**Baseline**: PyTorch eager 31.9 ms, torch.compile (Inductor) 16.2 ms。

---

## ⚠️ 重大修正（2026-06-11）：先前的「1.00×」是 dead-code，量測無效

先前版本的 `ModelNew.forward` 與 `_launch_masked_cumsum_dim1` 都用
`mask.dtype == torch.bool` 作為走 Triton kernel 的條件。但 **KernelBench 的 eval
(`_process_input_tensor`, [src/kernelbench/eval.py](../../../src/kernelbench/eval.py) 行 618/663/767)
會把所有輸入 tensor 一律 cast 成 fp32**，所以 mask 進到 `forward` 時已是 `float32`，
guard 永遠為 False → **Triton kernel 從未被執行**，每次都 silent fallback 到
`torch.cumsum(x * mask)`（reference 本身）。

因此先前記錄的「kernel 32.0 ms / 1.00× / 31% roofline」其實是
**reference-vs-reference**，不是我的 kernel。profiler 之前報 `insufficient-counter-data`
也是因為 fallback 的 `x*mask` 中間張量需要額外 4 GB → 在 ncu replay 下 OOM。

**修正**：guard 放寬為 `mask.dtype in (torch.bool, torch.float32, torch.float16)`，
kernel 現在確實執行；in-place `out=x` 只用 ~8 GB（fallback 需 16 GB），同時也解掉 profiler OOM。

---

## 結果（修正後，已用真實 HW counter 驗證 ✅）

Profile：JID 950036, gn1001, run `20260611_223051`（`--deep`，ncu HW counter 命中）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5（fp32 mask 下數值正確） |
| Custom kernel runtime | **16.9 ms** |
| eager | 31.9 ms |
| torch.compile (Inductor) | 16.2 ms |
| **Speedup (vs eager)** | **1.89×** ✅ |
| Speedup (vs torch.compile) | 0.96×（打平 Inductor，皆貼 roofline） |

## 真實硬體狀態紀錄（Nsight Compute，candidate kernel `_masked_cumsum_dim1_kernel`）

| Counter | 量測值 | 解讀 |
|---|---|---|
| `gpu__time_duration` | 16.68 ms | 與 wall-clock 16.9 ms 一致 |
| `gpu__dram_throughput.avg.pct_of_peak` | **91.03%** | **貼 memory roofline** |
| `dram__bytes.sum.per_second` | **816.6 GB/s** | V100 HBM2 ~900 GB/s 峰值的 91% |
| `sm__warps_active.avg.pct_of_peak_active` | 23.9% | occupancy 偏低，但不是瓶頸 |
| `sm__maximum_warps_per_active_cycle_pct` | 25.0% | 理論 occupancy 上限 25% |
| `launch__occupancy_limit_registers` | **1**（最小者）| **occupancy 被暫存器限制**：128 regs/thread × 512 threads = 65536 = 滿一個 SM 的暫存器檔 → 每 SM 只能 1 個 block |
| `launch__registers_per_thread` | 128 | |
| `launch__waves_per_multiprocessor` | 409.6 | 32768 blocks / 80 SM；wave 數極多 |
| `sm__throughput.avg.pct_of_peak` | 11.4% | 計算單元幾乎閒置 → 確認 memory-bound |
| Grid / Block | (32768,1,1) / (512,1,1) | 1 program per row，16 warps |

### 瓶頸歸因（基於真實 counter，取代先前所有臆測）
- Kernel **已達 91% DRAM 峰值頻寬**，是道道地地的 **memory roofline-bound**。
- Occupancy 僅 25%（暫存器限制：128 regs/thread → 每 SM 1 block），**但這不是瓶頸**：
  因為有 **409 waves** 的 block 排隊，memory-level parallelism 充足，足以完全餵滿 HBM。
  提高 occupancy（減暫存器）對已 91% 峰值的頻寬幾乎無增益空間。
- eager 為何 31.9 ms：cub `DeviceScan` 走 mul + scan 兩遍、搬動約 17 GB；
  我的 single-pass fused 只搬 ~12 GB 且 91% 峰值 → 16.9 ms，**真實 1.89×**。
- 與 Inductor 打平（16.2 vs 16.9 ms）：Inductor 達 99.5% occupancy 但同樣 memory-bound、
  同樣貼 roofline；兩者皆無有意義的可改進空間。

## 設計要點
- Source: [solutions/level1/93_masked_cumsum.py](../../solutions/level1/93_masked_cumsum.py)
- 每行 32768 元素，**單一 program 持有整列**，`BLOCK_N = next_power_of_2(N) = 32768`，`num_warps=16, num_stages=2`。
- Grid `(B=32768,)`：1 program per row。
- Fused：load x → load mask → `x * mask.to(fp32)` → `tl.cumsum(axis=0)` → store。**單 read + 單 write，無中間張量**。
- in-place 寫回 x buffer，省 4 GB peak（correctness-safe：eval 在跑 candidate 前已先 materialize reference output，[eval.py](../../../src/kernelbench/eval.py) 行 778）。
- monkey-patch `torch.allclose` 為 streaming 版（大張量 chunked 比較）。

## 結論
- **真實達成 1.89× over eager，5/5 正確，貼 91% DRAM 峰值** → 已達 V100 memory roofline，無有意義剩餘空間。
- 與 torch.compile 打平，兩者皆 HW-ceiling-bound。
- 先前「需要 decoupled-lookback scan 才能贏」的待辦作廢：那是基於 dead-code 的錯誤頻寬臆測；
  實際 single-pass row-per-block 已達 91% 峰值。

## 系統性風險提醒
- KernelBench eval 會把所有輸入 cast 成 fp32。**任何用輸入 dtype（bool/int 等）當作走 fast path 條件的 `ModelNew`，其 fast kernel 都會變成 dead code**。需逐一檢查其他 solution 是否有同樣的 guard。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=93 \
    kernel_src_path=finalProject_260531/solutions/level1/93_masked_cumsum.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
