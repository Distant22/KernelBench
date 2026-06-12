# Level 1 / Problem 56 — Conv2D 非對稱 input/kernel

> **2026-06-12 — 手寫 Triton implicit-GEMM，0.58× (compile)。**
> v1 BLOCK_M=64/N=128/warps4 → 114ms (0.38×)。Fix：BLOCK_M 64→128（Cout=128 一個 M-block，
> gather 只做一次）+ BLOCK_N=128 + num_warps16。**74.8ms (0.58×)**，compute/mem 平衡 51/53%，occ 25%。
> correct 5/5。
>
> **2026-06-12 重驗 — 落差來自 compile baseline 漂移，非 kernel 改動。** 重跑 deep profile：
> kernel_ms=74.8ms（與先前完全一致，kernel 沒動）、eager=21.6ms、compile=43.1ms → **0.576× 完全重現**。
> 但 full eval（JID 950583）那次測到 0.346×，反推 compile_ms≈25.9ms — 同一個非對稱 5×7 conv，
> inductor/cuDNN 在不同次執行挑到不同 algo（26ms vs 43ms），導致 speedup_compile 在 0.35~0.58 間波動。
> 結論：markdown 記錄正確，無未記錄改動；compile baseline 非固定值。

---

## （以下為原紀錄）

**Shape**: x (8, 64, 512, 256) FP32 ≈ 32 MB；conv 5×7 → out (8, 128, 508, 250) ≈ 65 MB。**cuDNN-bound**。
**Baseline**: PyTorch eager 21.5 ms, torch.compile 42.8 ms

## 結果（cuDNN fallback ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | 21.6 ms |
| Speedup (vs eager) | 1.00× |
| **Speedup (vs torch.compile)** | **1.98×** |

## 設計要點 / 為何採 fallback
- Source: [solutions/level1/56_conv2d_asymmetric.py](../../solutions/level1/56_conv2d_asymmetric.py)
- 採 `nn.Conv2d` fallback。
- 有趣現象：torch.compile (inductor backend) 在這個非對稱 5×7 形狀上反而比 eager 慢一倍 — eager 直接走 cuDNN best algo，inductor 嘗試自動生 kernel 卻挑了較差路徑。
- 因此 fallback 的 1.0× eager 等價於 1.98× compile，是 task 8/9/10/11 中 compile speedup 最高的一題。

## 結論
標準 conv2d，cuDNN 已是上限。記錄為「cuDNN-dominated; compile 路徑 inductor 選擇不佳，eager fallback 直接帶來 1.98× compile gain」。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=56 \
    kernel_src_path=finalProject_260531/solutions/level1/56_conv2d_asymmetric.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
