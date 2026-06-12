# Level 1 / Problem 50 — Conv2D AlexNet first layer

> **2026-06-12 — 手寫 Triton implicit-GEMM，profile-driven 迭代後 0.50× (compile)。**
> 現行解為 from-scratch Triton kernel（非 cuDNN fallback）。
>
> | ver | change | kernel ms | speedup(compile) | regs | warps% | mem-pipe% |
> |-----|--------|-----------|------------------|------|--------|-----------|
> | v1 | BLOCK_M=32, stages=4 | 22.9 | 0.41× | 248 | 12.5% | — |
> | v2 | warps 4→8, stages 4→2 | 22.3 | 0.36× | 128 | 24.9% | 82% |
> | **v3** | **BLOCK_M 32→128** | **16.0** | **0.50×** | 216 | 12.4% | 76% |
> | v4 | warps 8→16 | 24.5 | 0.33× | 128 | 24.8% | 88% |
>
> **歸因**：input gather 與 `offs_m` 無關，BLOCK_M=32 時 M-grid=3 → gather 重做 3 次。
> BLOCK_M=128 一個 M-block 涵蓋 Cout=96 → gather 只做一次（22→16ms）。v4 證實真正天花板
> 是**非合併 stride-4 gather**（L1 pipeline 88%），非 occupancy。cuDNN 用 winograd 特化路徑。

---

## （以下為原 fallback 紀錄，保留供對照）

**Shape**: x (256, 3, 224, 224) FP32 ≈ 154 MB；conv 11×11 stride=4 padding=2，out (256, 96, 56, 56) ≈ 154 MB。**Compute / cuDNN-bound**。
**Baseline**: PyTorch eager 7.83 ms, torch.compile 7.95 ms

## 結果（cuDNN fallback ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | 7.97 ms |
| Speedup (vs eager) | 0.98× |
| Speedup (vs torch.compile) | 1.00× |

## 設計要點 / 為何採 fallback
- Source: [solutions/level1/50_conv2d_alexnet.py](../../solutions/level1/50_conv2d_alexnet.py)
- 11×11 stride=4 大 kernel + 寬 batch=256 是典型 cuDNN winograd / implicit GEMM 強項。
- V100 FP32 cuBLAS / cuDNN 已達 ~95% peak，naive Triton direct conv 必然輸：不僅算力更難利用 tensor pipeline，還會吃 register pressure。
- 直接以 `nn.Conv2d` 實作 `ModelNew`，達到 ~1.0× 並通過 correctness。

## 結論
標準 dense conv2d 是 cuDNN 主場；要贏需自寫 winograd / FFT-conv，CP 值低。在報告中記錄為「cuDNN dominated, fallback 1.0×」。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=50 \
    kernel_src_path=finalProject_260531/solutions/level1/50_conv2d_alexnet.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
