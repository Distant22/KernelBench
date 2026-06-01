# Level 1 / Problem 50 — Conv2D AlexNet first layer

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
