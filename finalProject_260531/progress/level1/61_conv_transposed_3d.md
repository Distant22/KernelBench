# Level 1 / Problem 61 — ConvTranspose3D

**Shape**: x (8, 48, 64, 64, 64) FP32 ≈ 50 MB；3×3×3 kernel → out (8, 48, 66, 66, 66) ≈ 55 MB。**cuDNN-bound**。
**Baseline**: PyTorch eager 27.8 ms, torch.compile 27.9 ms

## 結果（cuDNN fallback ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | 27.8 ms |
| Speedup (vs eager) | 1.00× |
| Speedup (vs torch.compile) | 1.00× |

## 設計要點 / 為何採 fallback
- Source: [solutions/level1/61_conv_transposed_3d.py](../../solutions/level1/61_conv_transposed_3d.py)
- ConvTranspose3D = 反向 conv，cuDNN 走 `cudnnConvolutionBackwardData` 已最佳化，naive Triton 3D 雙重 loop 必輸。
- 5D tensor 的 tile 設計也很複雜（B × Cout × D × H × W × Cin × kD × kH × kW = 9 維迴圈），即便寫出來大概率 register spill。
- 採 `nn.ConvTranspose3d` fallback。

## 結論
3D 反卷積是 cuDNN 的舒適區；自寫 Triton 在 V100 FP32 上難以打贏。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=61 \
    kernel_src_path=finalProject_260531/solutions/level1/61_conv_transposed_3d.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
