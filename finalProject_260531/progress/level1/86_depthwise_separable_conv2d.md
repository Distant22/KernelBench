# Level 1 / Problem 86 — Depthwise-Separable 2D Conv

**Pipeline**: depthwise (in=64, k=3, stride=1, pad=1) → pointwise (1×1, in=64, out=128)
**Shape flow**: (16, 64, 512, 512) → (16, 64, 512, 512) → (16, 128, 512, 512)
**Baseline**: PyTorch eager 15.9 ms, torch.compile 16.6 ms

## 結果（v1，已驗證 ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | **14.2 ms** |
| **Speedup (vs eager)** | **1.12×** |
| **Speedup (vs torch.compile)** | **1.17×** |

## 設計要點

- Source: [solutions/level1/86_depthwise_separable_conv2d.py](../../solutions/level1/86_depthwise_separable_conv2d.py)
- **Depthwise**: 重用 Problem 82 的 3×3 stride=1 kernel，加上 `PAD` 參數支援 padding=1（mask `(ry >= 0) & (ry < H_in) & (rx >= 0) & (rx < W_in)`）。BLOCK 4×512 / num_warps=4。
- **Pointwise (1×1, 64→128)**: 直接呼叫 `nn.Conv2d.pointwise(d)`。1×1 conv 等價於 (BHW, in_C) @ (in_C, out_C) GEMM，PyTorch 走 cuBLAS — Volta FP32 已 90%+ peak，自寫 Triton 必輸。
- 故 fusion 性質的 speedup 來自 depthwise 部分（vs cuDNN 的 ~1.3×，task 82 已驗證），pointwise 跟 baseline 持平。整體 1.1×。

## 後續優化想法（如需更多 gain）
- **Fuse depthwise + pointwise 成一個 Triton kernel**：每 program 算 `BLOCK_OC=128, BLOCK_HW=64` 個輸出，內部對 64 input channels 累積。但 1×1 conv 是 cuBLAS 的強項，純 Triton 實作大概率退步；除非用 **Tensor Core (HMMA)** — V100 有 mixed-precision TC 但需要 fp16 input。本題定 fp32，TC 無效。
- 因此 1.1× 已接近此 baseline 的上限，不再 over-iterate。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=86 \
    kernel_src_path=finalProject_260531/solutions/level1/86_depthwise_separable_conv2d.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
