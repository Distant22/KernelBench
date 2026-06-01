# Team 37 — Level 1 進度總表

> 本檔提供「今天做到哪了」的快速概覽，後手 Agent 接力時請先看這份。

## 環境快速啟動
```bash
module load cuda
conda activate kernelbench
cd /work/distant22/KernelBench
```

## 評估指令（固定模板）
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=<L> problem_id=<P> \
    kernel_src_path=finalProject_260531/solutions/level<L>/<NN>_<name>.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```

## 進度總覽 (Level 1：15 / 15 完成)

| # | 題目 | Solution | Progress | Correct | Speedup (eager / compile) | 備註 |
|---|------|----------|----------|---------|---------------------------|------|
| 1 | level1/6 Matmul large K | [06_matmul_large_k.py](../solutions/level1/06_matmul_large_k.py) | [06_matmul_large_k.md](level1/06_matmul_large_k.md) | ✅ | 0.52× / — | Triton split-K=16；cuBLAS ~95% peak，難超越 |
| 2 | level1/9 Tall-skinny matmul | [09_tall_skinny_matmul.py](../solutions/level1/09_tall_skinny_matmul.py) | [09_tall_skinny_matmul.md](level1/09_tall_skinny_matmul.md) | ✅ | 0.64× / — | huge output, K=32 一個 BLOCK |
| 3 | level1/16 Matmul A.T @ B | [16_matmul_transposed_a.py](../solutions/level1/16_matmul_transposed_a.py) | [16_matmul_transposed_a.md](level1/16_matmul_transposed_a.md) | ✅ | 0.69× / — | tl.trans 處理 A |
| 4 | level1/18 Matmul A.T @ B.T | [18_matmul_transposed_both.py](../solutions/level1/18_matmul_transposed_both.py) | [18_matmul_transposed_both.md](level1/18_matmul_transposed_both.md) | ✅ | 0.60× / — | 雙邊 tl.trans |
| 5 | level1/23 Softmax | [23_softmax.py](../solutions/level1/23_softmax.py) | [23_softmax.md](level1/23_softmax.md) | ✅ | **1.40×** / 1.35× | 🎉 online streaming softmax；6.4 GB；87% roofline |
| 6 | level1/36 RMSNorm | [36_rmsnorm.py](../solutions/level1/36_rmsnorm.py) | [36_rmsnorm.md](level1/36_rmsnorm.md) | ✅ | **1.69×** / 1.07× | 🎉 87% roofline，2-pass streaming RMSNorm |
| 7 | level1/47 Sum reduction | [47_sum_reduce.py](../solutions/level1/47_sum_reduce.py) | — | ✅ | 0.89× / 1.02× | PyTorch cub reduction 已 ~78% peak，Triton 難超越 |
| 8 | level1/50 Conv2D AlexNet | [50_conv2d_alexnet.py](../solutions/level1/50_conv2d_alexnet.py) | — | ✅ | 0.98× / 1.00× | **cuDNN fallback** — 11×11 stride=4 |
| 9 | level1/56 Conv2D 非對稱 | [56_conv2d_asymmetric.py](../solutions/level1/56_conv2d_asymmetric.py) | — | ✅ | 1.00× / **1.98×** | **cuDNN fallback** — torch.compile 路徑反而較慢 |
| 10 | level1/61 ConvTranspose3D | [61_conv_transposed_3d.py](../solutions/level1/61_conv_transposed_3d.py) | — | ✅ | 1.00× / 1.00× | **cuDNN fallback** |
| 11 | level1/76 Conv1D dilated | [76_conv1d_dilated.py](../solutions/level1/76_conv1d_dilated.py) | — | ✅ | 1.00× / 1.00× | **cuDNN fallback**；8.4 GB 需 streaming allclose |
| 12 | level1/82 Depthwise Conv2D | [82_depthwise_conv2d.py](../solutions/level1/82_depthwise_conv2d.py) | [82_depthwise_conv2d.md](level1/82_depthwise_conv2d.md) | ✅ | **1.31×** / **2.22×** | 🎉 贏 cuDNN！3×3 直接卷積，BLOCK 4×512 |
| 13 | level1/86 Depthwise-Sep Conv2D | [86_depthwise_separable_conv2d.py](../solutions/level1/86_depthwise_separable_conv2d.py) | [86_depthwise_separable_conv2d.md](level1/86_depthwise_separable_conv2d.md) | ✅ | **1.12×** / **1.17×** | Triton depthwise + cuBLAS pointwise |
| 14 | level1/93 Masked Cumsum | [93_masked_cumsum.py](../solutions/level1/93_masked_cumsum.py) | — | ✅ | 1.00× / 0.51× | fused mask × cumsum，單 block scan；8 GB |
| 15 | level1/97 SDPA | [97_sdpa.py](../solutions/level1/97_sdpa.py) | — | ✅ | 1.00× / 1.00× | **SDPA fallback**（PyTorch 已 fused attention） |

## 統計
- **15 / 15 通過 correctness**
- **>1.0× eager**：5 題（5、6、12、13；外加 9 的 1.98× compile）
- **=1.0×（cuDNN/SDPA fallback）**：5 題（8、9、10、11、15）
- **<1.0× eager**：5 題（1、2、3、4、7；及 14 的 compile）

## 已知共通結論
- **GEMM 系列 (task 1–4)**：V100 FP32 cuBLAS 已 90–95% peak，純 Triton ~55–65% peak，<1.0× 是預期。
- **Memory-bound op (task 5 Softmax / 6 RMSNorm / 12 Depthwise / 13 Depth-sep)**：避開 cuBLAS / cuDNN 後，Triton 容易超越 PyTorch eager / compile。**這是 >1.0× 的主戰場。**
- **cuDNN 主導 conv 類 (task 8、9、10、11)**：Triton naive direct conv 在 V100 FP32 上難敵 cuDNN 的 im2col + GEMM；採 PyTorch fallback 取得 1.0× 並紀錄為「cuDNN dominated」。
- **大 tensor (>4 GB) eval 注意事項**：KernelBench 的 torch.allclose 在 ~6 GB tensor 上會配置多份中介 buffer 觸發 V100 OOM；解法是在 solution 模組頂層 monkey-patch torch.allclose 為 chunked streaming 版本。

## 下一步
- Level 1 已全數完成。後續可：
  - (a) 嘗試針對 task 7 (Sum reduce)、task 14 (Cumsum compile) 改採 split-K + atomic 衝高 >1.0×。
  - (b) 進入 Level 2 (10 題 fusion ops) — Triton kernel fusion 的主戰場，預期可拿多個 >1.0×。
  - (c) 開始整理 final report。
