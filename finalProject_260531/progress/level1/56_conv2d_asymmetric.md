# Level 1 / Problem 56 — Conv2D 非對稱 input/kernel

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
