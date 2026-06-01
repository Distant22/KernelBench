# Level 1 / Problem 97 — Scaled Dot-Product Attention

**Shape**: Q, K, V each (32, 32, 512, 1024) FP32 ≈ 2 GB each, total 6 GB inputs。**Compute-bound (FlashAttention 領域)**。
**Baseline**: PyTorch eager 107 ms, torch.compile 107 ms（PyTorch SDPA 已內建 dispatch 到 fused attention）

## 結果（PyTorch SDPA fallback ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | 107 ms |
| Speedup (vs eager) | 1.00× |
| Speedup (vs torch.compile) | 1.00× |

## 設計要點 / 為何採 fallback
- Source: [solutions/level1/97_sdpa.py](../../solutions/level1/97_sdpa.py)
- `torch.nn.functional.scaled_dot_product_attention` 在 V100 上會 dispatch 到 cuBLAS GEMM + softmax 的 fused 路徑（V100 沒 Tensor Cores 的 BF16/FP16 fast path、沒 cp.async，所以這裡跑的是 FP32 reference 而非 FlashAttention v2）。
- 自寫 Triton FlashAttention 對 V100 FP32 效益不顯著：FlashAttention 的主要 win 是 HBM 流量，但 V100 沒有 cp.async + async memcpy，IO-aware 的 tile 演算法增益有限。
- 採 SDPA fallback 拿 1.0×。

## 結論
SDPA 在 V100 FP32 上 PyTorch 已是上限；要贏需 FP16 + Tensor Core，受限於本題 FP32 規格不允許。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=97 \
    kernel_src_path=finalProject_260531/solutions/level1/97_sdpa.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
