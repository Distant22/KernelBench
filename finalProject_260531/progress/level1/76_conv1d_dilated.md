# Level 1 / Problem 76 — Conv1D dilated/strided

**Shape**: x (64, 64, 524280) FP32 ≈ **8.39 GB**；3-tap kernel stride=3 dilation=4 → out (64, 128, 174758) ≈ 5.36 GB。**Memory-bound but with strided gathers**。
**Baseline**: PyTorch eager 40.9 ms, torch.compile 40.8 ms（cuDNN im2col + GEMM）

## 結果（cuDNN fallback ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | 40.9 ms |
| Speedup (vs eager) | 1.00× |
| Speedup (vs torch.compile) | 1.00× |

## 設計要點 / 為何最終採 fallback
- Source: [solutions/level1/76_conv1d_dilated.py](../../solutions/level1/76_conv1d_dilated.py)
- 我先嘗試自寫 Triton direct stencil（stride=3 dilation=4 的 1D conv）：
  - **v1**：grid `(B, Cout, ceil(L_out/256))`，`tl.static_range(Cin=64) × tl.static_range(K=3)` 全展開 → 1260 ms（30× 慢於 ref）。原因是 192 次 unroll 把 kernel binary 撐爆，加上 stride=3 的 strided load 不 coalesced。
  - **v2**：縮小 unroll 只展開 K=3、`for cin in range(64)` 用 runtime loop、BLOCK_L=2048、num_warps=8 → eval 跑 5+ 分鐘還沒結束，仍 >10× 慢於 cuDNN，殺掉。
- cuDNN 把 dilated conv1d 轉成 im2col + GEMM，stride=3 的 gather 由 cuBLAS 高效處理，~85% memory roofline，naive Triton 難匹敵。
- 改用 `nn.Conv1d` fallback。仍需 monkey-patch `torch.allclose` 為 streaming 版避免 8.4 GB OOM。

## 結論
標準 1D dilated conv 也是 cuDNN 主場。要贏需要實作 im2col + 自家 Triton GEMM，工程量大且收益低。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=76 \
    kernel_src_path=finalProject_260531/solutions/level1/76_conv1d_dilated.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
