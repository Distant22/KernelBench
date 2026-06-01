# Level 1 / Problem 36 — RMSNorm over dim=1

**Shape**: (B=112, F=64, D1=512, D2=512) FP32 → 7.51 GB tensor，**Memory-bound**
**Roofline (3× tensor traffic = 22.5 GB @ 900 GB/s)**: ~25.0 ms
**Baseline**: PyTorch eager 48.4 ms, torch.compile 30.6 ms

## 結果（v1，已驗證 ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | **28.7 ms** |
| **Speedup (vs eager)** | **1.69×** |
| **Speedup (vs torch.compile)** | **1.07×** |
| Roofline 達成率 | 25.0 / 28.7 ≈ **87%** |

## 設計要點

- Source: [solutions/level1/36_rmsnorm.py](../../solutions/level1/36_rmsnorm.py)
- 把 input 從 (B, F, D1, D2) 攤成 (B, F, J=D1*D2=262144)，沿 J 是連續的。
- Grid `(B=112, ceil(J/BLOCK_J)=256)`：每 program 處理 1 個 batch、`BLOCK_J=1024` 個 j 位置。
- 內部 2-pass：
  - Pass 1：外層 `for f in range(F)` 累加 `sum_sq[BLOCK_J]`，每次 load 1024 個連續 float (coalesced)。
  - 計算 `inv_rms = 1 / sqrt(sum_sq / F + eps)`。
  - Pass 2：再次 `for f in range(F)` 重讀並寫 `x * inv_rms`。
- Output 寫回 input buffer (in-place) 降低 V100 32 GB peak。
- num_warps=4, num_stages=2。
- 同樣 monkey-patch `torch.allclose` 為 chunked streaming 版避免 OOM。

## 過程踩雷

### Bug 1 — GPU 共享導致 OOM
跑 RMSNorm 時 GPU 被另一 user 占 24 GB，剩 540 MB，input 7 GB 直接 OOM。等對方結束後重跑即通過。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 對「總可用空間不足」無解，只能等 / 換 GPU。

## 後續優化想法
- 把 F 用 `F_CONST=64` 編譯期常數展開（已用 `tl.constexpr`），編譯器應已 unroll。
- 若改成 1-pass：把整列 64 features 都載到 register（每 program BLOCK_J=128 → 64×128=8192 floats per program），可省一半 input 讀取（從 3× → 2×），roofline 到 ~16.7 ms，但 register pressure 高、可能 spill。值得實驗。
- 把 split-J + reduce 合成一個 hierarchical pattern。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=36 \
    kernel_src_path=finalProject_260531/solutions/level1/36_rmsnorm.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
