# Level 1 / Problem 82 — Depthwise 2D Conv (3x3, stride=1, padding=0)

**Shape**: input (16, 64, 512, 512) FP32 ≈ 64 MB, output (16, 64, 510, 510)
**Compute**: ~2.4 GFLOPs (very light) ⇒ launch / bandwidth dominated
**Baseline**: cuDNN eager 5.03 ms, torch.compile 8.51 ms

## 結果（v3，已驗證 ✅）

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Custom kernel runtime | **3.84 ms** |
| **Speedup (vs eager / cuDNN)** | **1.31×** |
| **Speedup (vs torch.compile)** | **2.22×** |

## 演進歷史

| Iter | BLOCK_Y × BLOCK_X | num_warps | Runtime | Speedup vs eager |
|---|---|---|---|---|
| v1 | 8 × 64 | 4 | 8.08 ms | 0.62× |
| v2 | 16 × 128 | 4 | 5.21 ms | 0.97× |
| v3 | **8 × 256** | 4 | 4.32 ms | 1.16× |
| **v4** ⭐ | **4 × 512** | **4** | **3.84 ms** | **1.31×** |
| v5 | 2 × 1024 | 4 | 4.81 ms | 1.05× |
| v6 | 4 × 512 | 8 | 3.89 ms | 1.29× |

## 設計要點 (v4)

- Source: [solutions/level1/82_depthwise_conv2d.py](../../solutions/level1/82_depthwise_conv2d.py)
- Grid `(B*C=1024, H_out/4=128, W_out/512=1)`  → ~131k programs。
- 每個 program 算 `BLOCK_Y=4 × BLOCK_X=512 = 2048` 個輸出。
- Inner 3×3 用 `tl.static_range` 雙層展開，`tl.load` 9 次 input + 9 次 weight scalar。
- Weight 在 host 端 `.repeat(B, 1, 1)` 廣播到 (B*C, 3, 3)（36 KB，可忽略），讓 kernel 直接用 `pid_bc * stride_wc` 取對應 channel 權重，省一次 modulo 計算。

## 過程踩雷

### Bug 1 — Triton 不支援 nested function
v0 在 kernel 中用 `def load_xy(dy, dx)` helper，編譯器報錯 `nested function definition is not supported`。改用兩層 `tl.static_range` 展開即可。

### Bug 2 — weight 廣播
原本想用 `torch.as_strided` 把 weight (C,1,3,3) zero-stride 廣播到 (B*C,3,3)。實作起來易錯，且 .repeat 的 36 KB 拷貝對 5 ms kernel 是 noise，直接 `.repeat(B, 1, 1).contiguous()`。

## 後續優化想法
- BLOCK_X=512 已接近 W_out=510，可去除 column mask 提升一點；目前 mask 比較成本應已被 cache 吃掉。
- 共享 input row 重用：BLOCK_Y=4 時 row 0/1/2 與 row 1/2/3 各重複用 2 次；若用 shared memory cache 一塊 (BLOCK_Y+2) × BLOCK_X 的 input tile，可從 9× 讀降到 ~1.5×。預期還能再下降。
- CUTLASS / cudnn 的 depthwise 可能用 winograd-3x3，本實作為 direct conv，理論上 winograd 可省 ~36% 算力但 Volta FP32 場合常被 register pressure 抵消。

## 評估命令
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=82 \
    kernel_src_path=finalProject_260531/solutions/level1/82_depthwise_conv2d.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```
