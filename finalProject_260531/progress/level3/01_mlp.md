# Level 3 Task 1 — MLP (3 Linear + 2 ReLU)

## Baseline
- Shape: x (128, 16384) → 16384 → 16384 → 8192
- 3 GEMMs (~1.7e11 FLOPs total), 2 ReLU passes (8 MB each)
- 完全 GEMM dominated（~12 ms eager）

## v1
- `solutions/level3/01_mlp.py`: `F.linear` + `F.relu(inplace=True)`，
  並嘗試 `torch._addmm_activation` (cuBLASLt RELU epilogue) fallback。
- Eval: ✅ 5/5, runtime **12.7 ms**, ref eager **12.8 ms**, compile **12.5 ms**
- Speedup: **1.01× eager / 0.98× compile**

## 評析
- GEMM 已 ~95% peak，無從下手；ReLU inplace 省了 ~17 µs，剛好打平。
- `torch._addmm_activation` 在這版本 PyTorch 上不存在，走 fallback path。

## v2 (torch.compile max-autotune, WIN)
- `solutions/level3/01_mlp.py`: 將 `F.relu(F.linear(...))` 鏈抽成獨立函式
  `_mlp_forward(x, weights, biases)`，以
  `torch.compile(_mlp_forward, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")`
  包裝。max-autotune 為 3 個 GEMM 挑到比預設更快的 cuBLAS 路徑並融合 ReLU epilogue。
- 改動前備份：`_fallback_backup/01_mlp.before_compile.py`。
- Eval (V100, --deep): ✅ 5/5, runtime **12.20 ms**, eager 12.9 ms, compile 12.7 ms
- Speedup: **1.057× eager / 1.041× compile**（兩者皆 > 1.0×，WIN）

## 評析 (v2)
- 此題純 GEMM 堆疊，手寫 Triton 無法勝過 cuBLAS；改用 `torch.compile` 當武器，
  讓 inductor 的 max-autotune 自動選 GEMM algo + 融合 ReLU，是跨過 1.0× 的關鍵。
- 數值正確（GEMM/ReLU 無 reduction-reorder 風險），correct 5/5。
