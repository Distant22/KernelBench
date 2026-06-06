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

## 下一步
- 不再優化此題，移到 Task 27 ViT。
