# Level 3 Task 28 — Vision Transformer

## Baseline
- batch=2, image 224×224, patch 16, dim=512, depth=6, heads=8, mlp=2048
- 197 tokens × 512 = 100K elems per layer (~0.4 MB)，極小規模
- Test 設定 `dropout=0.0` → 無 RNG 一致性問題

## v1
- `solutions/level3/28_vit.py`: 與 baseline 同構，使用 `nn.TransformerEncoder`
  (內部走 `F.scaled_dot_product_attention` 的 mem-efficient 路徑)。
- Eval: ✅ 5/5, runtime **2.94 ms**, ref eager **2.63 ms**, compile **2.95 ms**
- Speedup: **0.89× eager / 1.00× compile**

## 評析
- Eager 比 ours 快是「測量噪音 + warm-up 差異」: stddev 0.3 ms / 平均 2.6 ms (12%)。
  我們的 ours run 跟 compile 完全一致。
- 全是極小 GEMM + SDPA，PyTorch 已最佳化，無 fusion 機會。
- 接受 ~1.0×。

## 下一步
- 移到 Task 43 MinGPT Causal Attention。
