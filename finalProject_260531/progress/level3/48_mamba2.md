# Level 3 Task 48 — Mamba2 SSD ReturnY

## Baseline
- B=2048, T=128, H=8, hd=64, n=16, block_len=64, c=2 chunks/seq
- 4 個 einsum (chunked SSD)；無 RNG
- 數值極不穩定：`A/B/C` 為 randn parameter，`exp(cumsum)` 後可達 1e22

## v1 (BMM 重寫，FAIL)
- 把 4 個 einsum 重寫成 `permute + matmul` 形式
- 單元測試每個 einsum 都 0 / fp32-eps 級誤差
- **Eval FAIL**：完整 pipeline 因 contraction order 不同造成累積誤差 ~1e14 (rel ~1e-8)
  超出 KernelBench `atol/rtol = 1e-4` 容差
- 教訓：含 `exp(cumsum(randn))` 的算子重排計算順序高風險

## v2 (verbatim einsum + micro-optim, PASS)
- 維持 baseline einsum 表達式，PyTorch 內部選穩定的 contraction order
- micro-optim：
  - `torch.exp_` / `masked_fill_` in-place，省 ~24 MB 中介寫回
  - `view` + `permute` 取代 `einops.rearrange`，省 Python overhead
- Eval: ✅ 5/5, runtime **24.0 ms**, ref eager **25.3 ms**, compile **16.5 ms**
- Speedup: **1.05× eager / 0.69× compile**

## 評析
- Eager 端贏 1.05× 主要來自 in-place exp/mask 與 einops 開銷消除。
- `torch.compile` (inductor) 把 4 個 einsum 重新規劃為更高效的 GEMM 序列，
  我們純 eager-style 無法及。屬「compile-only win」題型。
