# Level 3 Task 43 — minGPT Causal Self-Attention

## Baseline
- B=128, T=512, dim=768, heads=8, head_dim=96
- 物化 (B,H,T,T)=128·8·512·512·4 B = 1 GB attention matrix
- attn_pdrop=resid_pdrop=0 → 無 RNG 議題

## v1
- `solutions/level3/43_mingpt_causal_attention.py`
- 把 manual `Q@K.T → mask → softmax → @V` 換成 `F.scaled_dot_product_attention(is_causal=True)`，
  V100 上走 mem-efficient SDPA。
- QKV split + transpose 用 `permute(2,0,3,1,4)` 一次到位。
- Eval: ✅ 5/5, runtime **28.8 ms**, ref eager **43.6 ms**, compile **35.1 ms**
- Speedup: **🎉 1.51× eager / 1.22× compile**

## 評析
- 主要省 1 GB attention matrix 的 read/write 與融合 mask+softmax，
  SDPA 內部 tile 後完全留在 SRAM。
- 連 `torch.compile` 都跑不出 SDPA 路徑（仍 35 ms），所以贏 1.22×。

## 下一步
- 移到 Task 44 MinGPT Block（會復用此 attention pattern）。
