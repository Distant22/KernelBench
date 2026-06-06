# Level 3 Task 44 — minGPT Block (LN + CausalAttn + LN + MLP)

## Baseline
- 同 task 43 形狀 (B=128, T=512, dim=768, heads=8) + 兩個 LayerNorm + MLP (4×)。
- attn/resid pdrop = 0 → 無 RNG 議題。

## v1
- `solutions/level3/44_minigpt_block.py`
- Attention 改用 `F.scaled_dot_product_attention(is_causal=True)`，
  GELU 改用 `F.gelu(approximate='tanh')` (省掉 manual `tanh + pow + mul` 數個 elementwise pass)。
- 其餘 (LN/MLP/residual) 維持 PyTorch 內建。
- Eval: ✅ 5/5, runtime **77.2 ms**, ref eager **107.0 ms**, compile **81.8 ms**
- Speedup: **🎉 1.39× eager / 1.06× compile**

## 評析
- Attention 部分跟 task 43 一樣享 SDPA causal 的紅利 (~15 ms 省掉)。
- Tanh-fused GELU 還能再壓 ~3 ms。
- 連 `torch.compile` 都被超過 1.06×，因為 inductor 沒選擇 SDPA 路徑。

## 下一步
- 移到最後一題 Task 48 Mamba2ReturnY。
