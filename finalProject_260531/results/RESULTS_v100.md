# Team 37 — V100 批次評估正式結果 (run_eval_all.py)

- 硬體：NVIDIA Tesla V100-SXM2-32GB (Volta, CC 7.0)
- 精度：FP32；correctness 5 trials，perf 100 trials
- 每題獨立子行程；總耗時 29.7 min
- baseline 計時檔：`results/timing/V100_SXM2_32GB_NCHC/`

## 總指標

- compiled：**30/30**
- correct：**30/30**
- geomean speedup (correct) vs eager：**1.026x**
- geomean speedup (correct) vs compile：**0.984x**

| metric | p=1.0 | p=1.5 | p=2.0 | p=3.0 |
|---|---|---|---|---|
| fast_p vs **eager** | 21/30 (70%) | 2/30 (7%) | 0/30 (0%) | 0/30 (0%) |
| fast_p vs **compile** | 17/30 (57%) | 2/30 (7%) | 1/30 (3%) | 0/30 (0%) |

## 逐題結果

| Lv | PID | solution | correct | kernel(ms) | eager(ms) | compile(ms) | sp_eager | sp_compile |
|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 06_matmul_large_k.py | ✅ | 8.820 | 4.660 | 4.650 | 0.528x | 0.527x |
| 1 | 9 | 09_tall_skinny_matmul.py | ✅ | 7.820 | 6.370 | 6.350 | 0.815x | 0.812x |
| 1 | 16 | 16_matmul_transposed_a.py | ✅ | 14.000 | 9.650 | 9.660 | 0.689x | 0.690x |
| 1 | 18 | 18_matmul_transposed_both.py | ✅ | 15.100 | 9.130 | 9.100 | 0.605x | 0.603x |
| 1 | 23 | 23_softmax.py | ✅ | 24.700 | 34.700 | 33.300 | 1.405x | 1.348x |
| 1 | 36 | 36_rmsnorm.py | ✅ | 28.700 | 48.400 | 30.500 | 1.686x | 1.063x |
| 1 | 47 | 47_sum_reduce.py | ✅ | 13.300 | 11.900 | 13.600 | 0.895x | 1.023x |
| 1 | 50 | 50_conv2d_alexnet.py | ✅ | 7.840 | 7.830 | 7.950 | 0.999x | 1.014x |
| 1 | 56 | 56_conv2d_asymmetric.py | ✅ | 21.600 | 21.500 | 42.800 | 0.995x | 1.981x |
| 1 | 61 | 61_conv_transposed_3d.py | ✅ | 27.800 | 27.900 | 27.900 | 1.004x | 1.004x |
| 1 | 76 | 76_conv1d_dilated.py | ✅ | 40.800 | 40.800 | 40.900 | 1.000x | 1.002x |
| 1 | 82 | 82_depthwise_conv2d.py | ✅ | 3.840 | 5.030 | 8.520 | 1.310x | 2.219x |
| 1 | 86 | 86_depthwise_separable_conv2d.py | ✅ | 14.200 | 16.000 | 16.600 | 1.127x | 1.169x |
| 1 | 93 | 93_masked_cumsum.py | ✅ | 32.100 | 31.900 | 16.200 | 0.994x | 0.505x |
| 1 | 97 | 97_sdpa.py | ✅ | 109.000 | 109.000 | 109.000 | 1.000x | 1.000x |
| 2 | 1 | 01_conv2d_relu_biasadd.py | ✅ | 22.600 | 24.400 | 24.600 | 1.080x | 1.088x |
| 2 | 12 | 12_gemm_mul_leakyrelu.py | ✅ | 9.800 | 9.890 | 9.550 | 1.009x | 0.974x |
| 2 | 21 | 21_conv_add_scale_sigmoid_gn.py | ✅ | 12.400 | 18.000 | 12.700 | 1.452x | 1.024x |
| 2 | 22 | 22_matmul_clamp_lse_mish.py | ✅ | 9.760 | 10.200 | 9.770 | 1.045x | 1.001x |
| 2 | 40 | 40_matmul_scale_residual.py | ✅ | 38.900 | 40.300 | 37.700 | 1.036x | 0.969x |
| 2 | 45 | 45_gemm_sigmoid_lse.py | ✅ | 29.500 | 29.900 | 28.800 | 1.014x | 0.976x |
| 2 | 56 | 56_matmul_sigmoid_sum.py | ✅ | 20.000 | 20.100 | 20.000 | 1.005x | 1.000x |
| 2 | 66 | 66_matmul_dropout_softmax.py | ✅ | 5.000 | 5.040 | 4.980 | 1.008x | 0.996x |
| 2 | 88 | 88_gemm_gn_swish_mul_swish.py | ✅ | 10.500 | 10.600 | 9.880 | 1.010x | 0.941x |
| 2 | 99 | 99_matmul_gelu_softmax.py | ✅ | 9.870 | 9.900 | 9.580 | 1.003x | 0.971x |
| 3 | 1 | 01_mlp.py | ✅ | 12.800 | 12.800 | 12.600 | 1.000x | 0.984x |
| 3 | 28 | 28_vit.py | ✅ | 2.690 | 2.630 | 2.960 | 0.978x | 1.100x |
| 3 | 43 | 43_mingpt_causal_attention.py | ✅ | 28.900 | 43.800 | 35.100 | 1.516x | 1.215x |
| 3 | 44 | 44_minigpt_block.py | ✅ | 77.200 | 107.000 | 81.800 | 1.386x | 1.060x |
| 3 | 48 | 48_mamba2.py | ✅ | 23.900 | 25.300 | 16.500 | 1.059x | 0.690x |
