# Team 37 — V100 批次評估正式結果 (run_eval_all.py)

- 硬體：NVIDIA Tesla V100-SXM2-32GB (Volta, CC 7.0)
- 精度：FP32；correctness 5 trials，perf 100 trials
- 每題獨立子行程；總耗時 30.9 min
- baseline 計時檔：`results/timing/V100_SXM2_32GB_NCHC/`

## 總指標

- compiled：**30/30**
- correct：**30/30**
- geomean speedup (correct) vs eager：**0.761x**
- geomean speedup (correct) vs compile：**0.732x**

| metric | p=1.0 | p=1.5 | p=2.0 | p=3.0 |
|---|---|---|---|---|
| fast_p vs **eager** | 19/30 (63%) | 4/30 (13%) | 0/30 (0%) | 0/30 (0%) |
| fast_p vs **compile** | 13/30 (43%) | 2/30 (7%) | 1/30 (3%) | 0/30 (0%) |

## 逐題結果

| Lv | PID | solution | correct | kernel(ms) | eager(ms) | compile(ms) | sp_eager | sp_compile |
|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 06_matmul_large_k.py | ✅ | 6.430 | 4.680 | 4.900 | 0.728x | 0.762x |
| 1 | 9 | 09_tall_skinny_matmul.py | ✅ | 7.310 | 6.460 | 6.420 | 0.884x | 0.878x |
| 1 | 16 | 16_matmul_transposed_a.py | ✅ | 11.800 | 9.660 | 9.670 | 0.819x | 0.819x |
| 1 | 18 | 18_matmul_transposed_both.py | ✅ | 11.800 | 9.150 | 9.190 | 0.775x | 0.779x |
| 1 | 23 | 23_softmax.py | ✅ | 24.700 | 34.700 | 33.200 | 1.405x | 1.344x |
| 1 | 36 | 36_rmsnorm.py | ✅ | 28.700 | 48.400 | 30.500 | 1.686x | 1.063x |
| 1 | 47 | 47_sum_reduce.py | ✅ | 12.500 | 11.900 | 13.600 | 0.952x | 1.088x |
| 1 | 50 | 50_conv2d_alexnet.py | ✅ | 15.900 | 8.110 | 8.700 | 0.510x | 0.547x |
| 1 | 56 | 56_conv2d_asymmetric.py | ✅ | 74.800 | 21.700 | 25.900 | 0.290x | 0.346x |
| 1 | 61 | 61_conv_transposed_3d.py | ✅ | 77.300 | 31.400 | 31.400 | 0.406x | 0.406x |
| 1 | 76 | 76_conv1d_dilated.py | ✅ | 70.800 | 45.600 | 45.100 | 0.644x | 0.637x |
| 1 | 82 | 82_depthwise_conv2d.py | ✅ | 3.840 | 5.030 | 8.530 | 1.310x | 2.221x |
| 1 | 86 | 86_depthwise_separable_conv2d.py | ✅ | 9.510 | 11.100 | 16.500 | 1.167x | 1.735x |
| 1 | 93 | 93_masked_cumsum.py | ✅ | 16.800 | 31.400 | 16.300 | 1.869x | 0.970x |
| 1 | 97 | 97_sdpa.py | ✅ | 739.000 | 110.000 | 110.000 | 0.149x | 0.149x |
| 2 | 1 | 01_conv2d_relu_biasadd.py | ✅ | 22.600 | 24.300 | 24.700 | 1.075x | 1.093x |
| 2 | 12 | 12_gemm_mul_leakyrelu.py | ✅ | 9.880 | 9.990 | 9.530 | 1.011x | 0.965x |
| 2 | 21 | 21_conv_add_scale_sigmoid_gn.py | ✅ | 12.400 | 18.000 | 14.200 | 1.452x | 1.145x |
| 2 | 22 | 22_matmul_clamp_lse_mish.py | ✅ | 9.850 | 10.300 | 9.520 | 1.046x | 0.966x |
| 2 | 40 | 40_matmul_scale_residual.py | ✅ | 38.600 | 40.800 | 38.000 | 1.057x | 0.984x |
| 2 | 45 | 45_gemm_sigmoid_lse.py | ✅ | 29.700 | 30.100 | 29.000 | 1.013x | 0.976x |
| 2 | 56 | 56_matmul_sigmoid_sum.py | ✅ | 20.200 | 20.200 | 20.200 | 1.000x | 1.000x |
| 2 | 66 | 66_matmul_dropout_softmax.py | ✅ | 5.160 | 5.290 | 5.110 | 1.025x | 0.990x |
| 2 | 88 | 88_gemm_gn_swish_mul_swish.py | ✅ | 9.890 | 10.700 | 9.940 | 1.082x | 1.005x |
| 2 | 99 | 99_matmul_gelu_softmax.py | ✅ | 9.940 | 9.990 | 9.810 | 1.005x | 0.987x |
| 3 | 1 | 01_mlp.py | ✅ | 12.200 | 13.100 | 12.600 | 1.074x | 1.033x |
| 3 | 28 | 28_vit.py | ✅ | 1020.000 | 2.630 | 3.190 | 0.003x | 0.003x |
| 3 | 43 | 43_mingpt_causal_attention.py | ✅ | 29.200 | 43.900 | 35.200 | 1.503x | 1.205x |
| 3 | 44 | 44_minigpt_block.py | ✅ | 78.100 | 108.000 | 82.500 | 1.383x | 1.056x |
| 3 | 48 | 48_mamba2.py | ✅ | 16.000 | 25.300 | 16.500 | 1.581x | 1.031x |
