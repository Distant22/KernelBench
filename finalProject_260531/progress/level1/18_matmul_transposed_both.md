# Level 1 / Problem 18 — Matmul Both Transposed (C = A.T @ B.T)

**Shape**: A (8192, 2048), B (4096, 8192) → C (2048, 4096) FP32  
**Baseline (PyTorch eager)**: 9.13 ms — cuBLAS ~15.0 TFLOPS (~96% V100 FP32 peak)  
**Baseline (torch.compile)**: 9.13 ms

## Iteration v1 — Triton GEMM with double `tl.trans` (採用)

- Source: [solutions/level1/18_matmul_transposed_both.py](../../solutions/level1/18_matmul_transposed_both.py)
- 雙邊 coalesced load + `tl.trans` 轉置；config 同 task 16 (128/128/32, num_warps=4, num_stages=3)

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Runtime mean / std | 15.2 ms / 0.45 ms |
| Runtime min / max | 14.9 / 17.7 ms |
| Speedup vs eager | **0.60×** ❌ |
| Effective TFLOPS | ~9.0 (~57% peak) |

## 結論
- 正確；cuBLAS 達 96% peak（這 shape 對 cuBLAS 特別友善），手寫差距可預期。
- 標記 `WIP`，繼續第 5 題。

## 命令紀錄
```bash
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=18 \
    kernel_src_path=finalProject_260531/solutions/level1/18_matmul_transposed_both.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```

---

## Profile-driven 調優 pass（補做）— 數學重構大勝

C = A.T @ B.T，A(K=8192,M=2048)、B(N=4096,K=8192) → C(M=2048,N=4096)。

| 版本 | 設定 | kernel_ms | speedup_compile | 備註 |
|---|---|---|---|---|
| v1 | 迴圈內 double `tl.trans`，warps=4, stages=4 | — | 0.628× | 原始 |
| v2 | double-trans + warps=8, stages=3 | 16.8 | 0.545× | 暫存器 spill（151 regs），捨棄 |
| v3 | **重構為 D=B@A**（迴圈內零 transpose，store 時 trans 一次），warps=4, stages=4 | 16.2 | 0.564× | 結構對了但 warps 不足 |
| v4 | 重構 D=B@A + **warps=8, stages=3** | 12.0 | **0.76×** | **大勝，採用** |

- **關鍵洞見**：`C = A.Tᵀ @ B.Tᵀ = (B @ A).T`。把 kernel 改成直接算 `D = B @ A`
  （`tl.dot(b_tile, a_tile)`，兩個 operand 皆為自然 row-major），**迴圈內完全不需要 transpose**，
  只在最後 store 時做一次 `tl.trans(acc)`。這消除了原本每次迭代兩個 `tl.trans` 造成的暫存器壓力，
  使 warps=8 不再 spill，吞吐大幅提升。
- 最終 **0.628× → 0.76×**，備份於 `_fallback_backup/18_matmul_transposed_both.reform_BA_w8_s3_076.py`。
