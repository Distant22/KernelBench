# Level 1 / Problem 16 — Matmul with Transposed A (C = A.T @ B)

**Shape**: A (8192, 2048) → A.T (2048, 8192) @ B (8192, 4096) → C (2048, 4096) FP32  
**Baseline (PyTorch eager)**: 9.65 ms — cuBLAS ~14.2 TFLOPS (~90% V100 FP32 peak)  
**Baseline (torch.compile)**: 9.93 ms  
**Roofline下界 (compute-bound)**: 137 GFLOPs / 15.7 TFLOPS ≈ 8.7 ms

## Iteration v1 — Triton GEMM with `tl.trans` 處理 A.T (採用)

- Source: [solutions/level1/16_matmul_transposed_a.py](../../solutions/level1/16_matmul_transposed_a.py)
- 載入 A tile `(BLOCK_K, BLOCK_M)`（沿 M 連續 → coalesced）→ `tl.trans` 後丟 `tl.dot`
- Tile：`BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8`，num_warps=4, num_stages=3
- Grid：`16 × 32 = 512` blocks (V100 80 SMs → 6.4 blocks/SM)
- shmem ≈ 32 KB/block → 3 blocks/SM 可駐留

| Metric | Value |
|---|---|
| Compiled | ✅ |
| Correctness | ✅ 5/5 |
| Runtime mean / std | 14.0 ms / 0.66 ms |
| Runtime min / max | 13.4 / 16.4 ms |
| Speedup vs eager | **0.69×** ❌ |
| Effective TFLOPS | ~9.8 (~62% V100 peak) |

## 結論
- 正確；cuBLAS 對 FP32 transposed GEMM 已調到 90% peak，純 Triton 落差 30% 屬正常。
- 標記 `WIP`，後續可嘗試更大 tile (但 V100 shmem 96 KB 限制 + register pressure 不好調)。

## 命令紀錄
```bash
python scripts/run_and_check.py \
    ref_origin=kernelbench level=1 problem_id=16 \
    kernel_src_path=finalProject_260531/solutions/level1/16_matmul_transposed_a.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```

---

## Profile-driven 調優 pass（補做）

C = A.T @ B，A(8192,2048)、B(8192,4096) → C(2048,4096)。

| 版本 | 設定 | kernel_ms | speedup_compile | 備註 |
|---|---|---|---|---|
| v1 | BLOCK 128×128, BLOCK_K=32, GROUP_M=8, warps=4, stages=4 | — | 0.785× | 起點 |
| v2 | 同上但 **warps=8, stages=3** | 12.0 | **0.805×** | 136 regs、SM 75%、占用率 12.5%，**採用** |

- **關鍵洞見**：此題只需單一 `tl.trans(a)`（A 為 (K,M)）。warps=8 增加 ILP，SM 利用率達 75%。
  嘗試過 reformulation 但 A.T@B 結構上一定要一次 transpose，無法消除。
- 最終 **0.785× → 0.805×**，屬計算受限（SM 75%），接近純 Triton FP32 對上 cuBLAS 的結構天花板。
