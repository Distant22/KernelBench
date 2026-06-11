## 平行程式期末報告

> 相關文件：[PROMPT.md](PROMPT.md) (Agent 指令規範) ・ [PIPELINE.md](PIPELINE.md) (整體流程) ・ [tasks.txt](tasks.txt) (30 題清單)

### 1. 在根目錄執行腳本
```
./finalProject_260531/setup_env.sh
```

### 2. 按照 Script 來除錯 / 繼續執行其他指令
 - 成功的話會像這樣：有 `done` 而非錯誤訊息

```
done
#
# To activate this environment, use
#
#     $ conda activate kernelbench
#
# To deactivate an active environment, use
#
#     $ conda deactivate

=============================================
🎉 環境建置完成！
請依序執行以下指令開始作業：
1. module load cuda
2. conda activate kernelbench
3. python ./finalProject_260531/check.py
=============================================
```

### 3. 跑測試檔案確認環境正常
 - 成功的話輸入如下

```
=== CUDA 檢查 ===
PyTorch 是否能使用 CUDA: True
當前 GPU 裝置: Tesla V100-SXM2-32GB
CUDA 版本: 12.1
成功在 GPU 上建立 Tensor!

=== Triton 檢查 ===
Triton 版本: 3.1.0
成功匯入 Triton 模組!
```

---

## 4. 專題進度與實驗結果存放位置

```
finalProject_260531/
├── PROMPT.md            # Agent 指令規範 (含產出路徑與評估指令模板)
├── PIPELINE.md          # 整體 pipeline / mermaid 流程圖
├── tasks.txt            # 固定 30 題清單
├── check.py             # 環境健檢
├── setup_env.sh         # 一鍵安裝環境 + editable install
├── environment.yml      # conda 相依
├── solutions/           # ✅ 每題的 Triton / CUDA kernel 實作 (ModelNew)
│   ├── level1/
│   │   └── 06_matmul_large_k.py
│   ├── level2/
│   └── level3/
├── handwritten/         # ✅ 對 5 題原本 fallback 題目的「誠實手寫 (no-fallback)」嘗試與量測
│   ├── RESULTS_handwritten.md   # 誠實結果總表 (即使必輸也照實紀錄)
│   └── run_handwritten_eval.sh  # 逐題評估 driver
└── progress/            # ✅ 每題的實驗紀錄 (baseline / 迭代 / speedup / TODO)
    ├── level1/
    │   └── 06_matmul_large_k.md
    ├── level2/
    └── level3/
```

- **`solutions/`**：Agent 產出的 30 份 kernel 程式碼，檔名格式 `<NN>_<short_name>.py` (NN = KernelBench 題號)。
- **`progress/`**：Agent 每跑完一次 `run_and_check.py` 後在這裡更新對應 `<NN>_<short_name>.md`，記錄 baseline / runtime / speedup / 迭代版本 / 後續優化方向。
- **`handwritten/`**：對 Level-1 中原本以 cuDNN/SDPA fallback 取得 1.0× 的 5 題（P50/56/61/76/97），另外**強制手寫 from-scratch kernel 並如實量測**。結果誠實紀錄於 `RESULTS_handwritten.md`：P50 0.08×、P56 0.26×、P61 0.40×（皆 correct，決定性落後 cuDNN）；P76 因登入節點 20GB cgroup 上限無法評估；P97 手寫 flash-attn 在 Triton 3.1 無法編譯。**政策：即使必輸也誠實量測並紀錄，不以無聲 fallback 掩蓋。**
- 任何接手的 AI Agent 都應 **先翻 `progress/` 目錄判斷哪些題目已完成**，再從 `tasks.txt` 中找下一題未完成的進入。

## 5. 單題評估指令模板

```bash
python scripts/run_and_check.py \
    ref_origin=kernelbench level=<L> problem_id=<P> \
    kernel_src_path=finalProject_260531/solutions/level<L>/<NN>_<name>.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```

說明：
- `check_kernel=False`：KernelBench 的靜態檢查器只認 CUDA C++ 的 `__global__`，會把合法的 Triton kernel 誤判成「Missing __global__ kernel definition」。`torch.allclose` 正確性驗證仍照跑。
- `backend=triton`：讓 KernelBench 經 `load_custom_model_with_tempfile` 載入，避免 `@triton.jit` 因為 `exec()` 拿不到 source 而 `OSError: could not get source code`。
- 若改寫成純 CUDA C++ (帶 `__global__` 與 `load_inline`)，可移除這兩個旗標。

## 6. Nsight / Profiler 回饋

GPU profiling 僅可透過 Slurm compute node 執行：

```bash
sbatch finalProject_260531/profile_feedback.sbatch doctor
sbatch finalProject_260531/profile_feedback.sbatch profile \
  --level 2 --problem-id 40 \
  --solution finalProject_260531/solutions/level2/40_matmul_scale_residual.py \
  --paths candidate,eager,compile --deep
```

結果位於 `results/profiles/L<level>_P<id>/<run-id>/profile.json` 與
`feedback.md`。Slurm wrapper 會載入 cluster 的 NVIDIA HPC SDK module，以使用
Nsight Systems、Nsight Compute 與 V100 hardware counters。若工具或 counter
權限在其他環境不可用，pipeline 會明確記錄並 fallback 至 PyTorch Profiler。
完整操作、artifact 解讀、counter diagnosis 規則與已知陷阱請見
[PIPELINE.md](PIPELINE.md#9-nsight-guided-agent-feedback-pipeline)。
