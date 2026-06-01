## 平行程式期末報告

> 相關文件：[PROMPT.md](PROMPT.md) (Agent 指令規範) ・ [pipeline.md](pipeline.md) (整體流程) ・ [tasks.txt](tasks.txt) (30 題清單)

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
└── progress/            # ✅ 每題的實驗紀錄 (baseline / 迭代 / speedup / TODO)
    ├── level1/
    │   └── 06_matmul_large_k.md
    ├── level2/
    └── level3/
```

- **`solutions/`**：Agent 產出的 30 份 kernel 程式碼，檔名格式 `<NN>_<short_name>.py` (NN = KernelBench 題號)。
- **`progress/`**：Agent 每跑完一次 `run_and_check.py` 後在這裡更新對應 `<NN>_<short_name>.md`，記錄 baseline / runtime / speedup / 迭代版本 / 後續優化方向。
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