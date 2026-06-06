# Team 37 — 期末報告 (Final Report)

> **專題類別：B 類別 — 以 AI Agent 分析與優化平行程式**
> （非 A 類別自訂題目；本專題以 KernelBench 既有題庫為標的，重點在 LLM Agent 對
> 平行 GPU kernel 的分析與優化能力評測。）

本目錄為期末報告論文與其全部原始資料。

## 目錄結構
```
report/
├── main.tex                     # 論文主體 (IEEEtran conference 格式)
├── refs.bib                     # 參考文獻 (亦內嵌於 main.tex 的 thebibliography)
├── README.md                    # 本檔
├── generate_tables.py           # 由評估 JSON 自動生成下列 tables/*.tex
├── tables/
│   ├── macros.tex               # 頭條數字巨集 (geomean / fast_p ...)
│   ├── aggregate.tex            # 總指標 + fast_p 表 (Table I)
│   └── per_problem.tex          # 30 題逐題表 (Table II)
└── raw_data/                    # ← 報告要求附上的原始資料
    ├── export_conversation.py   # 從 Copilot session-store 匯出逐字稿的腳本
    ├── agent_conversation_log.md# 完整對話 (input prompt + output)，3 sessions
    └── agent_changes.md         # 對 Agent / 評估流程所做的所有更改
```

## 如何編譯論文
本機目前未安裝 LaTeX 工具鏈。建議二擇一：

1. **Overleaf（最簡單）**：新建專案，上傳 `main.tex` 與整個 `tables/` 目錄，
   編譯器選 pdfLaTeX。IEEEtran 類別 Overleaf 內建，無需額外安裝。
2. **本機 TeX Live**：
   ```bash
   tlmgr install ieeetran   # 若無 IEEEtran
   cd finalProject_260531/report
   python generate_tables.py   # 需先有 ../results/eval_all_v100.json
   pdflatex main.tex && pdflatex main.tex
   ```

> 課程 template 日後提供 (TBA) 後，將 `main.tex` 的 `\documentclass` 換成指定
> template（IEEE 或 ACM `acmart`）並把各 `\section` 內容貼入即可；數字表格
> (`tables/*.tex`) 與 raw data 可直接沿用。

## 數字來源（單一事實來源, single source of truth）
- 論文中所有頭條數字、`fast_p` 表、逐題表，皆由
  `finalProject_260531/results/eval_all_v100.json` 經 `generate_tables.py`
  自動生成，**論文不會與量測數據不一致**。
- 重新量測：在 repo 根目錄、kernelbench 環境下執行
  `python finalProject_260531/run_eval_all.py`，再 `python report/generate_tables.py`。

## 報告要求對應表
| 報告要求 | 對應產出 |
|----------|----------|
| 格式 IEEE / ACM | `main.tex`（IEEEtran；可換 `acmart`） |
| input prompt | `raw_data/agent_conversation_log.md`（使用者訊息）+ `../tasks.txt` |
| system prompt | `../PROMPT.md`（角色/系統提示全文） |
| output | `raw_data/agent_conversation_log.md`（Agent 回應）+ `../solutions/`（30 份 kernel） |
| 對 agent 做的更改 | `raw_data/agent_changes.md` |
| 專題類別標註 | 本檔頂部 + `main.tex` 標題區與 Introduction |
