#!/bin/bash
set -e

echo "============================================="
echo "  KernelBench 專案環境自動化佈署腳本 (Team 37)"
echo "============================================="

# 取得 repo 根目錄 (本腳本位於 <repo>/finalProject_260531/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 1. 載入國網中心 CUDA 模組
echo "[1/5] 載入國網中心 CUDA 12.8..."
module load cuda || echo "  (跳過：非 NCHC 環境，請自行確認 CUDA 可用)"

# 2. 檢查 Conda
if ! command -v conda &> /dev/null
then
    echo "[2/5] 未偵測到 Conda，開始下載並安裝 Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    $HOME/miniconda3/bin/conda init bash
    echo "⚠️ Miniconda 安裝完成！請執行 'source ~/.bashrc' 後重新跑本腳本。"
    exit 1
else
    echo "[2/5] Conda 已存在，跳過安裝。"
fi

# 3. 建立 / 更新 conda 環境 (含 PyTorch / Triton / KernelBench 評估相依)
echo "[3/5] 依 environment.yml 建立 / 更新 conda 環境 'kernelbench'..."
if conda env list | awk '{print $1}' | grep -qx "kernelbench"; then
    conda env update -n kernelbench -f "$SCRIPT_DIR/environment.yml" --prune
else
    conda env create -f "$SCRIPT_DIR/environment.yml"
fi

# 4. 把 kernelbench 套件 (位於 src/kernelbench) 以 editable 方式裝進環境
echo "[4/5] 安裝 KernelBench 本地套件 (pip install -e .)..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kernelbench
pip install -e "$REPO_ROOT" --no-deps

# 5. 提示
echo "[5/5] 完成。"
echo "============================================="
echo "🎉 環境建置完成！"
echo "請依序執行以下指令開始作業："
echo "1. module load cuda"
echo "2. conda activate kernelbench"
echo "3. python ./finalProject_260531/check.py"
echo "============================================="