#!/usr/bin/env bash
# Team 37 — honest evaluation of the 5 HAND-WRITTEN fallback problems.
#
# These 5 Level-1 problems were originally shipped as library "fallbacks"
# (cuDNN / SDPA) in solutions/level1/. The handwritten/ versions are genuine
# from-scratch Triton kernels. This script measures each one honestly against
# the same PyTorch eager / torch.compile baselines, ONE problem per subprocess
# (so CUDA / monkey-patch state cannot leak and a single OOM cannot poison the
# rest of the run).
#
# Run via tmux so an ssh disconnect cannot kill it:
#   tmux new -s hw_eval -d 'bash finalProject_260531/handwritten/run_handwritten_eval.sh'
#   tmux attach -t hw_eval     # to watch
set -u

cd "$(dirname "$0")/../.." || exit 1   # -> repo root (KernelBench/)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG_DIR="finalProject_260531/handwritten"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_DIR}/results_hw_${STAMP}.log"

# level problem_id file
TASKS=(
  "1 50 50_conv2d_alexnet_hw.py"
  "1 56 56_conv2d_asymmetric_hw.py"
  "1 61 61_conv_transposed_3d_hw.py"
  "1 76 76_conv1d_dilated_hw.py"
  "1 97 97_sdpa_hw.py"
)

echo "=== Hand-written fallback eval — ${STAMP} ===" | tee "$SUMMARY"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee -a "$SUMMARY"

for t in "${TASKS[@]}"; do
  read -r LEVEL PID FILE <<< "$t"
  echo "" | tee -a "$SUMMARY"
  echo "######################################################################" | tee -a "$SUMMARY"
  echo ">>> L${LEVEL}/P${PID}  ${FILE}  ($(date +%H:%M:%S))" | tee -a "$SUMMARY"
  echo "######################################################################" | tee -a "$SUMMARY"

  PER_LOG="${LOG_DIR}/eval_${PID}_${STAMP}.log"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/run_and_check.py \
      ref_origin=kernelbench level="$LEVEL" problem_id="$PID" \
      kernel_src_path="${LOG_DIR}/${FILE}" \
      eval_mode=local gpu_arch='["Volta"]' \
      check_kernel=False backend=triton \
      > "$PER_LOG" 2>&1
  RC=$?

  # Surface the key correctness / speedup lines into the summary.
  grep -iE "correct|speedup|runtime|ms|error|OOM|out of memory|Traceback" "$PER_LOG" \
      | tail -n 25 | tee -a "$SUMMARY"
  echo "[exit ${RC}] full log: ${PER_LOG}" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== DONE  ($(date +%H:%M:%S)) ===" | tee -a "$SUMMARY"
echo "Summary written to: $SUMMARY"
