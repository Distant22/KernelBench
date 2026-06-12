#!/usr/bin/env bash
#SBATCH --account=ACD115083
#SBATCH --partition=gtest
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=kb-profile
#SBATCH --output=finalProject_260531/results/profiles/slurm-%j.out

set -euo pipefail

ROOT="${KERNELBENCH_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT}" || ! -f "${ROOT}/finalProject_260531/profile_feedback.py" ]]; then
  echo "Submit from the KernelBench repository root or set KERNELBENCH_ROOT." >&2
  exit 1
fi
cd "${ROOT}"

# TWCC exposes Nsight through the NVIDIA HPC SDK module. Prefer repository-local
# tools when present, otherwise load the cluster-provided CLI tools.
if [[ ! -x "${ROOT}/.tools/nsight/bin/ncu" || ! -x "${ROOT}/.tools/nsight/bin/nsys" ]]; then
  if type module >/dev/null 2>&1; then
    module load "${NSIGHT_MODULE:-nvhpc-24.11_hpcx-2.20_cuda-12.6}"
  else
    echo "Warning: environment-modules is unavailable; Nsight discovery may fail." >&2
  fi
fi

export PATH="${ROOT}/.tools/nsight/bin:${PATH}"
export XDG_CACHE_HOME="${ROOT}/.compute-cache"
export TRITON_CACHE_DIR="${ROOT}/.compute-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${ROOT}/.compute-cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resolve the python interpreter. Prefer an explicit override, then the
# repository-local environment, then a standard conda "kernelbench" env so the
# wrapper works across both teammates' setups.
PYTHON=""
for candidate in \
  "${KB_PYTHON:-}" \
  "${ROOT}/.conda-envs/kernelbench/bin/python" \
  "${HOME}/miniconda3/envs/kernelbench/bin/python" \
  "${HOME}/anaconda3/envs/kernelbench/bin/python"; do
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    PYTHON="${candidate}"
    break
  fi
done
if [[ -z "${PYTHON}" ]] && command -v conda >/dev/null 2>&1; then
  PYTHON="$(conda run -n kernelbench bash -c 'command -v python' 2>/dev/null || true)"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "Could not locate a kernelbench python. Set KB_PYTHON=/path/to/python." >&2
  exit 1
fi

exec "${PYTHON}" finalProject_260531/profile_feedback.py "$@"
