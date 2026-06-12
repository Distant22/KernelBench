#!/bin/bash
#==============================================================================
# Team 37 — SLURM batch script for the fixed 30-problem V100 evaluation.
#
# TWCC Taiwania 2 rule: DO NOT run compute directly on the login node.
# Always submit through SLURM, e.g.:
#     sbatch finalProject_260531/run_eval_all.sbatch
# Check status / logs:
#     squeue -u $USER
#     tail -f kb_eval-<jobid>.out
#==============================================================================
#SBATCH -A ACD115083            # Project ID
#SBATCH -J kb_eval              # Job name
#SBATCH -p gp2d                 # Partition (gtest=quick 30min test, gp1d/gp2d/gp4d)
#SBATCH -N 1                    # 1 node
#SBATCH -n 1                    # 1 task
#SBATCH -c 4                    # 4 CPUs for the task
#SBATCH --gres=gpu:1            # 1 V100 GPU
#SBATCH -t 12:00:00             # walltime limit
#SBATCH -o kb_eval-%j.out       # stdout  (%j = job id)
#SBATCH -e kb_eval-%j.err       # stderr

set -euo pipefail

# --- provenance: record which compute node / GPU actually ran the job --------
echo "=== SLURM job $SLURM_JOB_ID on $(hostname) ==="
echo "Partition : ${SLURM_JOB_PARTITION:-?}"
echo "Node list : ${SLURM_JOB_NODELIST:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "============================================="

# --- environment -------------------------------------------------------------
module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kernelbench

cd /work/distant22/KernelBench

# --- run the full 30-problem evaluation --------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python finalProject_260531/run_eval_all.py

echo "=== ALL_DONE (job $SLURM_JOB_ID) ==="
