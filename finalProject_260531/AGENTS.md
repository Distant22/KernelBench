# Team 37 Final Project Agent Guide

This file applies to everything under `finalProject_260531/`.

## Mission

Study how well an LLM agent can translate PyTorch programs into efficient,
correct GPU implementations for an NVIDIA Tesla V100-SXM2-32GB. Optimize the
fixed 30-problem subset in `tasks.txt`, measure correctness and speedup against
PyTorch, and report wins and losses honestly.

Read these files before substantial work:

1. `FINAL_PROJECT.md` for project layout and quick-start instructions.
2. `PROMPT.md` for the full research requirements and fixed task list.
3. `PIPELINE.md` for the evaluation workflow and environment hazards.
4. `progress/level{1,2,3}/_summary.md` for the current state and conclusions.
5. The problem-specific progress file before changing a solution.
6. The newest `results/profiles/L<L>_P<P>/*/feedback.md`, when present, before
   optimizing a previously profiled task.

## Current State

- All fixed tasks are implemented in `solutions/`.
- Level 1: 15/15 correct.
- Level 2: 10/10 correct.
- Level 3: 5/5 correct.
- Formal batch results exist in `results/eval_all_v100.json` and
  `results/RESULTS_v100.md`.
- The report source and generated PDF exist in `report/`.
- Honest no-fallback attempts for Level-1 P50, P56, P61, P76, and P97 are in
  `handwritten/`, with results documented in `handwritten/RESULTS_handwritten.md`.

Do not assume the next task is an unfinished kernel. First inspect the current
results and the user's request. Unless explicitly asked to re-optimize a
problem, prioritize verification, analysis, tables, and report improvements.

## Hardware Constraints

Target only NVIDIA Volta V100, compute capability 7.0:

- FP32 is the primary precision.
- No TF32.
- No native BF16 acceleration.
- No `cp.async`, WGMMA, or Ampere/Hopper-only features.
- Prefer strategies appropriate for V100 registers, shared memory, warps, and
  approximately 900 GB/s HBM2 bandwidth.

The working shell may be a login node without an active GPU or the correct
Python environment. Before GPU evaluation, verify the environment rather than
assuming CUDA is available:

```bash
module load cuda
conda activate kernelbench
python finalProject_260531/check.py
```

Never run CUDA checks, kernel compilation, profiling, or evaluation directly on
the login node (`un-ln01.twcc.ai`). Submit GPU work through Slurm to a compute
node. The verified course account is `ACD115083`; `gtest` is suitable for short
checks (30-minute limit):

```bash
srun --account=ACD115083 --partition=gtest --gres=gpu:1 \
    --time=00:05:00 --nodes=1 --ntasks-per-node=1 \
    --cpus-per-task=4 --mem=8G <command>
```

Confirm that the allocated hostname is a compute node such as `gn1001.twcc.ai`
and that `nvidia-smi` reports a Tesla V100-SXM2-32GB before evaluating. Use
`sbatch` instead of `srun` for long or disconnect-sensitive jobs.

Use the repository's current path; do not hard-code paths from old progress
notes such as `/work/distant22/KernelBench`.

## Profiling Feedback

Use `profile_feedback.py` only through Slurm. It validates correctness/timing,
compares candidate/eager/compile execution paths, and emits structured evidence
plus an agent-facing diagnosis:

```bash
sbatch finalProject_260531/profile_feedback.sbatch doctor

sbatch finalProject_260531/profile_feedback.sbatch profile \
    --level <L> --problem-id <P> \
    --solution finalProject_260531/solutions/level<L>/<NN>_<name>.py \
    --paths candidate,eager,compile --deep
```

Structured outputs live in `results/profiles/L<L>_P<P>/<run-id>/`. Native
Nsight reports are diagnostics, not formal timing results. Never sum per-kernel
rate metrics such as occupancy or bandwidth utilization. Do not automatically
change a kernel based on profiler feedback; use the evidence to propose and
measure one scoped optimization at a time.

The Slurm wrapper loads `nvhpc-24.11_hpcx-2.20_cuda-12.6` when repository-local
Nsight tools are absent. On the verified V100 node this exposes `ncu 2024.3.2`,
`nsys 2024.6.1`, and permits non-admin hardware-counter collection.

Follow the operational guide in `PIPELINE.md`, section
“Nsight-Guided Agent Feedback Pipeline.” Always read the newest `feedback.md`
before inspecting raw artifacts. Use `profile.json` for structured evidence,
and preserve raw rates per kernel rather than aggregating them.

## Required Workflow

For a kernel change:

1. Read the corresponding `KernelBench/level<L>/<problem>.py` baseline.
2. Read the existing solution and its progress record.
3. Explain the operator behavior and whether it is compute- or memory-bound.
4. State the V100 tiling, memory-access, occupancy, and conflict-avoidance plan.
5. Make the smallest useful implementation change.
6. Run correctness and performance evaluation on a V100 when available.
7. Update the corresponding progress Markdown with configuration, correctness,
   runtime, speedup, interpretation, and the exact evaluation command.
8. Update the relevant `_summary.md` and formal result/report files only when
   new measurements justify it.

Keep each `ModelNew` interface identical to its baseline `Model`. Follow
existing solution patterns and preserve useful comments and compatibility
workarounds.

## Output Locations

- Solutions:
  `solutions/level<L>/<NN>_<short_name>.py`
- Per-problem records:
  `progress/level<L>/<NN>_<short_name>.md`
- Honest from-scratch alternatives to library-backed official solutions:
  `handwritten/`
- Batch results:
  `results/`
- Final report:
  `report/`

Do not place final-project kernels or progress records in `runs/` or elsewhere.

## Evaluation

Single-problem Triton evaluation template:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_and_check.py \
    ref_origin=kernelbench level=<L> problem_id=<P> \
    kernel_src_path=finalProject_260531/solutions/level<L>/<NN>_<name>.py \
    eval_mode=local gpu_arch='["Volta"]' \
    check_kernel=False backend=triton
```

For all 30 tasks, use:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python finalProject_260531/run_eval_all.py
```

`run_eval_all.py` deliberately evaluates each problem in a separate subprocess
to prevent monkey-patches and CUDA state from leaking between tasks.

Correctness is mandatory. Treat speedups near 1.0 as measurement-sensitive and
retain raw measurements. Do not report rough summary estimates as formal
`fast_p` results when batch results are available.

## Honest Attempt Policy

Every claimed hand-written optimization must be evaluated honestly. Never hide
a losing or broken hand-written kernel behind a silent cuBLAS, cuDNN, PyTorch,
or SDPA fallback.

A library-backed implementation may remain the official parity solution when
appropriate, but its corresponding from-scratch attempt and outcome must be
kept in `handwritten/` and documented, including losses, compilation failures,
or environment limitations.

## Known Hazards

- The login node has a roughly 20 GB per-user host-memory cgroup limit. Large
  tasks, especially Level-1 P76, can trigger the OOM killer while the harness
  constructs host tensors. Do not casually rerun P76 on the login node.
- Run risky or long V100 evaluations inside `tmux` and one problem per
  subprocess.
- Some large-output solutions use chunked `torch.allclose` workarounds to avoid
  allocating several multi-GB intermediates. Preserve them unless replacing
  them with a verified safer approach.
- KernelBench RNG handling can make dropout correctness comparisons
  inconsistent. Existing dropout workarounds are intentional; inspect their
  progress notes before changing them.
- Triton kernels require `check_kernel=False backend=triton` in the standard
  evaluator because the CUDA-oriented static checker otherwise rejects them.
- Matmul-heavy FP32 tasks are generally dominated by highly tuned cuBLAS on
  V100. Fusion and memory-bound operators are the most promising optimization
  targets.

## Editing Discipline

- Preserve existing measurements and user work.
- Do not replace measured values without a new reproducible run.
- Keep changes scoped to the requested task.
- Prefer ASCII in code and concise comments.
- Clearly distinguish measured results, estimates, and unsupported hypotheses.
