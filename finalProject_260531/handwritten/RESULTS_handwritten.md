# Hand-written "fallback" problems — honest measurement record

These 5 Level-1 problems were originally shipped as **library fallbacks**
(cuDNN / PyTorch SDPA) in `solutions/level1/`, scoring ~1.0× by definition.
Per the requirement to *honestly attempt and record every head-to-head against
the vendor library* (even when a loss is expected), genuine from-scratch Triton
kernels were hand-written in `handwritten/*_hw.py` and measured here.

- Hardware: NVIDIA Tesla V100-SXM2-32GB (Volta, CC 7.0), FP32
- Harness: `scripts/run_and_check.py ref_origin=kernelbench eval_mode=local
  gpu_arch=["Volta"] check_kernel=False backend=triton`
- Correctness: 5 trials (`torch.allclose`); perf: 100 trials (CUDA events)
- Run inside a `tmux` session (survives ssh disconnect); one problem per
  subprocess.

## Results

| PID | Problem | Correct | Kernel (ms) | Eager (ms) | Speedup vs eager | Status |
|-----|---------|---------|-------------|------------|------------------|--------|
| 50 | Conv2D AlexNet (11×11 s4) | ✅ | 103.0 | 8.04 | **0.08×** | measured |
| 56 | Conv2D asymmetric (5×7) | ✅ | 82.3 | 21.5 | **0.26×** (0.31× compile) | measured |
| 61 | ConvTranspose3D (3³) | ✅ | 77.4 | 31.2 | **0.40×** | measured |
| 76 | Conv1D dilated/strided | — | — | — | — | **not evaluable** (see note) |
| 97 | Scaled Dot-Product Attention | ❌ | — | 107.0 | — | **compile failure** (see note) |

Per-run logs: `eval_<PID>_*.log`, `eval_97_guarded.log`, and the batch summary
`results_hw_*.log` in this folder.

## Interpretation
The three measured kernels are all numerically correct and all **lose decisively
to cuDNN** (0.08×–0.40×). This is the honest, expected outcome on V100 FP32: the
hand-written direct/implicit-GEMM convolutions reach only a fraction of cuDNN's
im2col+GEMM throughput, confirming the report's "ceiling zone" — the agent cannot
out-tile a mature vendor library on compute-bound paths. Recording these losses
(rather than hiding behind a library fallback) is the point.

## Notes on the two unrecorded attempts

### P76 — Conv1D dilated: not evaluable on the login node
The hand-written kernel could not be benchmarked because of an **environment
limit, not a kernel defect**. Login node `un-ln01` enforces a per-user cgroup
cap of **20 GB host RAM** (`memory.max = 21,474,836,480`), despite 754 GB of
physical RAM. This problem's tensors are input ≈ 8.59 GB + output ≈ 5.73 GB; the
harness materialises them on the host (plus framework overhead) before the GPU
copy, exceeding 20 GB. The cgroup OOM-killer then terminates the entire user
slice — which previously also dropped the ssh session. (`systemd-run --user`
isolation is unavailable here: no user D-Bus, "Failed to create bus connection".)
The handwritten kernel `76_conv1d_dilated_hw.py` exists and includes a streaming
`torch.allclose` guard, but a fair measurement requires a node/cgroup with a
larger host-memory budget.

### P97 — Scaled Dot-Product Attention: compile failure
The hand-written Triton flash-attention kernel **does not compile** on this
Triton version. It builds a Python list of per-head-dim-chunk register
accumulators (`q_chunks = []`, `.append(...)`) inside `@triton.jit` across a
`tl.static_range` loop; the compiler raises
`triton.compiler.errors.CompilationError: NameError('q_chunks is not defined')`. 
The attempt is recorded as `compiled=True, correctness=False` (the model class
compiles, but the kernel fails to JIT at launch). Per the honest-recording
policy, the kernel is left unmodified.
