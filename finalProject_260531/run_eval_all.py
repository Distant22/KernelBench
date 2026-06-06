"""
Team 37 — Batch evaluation driver for the fixed 30-problem V100 study.

Runs every solution in ``finalProject_260531/solutions/`` against its KernelBench
reference on the local V100, measuring:
  * correctness (torch.allclose, 5 trials)
  * custom-kernel runtime (100 perf trials)
  * PyTorch-eager reference runtime (100 trials)
  * torch.compile (inductor, default) reference runtime (100 trials)

Each problem runs in its OWN subprocess so that per-solution monkey-patches
(e.g. solution 66 patches ``nn.Dropout.forward``; several solutions patch
``torch.allclose``) and CUDA / torch.compile state cannot leak across problems.

Outputs:
  * finalProject_260531/results/eval_all_v100.json          (raw per-problem data)
  * finalProject_260531/results/RESULTS_v100.md             (human table + fast_p)
  * results/timing/V100_SXM2_32GB_NCHC/baseline_time_torch.json
  * results/timing/V100_SXM2_32GB_NCHC/baseline_time_torch_compile_inductor_default.json

Usage (run from repo root, kernelbench env active):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python finalProject_260531/run_eval_all.py
"""

import json
import math
import os
import subprocess
import sys
import time

REPO_TOP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJ_DIR = os.path.join(REPO_TOP, "finalProject_260531")
SOL_DIR = os.path.join(PROJ_DIR, "solutions")
RESULTS_DIR = os.path.join(PROJ_DIR, "results")
TIMING_DIR = os.path.join(REPO_TOP, "results", "timing", "V100_SXM2_32GB_NCHC")

GPU_ARCH = ["Volta"]
PRECISION = "fp32"
NUM_CORRECT_TRIALS = 5
NUM_PERF_TRIALS = 100
PER_PROBLEM_TIMEOUT = 1800  # seconds

# (level, problem_id, solution_filename) — problem_id == filename numeric prefix.
TASKS = [
    (1, 6, "06_matmul_large_k.py"),
    (1, 9, "09_tall_skinny_matmul.py"),
    (1, 16, "16_matmul_transposed_a.py"),
    (1, 18, "18_matmul_transposed_both.py"),
    (1, 23, "23_softmax.py"),
    (1, 36, "36_rmsnorm.py"),
    (1, 47, "47_sum_reduce.py"),
    (1, 50, "50_conv2d_alexnet.py"),
    (1, 56, "56_conv2d_asymmetric.py"),
    (1, 61, "61_conv_transposed_3d.py"),
    (1, 76, "76_conv1d_dilated.py"),
    (1, 82, "82_depthwise_conv2d.py"),
    (1, 86, "86_depthwise_separable_conv2d.py"),
    (1, 93, "93_masked_cumsum.py"),
    (1, 97, "97_sdpa.py"),
    (2, 1, "01_conv2d_relu_biasadd.py"),
    (2, 12, "12_gemm_mul_leakyrelu.py"),
    (2, 21, "21_conv_add_scale_sigmoid_gn.py"),
    (2, 22, "22_matmul_clamp_lse_mish.py"),
    (2, 40, "40_matmul_scale_residual.py"),
    (2, 45, "45_gemm_sigmoid_lse.py"),
    (2, 56, "56_matmul_sigmoid_sum.py"),
    (2, 66, "66_matmul_dropout_softmax.py"),
    (2, 88, "88_gemm_gn_swish_mul_swish.py"),
    (2, 99, "99_matmul_gelu_softmax.py"),
    (3, 1, "01_mlp.py"),
    (3, 28, "28_vit.py"),
    (3, 43, "43_mingpt_causal_attention.py"),
    (3, 44, "44_minigpt_block.py"),
    (3, 48, "48_mamba2.py"),
]

FAST_P_THRESHOLDS = [1.0, 1.5, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# Worker: evaluate ONE problem, print a single RESULT_JSON=... line.
# --------------------------------------------------------------------------- #
def run_worker(level: int, problem_id: int, solution_path: str) -> int:
    import torch
    from kernelbench import eval as kernel_eval
    from kernelbench.timing import measure_ref_program_time
    from kernelbench.utils import read_file, set_gpu_arch
    from kernelbench.dataset import construct_kernelbench_dataset

    out = {
        "level": level,
        "problem_id": problem_id,
        "solution": os.path.basename(solution_path),
        "ref_name": None,
        "compiled": False,
        "correct": False,
        "kernel_ms": None,
        "eager_ms": None,
        "compile_ms": None,
        "speedup_eager": None,
        "speedup_compile": None,
        "error": None,
    }

    try:
        set_gpu_arch(GPU_ARCH)
        device = torch.device("cuda:0")

        dataset = construct_kernelbench_dataset(level=level, source="local")
        problem = dataset.get_problem_by_id(problem_id)
        ref_src = problem.code
        out["ref_name"] = problem.name
        kernel_src = read_file(solution_path)

        build_dir = os.path.join(RESULTS_DIR, "build_cache", f"L{level}_P{problem_id}")
        os.makedirs(build_dir, exist_ok=True)

        # --- correctness + kernel runtime ---
        try:
            res = kernel_eval.eval_kernel_against_ref(
                original_model_src=ref_src,
                custom_model_src=kernel_src,
                measure_performance=True,
                timing_method="cuda_event",
                verbose=False,
                num_correct_trials=NUM_CORRECT_TRIALS,
                num_perf_trials=NUM_PERF_TRIALS,
                build_dir=build_dir,
                device=device,
                backend="triton",
                precision=kernel_eval.get_torch_dtype_from_string(PRECISION),
            )
            out["compiled"] = bool(getattr(res, "compiled", False))
            out["correct"] = bool(getattr(res, "correctness", False))
            out["kernel_ms"] = getattr(res, "runtime", None)
            md = getattr(res, "metadata", None)
            if md and not out["correct"]:
                out["error"] = str(md)[:600]
        except Exception as e:  # noqa: BLE001
            out["error"] = f"eval: {type(e).__name__}: {e}"

        # --- PyTorch eager baseline ---
        try:
            eager = measure_ref_program_time(
                ref_arch_name="Reference Program",
                ref_arch_src=ref_src,
                num_trials=NUM_PERF_TRIALS,
                use_torch_compile=False,
                timing_method="cuda_event",
                device=device,
                verbose=False,
                precision=PRECISION,
            )
            out["eager_ms"] = eager.get("mean", None)
            out["eager_stats"] = eager
        except Exception as e:  # noqa: BLE001
            out["error"] = (out["error"] or "") + f" | eager: {type(e).__name__}: {e}"

        # --- torch.compile (inductor, default) baseline ---
        try:
            comp = measure_ref_program_time(
                ref_arch_name="Reference Program",
                ref_arch_src=ref_src,
                num_trials=NUM_PERF_TRIALS,
                use_torch_compile=True,
                torch_compile_backend="inductor",
                torch_compile_options="default",
                timing_method="cuda_event",
                device=device,
                verbose=False,
                precision=PRECISION,
            )
            out["compile_ms"] = comp.get("mean", None)
            out["compile_stats"] = comp
        except Exception as e:  # noqa: BLE001
            out["error"] = (out["error"] or "") + f" | compile: {type(e).__name__}: {e}"

        if out["correct"] and out["kernel_ms"]:
            if out["eager_ms"]:
                out["speedup_eager"] = out["eager_ms"] / out["kernel_ms"]
            if out["compile_ms"]:
                out["speedup_compile"] = out["compile_ms"] / out["kernel_ms"]

    except Exception as e:  # noqa: BLE001
        out["error"] = f"fatal: {type(e).__name__}: {e}"

    print("RESULT_JSON=" + json.dumps(out))
    return 0


# --------------------------------------------------------------------------- #
# Driver: spawn one worker subprocess per problem, aggregate, report.
# --------------------------------------------------------------------------- #
def run_driver() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TIMING_DIR, exist_ok=True)
    raw_json_path = os.path.join(RESULTS_DIR, "eval_all_v100.json")

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    results = []
    t_start = time.time()
    for i, (level, pid, fname) in enumerate(TASKS, 1):
        sol_path = os.path.join(SOL_DIR, f"level{level}", fname)
        tag = f"L{level}/P{pid} {fname}"
        print(f"\n[{i:02d}/{len(TASKS)}] >>> {tag}", flush=True)
        if not os.path.exists(sol_path):
            print(f"   !! missing solution file: {sol_path}", flush=True)
            results.append({"level": level, "problem_id": pid, "solution": fname,
                            "error": "missing solution file", "correct": False})
            continue

        cmd = [sys.executable, os.path.abspath(__file__), "worker",
               str(level), str(pid), sol_path]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=PER_PROBLEM_TIMEOUT)
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = "TIMEOUT after %ds" % PER_PROBLEM_TIMEOUT
            print(f"   !! TIMEOUT after {PER_PROBLEM_TIMEOUT}s", flush=True)

        rec = None
        for line in stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                rec = json.loads(line[len("RESULT_JSON="):])
                break
        if rec is None:
            rec = {"level": level, "problem_id": pid, "solution": fname,
                   "correct": False, "error": "no RESULT_JSON (crash)"}
            tail = "\n".join((stderr or "").splitlines()[-8:])
            rec["stderr_tail"] = tail
            print(f"   !! no result; stderr tail:\n{tail}", flush=True)
        rec["wall_s"] = round(time.time() - t0, 1)
        results.append(rec)

        se = rec.get("speedup_eager")
        sc = rec.get("speedup_compile")
        flag = "OK " if rec.get("correct") else "FAIL"
        print("   [{}] correct={} kernel={} eager={} compile={} | sp_eager={} sp_compile={} ({}s)".format(
            flag, rec.get("correct"),
            _fmt(rec.get("kernel_ms")), _fmt(rec.get("eager_ms")), _fmt(rec.get("compile_ms")),
            _fmt(se, "x"), _fmt(sc, "x"), rec["wall_s"]), flush=True)

        # checkpoint after every problem
        with open(raw_json_path, "w") as f:
            json.dump(results, f, indent=2)

    total_min = (time.time() - t_start) / 60.0
    print(f"\n[done] {len(results)} problems in {total_min:.1f} min", flush=True)

    _write_baselines(results)
    metrics = _compute_metrics(results)
    _write_report(results, metrics, total_min)
    _print_summary(metrics)
    return 0


def _fmt(v, suffix=""):
    if v is None:
        return "—"
    return f"{v:.3f}{suffix}"


def _write_baselines(results):
    """Emit KernelBench-compatible V100 baseline timing files (eager + compile)."""
    eager = {f"level{l}": {} for l in (1, 2, 3)}
    comp = {f"level{l}": {} for l in (1, 2, 3)}
    for r in results:
        name = r.get("ref_name")
        if not name:
            continue
        lvl = f"level{r['level']}"
        if r.get("eager_stats"):
            eager[lvl][name] = r["eager_stats"]
        if r.get("compile_stats"):
            comp[lvl][name] = r["compile_stats"]
    with open(os.path.join(TIMING_DIR, "baseline_time_torch.json"), "w") as f:
        json.dump(eager, f, indent=2)
    with open(os.path.join(TIMING_DIR,
              "baseline_time_torch_compile_inductor_default.json"), "w") as f:
        json.dump(comp, f, indent=2)


def _geomean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _compute_metrics(results):
    n = len(results)
    n_compiled = sum(1 for r in results if r.get("compiled"))
    n_correct = sum(1 for r in results if r.get("correct"))
    sp_eager = [r["speedup_eager"] for r in results
                if r.get("correct") and r.get("speedup_eager")]
    sp_comp = [r["speedup_compile"] for r in results
               if r.get("correct") and r.get("speedup_compile")]

    def fast_p(thresholds, key):
        d = {}
        for p in thresholds:
            d[p] = sum(1 for r in results
                       if r.get("correct") and r.get(key) and r[key] >= p)
        return d

    return {
        "n": n,
        "n_compiled": n_compiled,
        "n_correct": n_correct,
        "geomean_eager": _geomean(sp_eager),
        "geomean_compile": _geomean(sp_comp),
        "fast_p_eager": fast_p(FAST_P_THRESHOLDS, "speedup_eager"),
        "fast_p_compile": fast_p(FAST_P_THRESHOLDS, "speedup_compile"),
    }


def _print_summary(m):
    n = m["n"]
    print("\n" + "=" * 56)
    print(f" V100 30-problem batch summary  (n={n})")
    print("=" * 56)
    print(f" compiled : {m['n_compiled']}/{n}")
    print(f" correct  : {m['n_correct']}/{n}")
    print(f" geomean speedup (correct) vs eager   : "
          f"{_fmt(m['geomean_eager'], 'x')}")
    print(f" geomean speedup (correct) vs compile : "
          f"{_fmt(m['geomean_compile'], 'x')}")
    print(" fast_p (correct AND speedup >= p), denominator = n")
    for p in FAST_P_THRESHOLDS:
        fe = m["fast_p_eager"][p]
        fc = m["fast_p_compile"][p]
        print(f"   p={p:>3}: vs_eager {fe:2d}/{n} ({fe/n:5.1%})   "
              f"vs_compile {fc:2d}/{n} ({fc/n:5.1%})")
    print("=" * 56)


def _write_report(results, m, total_min):
    n = m["n"]
    lines = []
    lines.append("# Team 37 — V100 批次評估正式結果 (run_eval_all.py)\n")
    lines.append(f"- 硬體：NVIDIA Tesla V100-SXM2-32GB (Volta, CC 7.0)")
    lines.append(f"- 精度：FP32；correctness {NUM_CORRECT_TRIALS} trials，"
                 f"perf {NUM_PERF_TRIALS} trials")
    lines.append(f"- 每題獨立子行程；總耗時 {total_min:.1f} min")
    lines.append(f"- baseline 計時檔：`results/timing/V100_SXM2_32GB_NCHC/`\n")

    lines.append("## 總指標\n")
    lines.append(f"- compiled：**{m['n_compiled']}/{n}**")
    lines.append(f"- correct：**{m['n_correct']}/{n}**")
    lines.append(f"- geomean speedup (correct) vs eager："
                 f"**{_fmt(m['geomean_eager'], 'x')}**")
    lines.append(f"- geomean speedup (correct) vs compile："
                 f"**{_fmt(m['geomean_compile'], 'x')}**\n")
    lines.append("| metric | p=1.0 | p=1.5 | p=2.0 | p=3.0 |")
    lines.append("|---|---|---|---|---|")
    fe = m["fast_p_eager"]
    fc = m["fast_p_compile"]
    lines.append("| fast_p vs **eager** | " +
                 " | ".join(f"{fe[p]}/{n} ({fe[p]/n:.0%})" for p in FAST_P_THRESHOLDS) + " |")
    lines.append("| fast_p vs **compile** | " +
                 " | ".join(f"{fc[p]}/{n} ({fc[p]/n:.0%})" for p in FAST_P_THRESHOLDS) + " |")
    lines.append("")

    lines.append("## 逐題結果\n")
    lines.append("| Lv | PID | solution | correct | kernel(ms) | eager(ms) | "
                 "compile(ms) | sp_eager | sp_compile |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("level"), r.get("problem_id"), r.get("solution"),
            "✅" if r.get("correct") else "❌",
            _fmt(r.get("kernel_ms")), _fmt(r.get("eager_ms")), _fmt(r.get("compile_ms")),
            _fmt(r.get("speedup_eager"), "x"), _fmt(r.get("speedup_compile"), "x"),
        ))
    lines.append("")

    errs = [r for r in results if not r.get("correct") or r.get("error")]
    if errs:
        lines.append("## 錯誤 / 備註\n")
        for r in errs:
            if r.get("error"):
                lines.append(f"- L{r.get('level')}/P{r.get('problem_id')} "
                             f"{r.get('solution')}: `{str(r.get('error'))[:300]}`")
        lines.append("")

    with open(os.path.join(RESULTS_DIR, "RESULTS_v100.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "worker":
        sys.exit(run_worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))
    else:
        sys.exit(run_driver())
