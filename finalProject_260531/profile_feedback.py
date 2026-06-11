#!/usr/bin/env python3
"""Compute-node-only profiling feedback for Team 37 KernelBench solutions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = REPO_ROOT / "finalProject_260531"
PROFILE_ROOT = PROJECT_DIR / "results" / "profiles"
TOOLS_BIN = REPO_ROOT / ".tools" / "nsight" / "bin"
COMPUTE_HOST_RE = re.compile(r"^gn\d+(?:\.twcc\.ai)?$")
EXPECTED_GPU = "Tesla V100-SXM2-32GB"

DIAGNOSIS_THRESHOLDS = {
    "launch_api_pct": 20.0,
    "many_kernel_count": 10,
    "short_kernel_us": 20.0,
    "memory_pct": 70.0,
    "compute_pct": 70.0,
    "low_resource_pct": 50.0,
    "low_occupancy_pct": 40.0,
    "low_eligible_warps": 1.0,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")


def _run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def _tool_path(name: str) -> str | None:
    local = TOOLS_BIN / name
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which(name)


def _require_compute_node() -> None:
    host = socket.gethostname()
    if not COMPUTE_HOST_RE.match(host):
        raise RuntimeError(
            f"GPU profiling is compute-node-only; current host is {host!r}. "
            "Submit finalProject_260531/profile_feedback.sbatch instead."
        )


def _gpu_info() -> dict[str, Any]:
    import torch

    available = torch.cuda.is_available()
    info: dict[str, Any] = {"available": available}
    if available:
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"{props.major}.{props.minor}",
                "memory_bytes": props.total_memory,
            }
        )
    return info


def doctor_report(probe_counters: bool = True) -> dict[str, Any]:
    host = socket.gethostname()
    nsys = _tool_path("nsys")
    ncu = _tool_path("ncu")
    report: dict[str, Any] = {
        "host": host,
        "compute_node": bool(COMPUTE_HOST_RE.match(host)),
        "gpu": _gpu_info(),
        "tools": {
            "nsys": {"path": nsys, "available": bool(nsys)},
            "ncu": {"path": ncu, "available": bool(ncu)},
        },
        "hardware_counters": {
            "available": False,
            "probed": False,
            "error": None,
            "admin_action": (
                "Ask the cluster administrator to permit non-admin NVIDIA GPU "
                "performance-counter access (NVreg_RestrictProfilingToAdminUsers=0)."
            ),
        },
    }
    for name, path in (("nsys", nsys), ("ncu", ncu)):
        if path:
            proc = _run([path, "--version"])
            report["tools"][name]["version"] = (proc.stdout or proc.stderr).strip()
            report["tools"][name]["returncode"] = proc.returncode

    if probe_counters and ncu and report["compute_node"] and report["gpu"]["available"]:
        report["hardware_counters"]["probed"] = True
        proc = _run(
            [
                ncu,
                "--section",
                "LaunchStats",
                "--launch-count",
                "1",
                sys.executable,
                str(Path(__file__).resolve()),
                "_counter_probe",
            ]
        )
        output = f"{proc.stdout}\n{proc.stderr}".strip()
        report["hardware_counters"]["available"] = proc.returncode == 0
        if proc.returncode == 0:
            report["hardware_counters"]["admin_action"] = None
        else:
            report["hardware_counters"]["error"] = output[-2000:]
    return report


def _load_problem(level: int, problem_id: int):
    from kernelbench.dataset import construct_kernelbench_dataset

    dataset = construct_kernelbench_dataset(level=level, source="local")
    return dataset.get_problem_by_id(problem_id)


def _evaluate(level: int, problem_id: int, solution: Path) -> dict[str, Any]:
    import torch
    from kernelbench import eval as kernel_eval
    from kernelbench.timing import measure_ref_program_time
    from kernelbench.utils import read_file, set_gpu_arch

    set_gpu_arch(["Volta"])
    problem = _load_problem(level, problem_id)
    ref_src = problem.code
    custom_src = read_file(solution)
    device = torch.device("cuda:0")
    result = kernel_eval.eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=custom_src,
        measure_performance=True,
        timing_method="cuda_event",
        num_correct_trials=5,
        num_perf_trials=100,
        device=device,
        backend="triton",
        precision=torch.float32,
    )
    eager = measure_ref_program_time(
        ref_arch_name="Reference Program",
        ref_arch_src=ref_src,
        num_trials=100,
        use_torch_compile=False,
        timing_method="cuda_event",
        device=device,
        verbose=False,
        precision="fp32",
    )
    compiled = measure_ref_program_time(
        ref_arch_name="Reference Program",
        ref_arch_src=ref_src,
        num_trials=100,
        use_torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_options="default",
        timing_method="cuda_event",
        device=device,
        verbose=False,
        precision="fp32",
    )
    kernel_ms = result.runtime if result.correctness else None
    eager_ms = eager.get("mean")
    compile_ms = compiled.get("mean")
    output = {
        "compiled": result.compiled,
        "correct": result.correctness,
        "metadata": result.metadata,
        "kernel_ms": kernel_ms,
        "eager_ms": eager_ms,
        "compile_ms": compile_ms,
        "speedup_eager": eager_ms / kernel_ms if kernel_ms and eager_ms else None,
        "speedup_compile": compile_ms / kernel_ms if kernel_ms and compile_ms else None,
    }
    formal_results = PROJECT_DIR / "results" / "eval_all_v100.json"
    if formal_results.is_file():
        for record in json.loads(formal_results.read_text(encoding="utf-8")):
            if record.get("level") == level and record.get("problem_id") == problem_id:
                formal_ms = record.get("kernel_ms")
                output["formal_kernel_ms"] = formal_ms
                output["runtime_regression_pct"] = (
                    (kernel_ms / formal_ms - 1.0) * 100.0
                    if kernel_ms and formal_ms
                    else None
                )
                break
    return output


def _load_execution_path(level: int, problem_id: int, solution: Path, path_name: str):
    import torch
    from kernelbench.eval import (
        _process_input_tensor,
        load_custom_model_with_tempfile,
        load_original_model_and_inputs,
        set_seed,
    )
    from kernelbench.utils import read_file, set_gpu_arch

    set_gpu_arch(["Volta"])
    device = torch.device("cuda:0")
    problem = _load_problem(level, problem_id)
    context: dict[str, Any] = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem.code, context)
    set_seed(42)
    init_inputs = [_process_input_tensor(x, device, "triton", torch.float32) for x in get_init_inputs()]

    temp_file = None
    set_seed(42)
    if path_name == "candidate":
        ModelNew, temp_file = load_custom_model_with_tempfile(read_file(solution), "ModelNew")
        model = ModelNew(*init_inputs).to(device=device, dtype=torch.float32)
    else:
        model = Model(*init_inputs).to(device=device, dtype=torch.float32)
        if path_name == "compile":
            model = torch.compile(model, backend="inductor")
    set_seed(42)
    inputs = [_process_input_tensor(x, device, "triton", torch.float32) for x in get_inputs()]
    return model, inputs, temp_file


def _workload(args: argparse.Namespace) -> int:
    _require_compute_node()
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    model, inputs, temp_file = _load_execution_path(
        args.level, args.problem_id, Path(args.solution), args.path_name
    )
    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                model(*inputs)
            torch.cuda.synchronize()
            if args.torch_profile:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function(f"KB_PROFILE_{args.path_name}"):
                        torch.cuda.nvtx.range_push(f"KB_PROFILE_{args.path_name}")
                        for _ in range(args.iterations):
                            model(*inputs)
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                events = []
                for event in prof.events():
                    if "cuda" not in str(getattr(event, "device_type", "")).lower():
                        continue
                    cuda_us = float(
                        getattr(event, "device_time_total", 0.0)
                        or getattr(event, "self_device_time_total", 0.0)
                        or (
                            event.time_range.elapsed_us()
                            if getattr(event, "time_range", None) is not None
                            else 0.0
                        )
                        or 0.0
                    )
                    event_name = getattr(event, "name", getattr(event, "key", "<unknown>"))
                    if cuda_us > 0 and not str(event_name).startswith("KB_PROFILE_"):
                        events.append(
                            {
                                "name": event_name,
                                "calls": 1,
                                "total_us": cuda_us,
                                "avg_us": cuda_us,
                            }
                        )
                _write_json(Path(args.output), {"path": args.path_name, "events": events})
            else:
                torch.cuda.nvtx.range_push(f"KB_PROFILE_{args.path_name}")
                torch.cuda.cudart().cudaProfilerStart()
                for _ in range(args.iterations):
                    model(*inputs)
                torch.cuda.cudart().cudaProfilerStop()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
    finally:
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
    return 0


def parse_torch_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    kernels = sorted(data.get("events", []), key=lambda row: row.get("total_us", 0), reverse=True)
    total_us = sum(float(row.get("total_us", 0)) for row in kernels)
    return {
        "source": "torch_profiler",
        "kernel_count": sum(int(row.get("calls", 0)) for row in kernels),
        "unique_kernel_count": len(kernels),
        "gpu_total_us": total_us,
        "cuda_api_total_us": None,
        "cuda_api_pct": None,
        "kernels": kernels,
    }


def _find_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {
        re.sub(r"[^a-z0-9]", "", col.lower()): col
        for col in columns
        if isinstance(col, str)
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _read_nsys_stats_rows(path: Path) -> list[dict[str, str]]:
    """Skip Nsight status/NOTICE text and parse from the actual CSV header."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip('"').startswith("Time (%)")
        ),
        None,
    )
    if header_index is None:
        return []
    return [dict(row) for row in csv.DictReader(lines[header_index:])]


def parse_nsys_csv(path: Path) -> dict[str, Any]:
    """Parse an Nsight Systems stats CSV without depending on one CLI version."""
    rows = _read_nsys_stats_rows(path)
    if not rows:
        return {"source": "nsys", "kernels": [], "kernel_count": 0}
    columns = list(rows[0])
    name_col = _find_col(columns, ("Name", "Kernel Name"))
    time_col = _find_col(columns, ("Total Time (ns)", "Total Time", "Time (ns)"))
    instances_col = _find_col(columns, ("Instances", "Calls", "Count"))
    kernels = []
    for row in rows:
        try:
            total_ns = float(str(row.get(time_col, "0")).replace(",", ""))
        except (TypeError, ValueError):
            continue
        calls = int(float(str(row.get(instances_col, "1")).replace(",", ""))) if instances_col else 1
        kernels.append(
            {
                "name": row.get(name_col, "<unknown>") if name_col else "<unknown>",
                "calls": calls,
                "total_us": total_ns / 1000.0,
                "avg_us": total_ns / 1000.0 / max(calls, 1),
            }
        )
    kernels.sort(key=lambda row: row["total_us"], reverse=True)
    return {
        "source": "nsys",
        "kernel_count": sum(row["calls"] for row in kernels),
        "unique_kernel_count": len(kernels),
        "gpu_total_us": sum(row["total_us"] for row in kernels),
        "kernels": kernels,
    }


def parse_nsys_api_csv(path: Path) -> dict[str, Any]:
    rows = _read_nsys_stats_rows(path)
    if not rows:
        return {"cuda_api_total_us": None, "cuda_api_calls": 0}
    columns = list(rows[0])
    time_col = _find_col(columns, ("Total Time (ns)", "Total Time", "Time (ns)"))
    calls_col = _find_col(columns, ("Num Calls", "Calls", "Instances", "Count"))
    total_ns = 0.0
    calls = 0
    for row in rows:
        try:
            total_ns += float(str(row.get(time_col, "0")).replace(",", ""))
            calls += int(float(str(row.get(calls_col, "0")).replace(",", ""))) if calls_col else 0
        except (TypeError, ValueError):
            continue
    return {"cuda_api_total_us": total_ns / 1000.0, "cuda_api_calls": calls}


def parse_ncu_csv(path: Path) -> list[dict[str, Any]]:
    """Preserve Nsight Compute metrics per kernel launch."""
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        filtered = (line for line in handle if line.startswith('"') or line[:1].isdigit())
        rows = [dict(row) for row in csv.DictReader(filtered)]
    if rows and "Metric Name" not in rows[0] and "Metric" not in rows[0]:
        records = []
        for row in rows:
            kernel = row.get("Kernel Name")
            launch = row.get("ID")
            if not kernel or not launch:
                continue
            metrics: dict[str, Any] = {}
            for name, value in row.items():
                if not name or value in (None, ""):
                    continue
                try:
                    metrics[name] = float(value.replace(",", ""))
                except ValueError:
                    metrics[name] = value
            records.append({"kernel": kernel, "launch_id": launch, "metrics": metrics})
        return records
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        kernel = row.get("Kernel Name") or row.get("Kernel") or "<unknown>"
        launch = row.get("ID") or row.get("Launch ID") or "0"
        key = (kernel, launch)
        record = grouped.setdefault(key, {"kernel": kernel, "launch_id": launch, "metrics": {}})
        metric = row.get("Metric Name") or row.get("Metric")
        value = row.get("Metric Value") or row.get("Value")
        if metric and value is not None:
            try:
                parsed: Any = float(value.replace(",", ""))
            except ValueError:
                parsed = value
            record["metrics"][metric] = parsed
    return list(grouped.values())


def select_top_kernels(timeline: dict[str, Any], coverage: float = 0.80, cap: int = 3) -> list[dict[str, Any]]:
    kernels = timeline.get("kernels", [])
    total = sum(float(k.get("total_us", 0)) for k in kernels)
    selected = []
    covered = 0.0
    for kernel in kernels:
        if len(selected) >= cap:
            break
        selected.append(kernel)
        covered += float(kernel.get("total_us", 0))
        if total > 0 and covered / total >= coverage:
            break
    return selected


def _metric(metrics: dict[str, Any], fragments: tuple[str, ...]) -> float | None:
    for name, value in metrics.items():
        tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
        required = {
            token
            for fragment in fragments
            for token in re.findall(r"[a-z0-9]+", fragment.lower())
        }
        if required.issubset(tokens):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _raw_metric(metrics: dict[str, Any], name: str) -> float | None:
    try:
        return float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return None


def summarize_deep_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract comparable values while keeping raw metrics per kernel."""
    summaries = []
    for record in records:
        metrics = record.get("metrics", {})
        summaries.append(
            {
                "kernel": record.get("kernel"),
                "launch_id": record.get("launch_id"),
                "compute_pct": _raw_metric(
                    metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed"
                )
                or _metric(metrics, ("compute", "throughput")),
                "memory_pct": _raw_metric(
                    metrics,
                    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                )
                or _metric(metrics, ("memory", "throughput")),
                "occupancy_pct": _raw_metric(
                    metrics, "sm__warps_active.avg.pct_of_peak_sustained_active"
                )
                or _metric(metrics, ("achieved occupancy",)),
                "eligible_warps": _raw_metric(
                    metrics, "smsp__warps_eligible.avg.per_cycle_active"
                )
                or _metric(metrics, ("eligible", "warp")),
                "branch_efficiency_pct": _metric(metrics, ("branch efficiency",)),
                "shared_conflicts": _metric(metrics, ("shared", "conflict")),
                "raw_metrics": metrics,
            }
        )
    return {"kernels": summaries}


def classify_profile(timeline: dict[str, Any], deep: dict[str, Any] | None) -> dict[str, Any]:
    t = DIAGNOSIS_THRESHOLDS
    evidence: list[str] = []
    labels: list[str] = []
    kernel_count = int(timeline.get("kernel_count") or 0)
    kernels = timeline.get("kernels", [])
    avg_us = (
        sum(float(k.get("total_us", 0)) for k in kernels) / max(kernel_count, 1)
        if kernels
        else None
    )
    api_pct = timeline.get("cuda_api_pct")
    if (api_pct is not None and api_pct > t["launch_api_pct"]) or (
        kernel_count >= t["many_kernel_count"] and avg_us is not None and avg_us < t["short_kernel_us"]
    ):
        labels.append("launch-bound")
        evidence.append(f"{kernel_count} launches, average kernel {avg_us:.1f} us")
    if kernels:
        top_name = str(kernels[0].get("name", "")).lower()
        if any(token in top_name for token in ("cublas", "sgemm", "gemm_", "cudnn")):
            labels.append("library-dominated")
            evidence.append(f"dominant vendor-library kernel: {kernels[0].get('name')}")

    summaries = (deep or {}).get("kernels", [])
    if summaries:
        dominant = summaries[0]
        compute = dominant.get("compute_pct")
        memory = dominant.get("memory_pct")
        occupancy = dominant.get("occupancy_pct")
        eligible = dominant.get("eligible_warps")
        if memory is not None and compute is not None and memory >= t["memory_pct"] and compute < t["low_resource_pct"]:
            labels.append("memory-bound")
            evidence.append(f"memory throughput {memory:.1f}%, compute throughput {compute:.1f}%")
        if compute is not None and memory is not None and compute >= t["compute_pct"] and memory < t["low_resource_pct"]:
            labels.append("compute-bound")
            evidence.append(f"compute throughput {compute:.1f}%, memory throughput {memory:.1f}%")
        if (
            compute is not None
            and memory is not None
            and compute < t["low_resource_pct"]
            and memory < t["low_resource_pct"]
            and ((occupancy is not None and occupancy < t["low_occupancy_pct"]) or (eligible is not None and eligible < t["low_eligible_warps"]))
        ):
            labels.append("occupancy/latency-bound")
            evidence.append(f"occupancy={occupancy}, eligible warps={eligible}")
        branch = dominant.get("branch_efficiency_pct")
        conflicts = dominant.get("shared_conflicts")
        if (branch is not None and branch < 80.0) or (conflicts is not None and conflicts > 0):
            labels.append("divergence/conflict-risk")
            evidence.append(f"branch efficiency={branch}, shared conflicts={conflicts}")
    if not labels:
        labels.append("insufficient-counter-data" if not summaries else "balanced/unclear")
    return {"primary": labels[0], "labels": labels, "evidence": evidence}


def recommendations(diagnosis: dict[str, Any], evaluation: dict[str, Any], timeline: dict[str, Any]) -> list[str]:
    labels = set(diagnosis.get("labels", []))
    actions: list[str] = []
    if "launch-bound" in labels:
        actions.append("Fuse adjacent pointwise/reduction launches and remove avoidable materializations.")
    if "memory-bound" in labels:
        actions.append("Reduce HBM passes, preserve coalescing, and keep reused values in registers.")
    if "compute-bound" in labels:
        actions.append("Keep vendor GEMM/convolution paths unless a mathematically different algorithm removes work.")
    if "library-dominated" in labels:
        actions.append("Preserve the dominant vendor-library kernel and optimize only surrounding launches or algorithms.")
    if "occupancy/latency-bound" in labels:
        actions.append("Reduce register/shared-memory pressure or increase independent work per warp.")
    if "divergence/conflict-risk" in labels:
        actions.append("Regularize control flow and adjust shared-memory layout to remove conflicts.")
    if evaluation.get("speedup_compile") is not None and evaluation["speedup_compile"] < 1.0:
        actions.append("Compare against Inductor's fused path before adding another custom launch.")
    if timeline.get("kernel_count", 0) > 1:
        actions.append("Inspect the top-kernel timeline and target the launches covering at least 80% of GPU time.")
    default = [
        "Use the top-kernel timing breakdown to choose one measurable change.",
        "Re-run correctness and CUDA-event timing after every candidate change.",
        "Do not replace a near-peak cuBLAS GEMM with naive Triton.",
    ]
    for action in default:
        if len(actions) >= 3:
            break
        if action not in actions:
            actions.append(action)
    return actions[:3]


def render_feedback(profile: dict[str, Any]) -> str:
    ev = profile["evaluation"]
    lines = [
        f"# Profiling Feedback: L{profile['level']}/P{profile['problem_id']}",
        "",
        f"- Correct: **{ev.get('correct')}**",
        f"- Candidate runtime: **{ev.get('kernel_ms')} ms**",
        f"- Speedup vs eager: **{ev.get('speedup_eager'):.3f}x**" if ev.get("speedup_eager") else "- Speedup vs eager: unavailable",
        f"- Speedup vs compile: **{ev.get('speedup_compile'):.3f}x**" if ev.get("speedup_compile") else "- Speedup vs compile: unavailable",
        f"- Deep Nsight Compute metrics: **{'available' if profile['deep_profile']['available'] else 'unavailable'}**",
        "",
        "## Path Breakdown",
        "",
        "| Path | GPU kernels | GPU total (us) | Top kernel |",
        "|---|---:|---:|---|",
    ]
    for path_name, path in profile["paths"].items():
        timeline = path["timeline"]
        top = timeline.get("kernels", [{}])[0].get("name", "n/a") if timeline.get("kernels") else "n/a"
        lines.append(
            f"| {path_name} | {timeline.get('kernel_count', 0)} | "
            f"{timeline.get('gpu_total_us', 0):.1f} | `{top}` |"
        )
    diagnosis = profile["diagnosis"]
    lines.extend(["", "## Diagnosis", "", f"Primary: **{diagnosis['primary']}**"])
    for evidence in diagnosis.get("evidence", []):
        lines.append(f"- Evidence: {evidence}")
    lines.extend(["", "## Prioritized Actions", ""])
    for index, action in enumerate(profile["recommendations"], 1):
        lines.append(f"{index}. {action}")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Preserve correctness before optimizing.",
            "- Do not infer occupancy or bandwidth by summing rate metrics across kernels.",
            "- Do not replace a near-peak vendor GEMM/convolution with naive Triton.",
            "",
        ]
    )
    return "\n".join(lines)


def _profile_path_fallback(args: argparse.Namespace, path_name: str, run_dir: Path) -> dict[str, Any]:
    output = run_dir / f"{path_name}_torch_profile.json"
    proc = _run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_workload",
            "--level",
            str(args.level),
            "--problem-id",
            str(args.problem_id),
            "--solution",
            str(Path(args.solution).resolve()),
            "--path-name",
            path_name,
            "--torch-profile",
            "--output",
            str(output),
        ]
    )
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout)[-3000:], "timeline": {"kernels": [], "kernel_count": 0}}
    return {"timeline": parse_torch_profile(output), "artifacts": {"torch_profile": str(output)}}


def _profile_path_nsys(args: argparse.Namespace, path_name: str, run_dir: Path, nsys: str) -> dict[str, Any]:
    prefix = run_dir / f"{path_name}_nsys"
    cmd = [
        nsys,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--output",
        str(prefix),
        sys.executable,
        str(Path(__file__).resolve()),
        "_workload",
        "--level",
        str(args.level),
        "--problem-id",
        str(args.problem_id),
        "--solution",
        str(Path(args.solution).resolve()),
        "--path-name",
        path_name,
    ]
    proc = _run(cmd)
    report = prefix.with_suffix(".nsys-rep")
    csv_path = run_dir / f"{path_name}_nsys_cuda_gpu_kern_sum.csv"
    stats = _run(
        [nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(report)]
    )
    csv_path.write_text(stats.stdout, encoding="utf-8")
    api_csv_path = run_dir / f"{path_name}_nsys_cuda_api_sum.csv"
    api_stats = _run(
        [nsys, "stats", "--report", "cuda_api_sum", "--format", "csv", str(report)]
    )
    api_csv_path.write_text(api_stats.stdout, encoding="utf-8")
    if proc.returncode != 0 or stats.returncode != 0:
        fallback = _profile_path_fallback(args, path_name, run_dir)
        fallback["nsys_error"] = (proc.stderr + stats.stderr)[-3000:]
        return fallback
    timeline = parse_nsys_csv(csv_path)
    api = parse_nsys_api_csv(api_csv_path)
    timeline.update(api)
    denominator = timeline["gpu_total_us"] + (api["cuda_api_total_us"] or 0.0)
    timeline["cuda_api_pct"] = (
        api["cuda_api_total_us"] / denominator * 100.0
        if api["cuda_api_total_us"] is not None and denominator > 0
        else None
    )
    return {
        "timeline": timeline,
        "artifacts": {
            "nsys_report": str(report),
            "nsys_kernel_csv": str(csv_path),
            "nsys_api_csv": str(api_csv_path),
        },
    }


def _run_deep_profile(
    args: argparse.Namespace,
    path_name: str,
    selected: list[dict[str, Any]],
    run_dir: Path,
    ncu: str,
) -> dict[str, Any]:
    records = []
    artifacts = []
    for index, kernel in enumerate(selected):
        kernel_name = str(kernel.get("name", "")).replace("::", r"\:\:")
        csv_path = run_dir / f"{path_name}_ncu_{index}.csv"
        report_prefix = run_dir / f"{path_name}_ncu_{index}"
        cmd = [
            ncu,
            "--csv",
            "--force-overwrite",
            "--export",
            str(report_prefix),
            "--section",
            "SpeedOfLight",
            "--section",
            "LaunchStats",
            "--section",
            "Occupancy",
            "--section",
            "MemoryWorkloadAnalysis",
            "--section",
            "SchedulerStats",
            "--kernel-name",
            f"regex:{re.escape(kernel_name)}",
            "--launch-count",
            "1",
            sys.executable,
            str(Path(__file__).resolve()),
            "_workload",
            "--level",
            str(args.level),
            "--problem-id",
            str(args.problem_id),
            "--solution",
            str(Path(args.solution).resolve()),
            "--path-name",
            path_name,
        ]
        proc = _run(cmd)
        report = report_prefix.with_suffix(".ncu-rep")
        export = _run([ncu, "--import", str(report), "--csv", "--page", "raw"])
        csv_path.write_text(export.stdout, encoding="utf-8")
        artifacts.append(str(csv_path))
        artifacts.append(str(report))
        if proc.returncode == 0 and export.returncode == 0:
            records.extend(parse_ncu_csv(csv_path))
    return {"records": records, "artifacts": artifacts, "summary": summarize_deep_metrics(records)}


def cmd_doctor(args: argparse.Namespace) -> int:
    _require_compute_node()
    report = doctor_report(probe_counters=not args.no_counter_probe)
    print(json.dumps(report, indent=2))
    return 0 if report["compute_node"] and report["gpu"].get("name") == EXPECTED_GPU else 1


def cmd_profile(args: argparse.Namespace) -> int:
    _require_compute_node()
    solution = Path(args.solution).resolve()
    if not solution.is_file():
        raise FileNotFoundError(solution)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = PROFILE_ROOT / f"L{args.level}_P{args.problem_id}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    doctor = doctor_report()
    evaluation = _evaluate(args.level, args.problem_id, solution)
    profile: dict[str, Any] = {
        "schema_version": 1,
        "level": args.level,
        "problem_id": args.problem_id,
        "solution": str(solution.relative_to(REPO_ROOT)),
        "run_id": run_id,
        "environment": doctor,
        "evaluation": evaluation,
        "paths": {},
        "deep_profile": {"requested": args.deep, "triggered": False, "available": False},
    }
    if not evaluation.get("correct"):
        profile["diagnosis"] = {"primary": "incorrect", "labels": ["incorrect"], "evidence": []}
        profile["recommendations"] = ["Fix correctness before profiling performance."]
        _write_json(run_dir / "profile.json", profile)
        (run_dir / "feedback.md").write_text(render_feedback(profile), encoding="utf-8")
        return 2

    nsys = _tool_path("nsys")
    for path_name in args.paths.split(","):
        path_name = path_name.strip()
        if path_name not in {"candidate", "eager", "compile"}:
            raise ValueError(f"unknown path: {path_name}")
        path_result = (
            _profile_path_nsys(args, path_name, run_dir, nsys)
            if nsys
            else _profile_path_fallback(args, path_name, run_dir)
        )
        path_result["selected_kernels"] = select_top_kernels(path_result["timeline"])
        profile["paths"][path_name] = path_result

    triggered = bool(
        args.deep
        or (evaluation.get("speedup_eager") is not None and evaluation["speedup_eager"] < 1.05)
        or (evaluation.get("speedup_compile") is not None and evaluation["speedup_compile"] < 1.00)
        or (
            evaluation.get("runtime_regression_pct") is not None
            and evaluation["runtime_regression_pct"] > 3.0
        )
    )
    profile["deep_profile"]["triggered"] = triggered
    candidate_deep = None
    ncu = _tool_path("ncu")
    if triggered and ncu and doctor["hardware_counters"]["available"]:
        deep_paths = {}
        for path_name, path_result in profile["paths"].items():
            deep_paths[path_name] = _run_deep_profile(
                args, path_name, path_result["selected_kernels"], run_dir, ncu
            )
        profile["deep_profile"].update({"available": True, "paths": deep_paths})
        candidate_deep = deep_paths.get("candidate", {}).get("summary")
    elif triggered:
        profile["deep_profile"]["reason"] = (
            doctor["hardware_counters"].get("error")
            or "ncu is unavailable; retained Nsight Systems/PyTorch Profiler feedback."
        )

    candidate_timeline = profile["paths"].get("candidate", {}).get("timeline", {})
    profile["diagnosis"] = classify_profile(candidate_timeline, candidate_deep)
    profile["recommendations"] = recommendations(
        profile["diagnosis"], evaluation, candidate_timeline
    )
    _write_json(run_dir / "profile.json", profile)
    (run_dir / "feedback.md").write_text(render_feedback(profile), encoding="utf-8")
    print(run_dir / "feedback.md")
    return 0


def cmd_counter_probe() -> int:
    _require_compute_node()
    import torch

    x = torch.ones(1024, device="cuda")
    (x + 1).sum().item()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--no-counter-probe", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    profile = sub.add_parser("profile")
    profile.add_argument("--level", type=int, required=True)
    profile.add_argument("--problem-id", type=int, required=True)
    profile.add_argument("--solution", required=True)
    profile.add_argument("--paths", default="candidate,eager,compile")
    profile.add_argument("--deep", action="store_true")
    profile.add_argument("--run-id")
    profile.set_defaults(func=cmd_profile)

    workload = sub.add_parser("_workload")
    workload.add_argument("--level", type=int, required=True)
    workload.add_argument("--problem-id", type=int, required=True)
    workload.add_argument("--solution", required=True)
    workload.add_argument("--path-name", choices=("candidate", "eager", "compile"), required=True)
    workload.add_argument("--warmup", type=int, default=5)
    workload.add_argument("--iterations", type=int, default=1)
    workload.add_argument("--torch-profile", action="store_true")
    workload.add_argument("--output", default="/tmp/kernelbench_torch_profile.json")
    workload.set_defaults(func=_workload)

    probe = sub.add_parser("_counter_probe")
    probe.set_defaults(func=lambda _args: cmd_counter_probe())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
