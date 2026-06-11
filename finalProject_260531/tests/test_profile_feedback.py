import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "profile_feedback.py"
SPEC = importlib.util.spec_from_file_location("profile_feedback", MODULE_PATH)
profile_feedback = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(profile_feedback)


class ProfileFeedbackTests(unittest.TestCase):
    def test_select_top_kernels_stops_at_coverage_and_cap(self):
        timeline = {
            "kernels": [
                {"name": "a", "total_us": 70},
                {"name": "b", "total_us": 20},
                {"name": "c", "total_us": 10},
            ]
        }
        self.assertEqual(
            [k["name"] for k in profile_feedback.select_top_kernels(timeline)],
            ["a", "b"],
        )
        self.assertEqual(len(profile_feedback.select_top_kernels(timeline, coverage=1.0, cap=2)), 2)

    def test_parse_ncu_csv_preserves_per_kernel_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ncu.csv"
            path.write_text(
                '"ID","Kernel Name","Metric Name","Metric Value"\n'
                '"1","kernel_a","Compute Throughput","80"\n'
                '"1","kernel_a","Memory Throughput","20"\n'
                '"2","kernel_b","Compute Throughput","10"\n',
                encoding="utf-8",
            )
            records = profile_feedback.parse_ncu_csv(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["metrics"]["Compute Throughput"], 80.0)
        self.assertEqual(records[1]["metrics"]["Compute Throughput"], 10.0)

    def test_parse_ncu_raw_wide_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ncu.csv"
            path.write_text(
                '"ID","Kernel Name","sm__throughput.avg.pct_of_peak_sustained_elapsed",'
                '"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"\n'
                '"","","%","%"\n'
                '"0","gemm","81.5","33.0"\n',
                encoding="utf-8",
            )
            records = profile_feedback.parse_ncu_csv(path)
        summary = profile_feedback.summarize_deep_metrics(records)
        self.assertEqual(records[0]["metrics"]["sm__throughput.avg.pct_of_peak_sustained_elapsed"], 81.5)
        self.assertEqual(summary["kernels"][0]["compute_pct"], 81.5)
        self.assertEqual(summary["kernels"][0]["memory_pct"], 33.0)

    def test_metric_does_not_match_sm_inside_sustained(self):
        metrics = {
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 39.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 94.0,
        }
        self.assertEqual(profile_feedback._metric(metrics, ("sm", "throughput", "avg", "pct")), 94.0)

    def test_summary_prefers_exact_raw_sm_throughput(self):
        records = [{
            "kernel": "gemm",
            "launch_id": "0",
            "metrics": {
                "sm__instruction_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed": 0.0,
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 94.0,
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 39.0,
            },
        }]
        summary = profile_feedback.summarize_deep_metrics(records)
        self.assertEqual(summary["kernels"][0]["compute_pct"], 94.0)

    def test_parse_nsys_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nsys.csv"
            path.write_text(
                '"Time (%)","Total Time (ns)","Instances","Avg (ns)","Name"\n'
                '"75","300000","3","100000","kernel_a"\n'
                '"25","100000","1","100000","kernel_b"\n',
                encoding="utf-8",
            )
            result = profile_feedback.parse_nsys_csv(path)
        self.assertEqual(result["kernel_count"], 4)
        self.assertEqual(result["gpu_total_us"], 400.0)
        self.assertEqual(result["kernels"][0]["name"], "kernel_a")

    def test_parse_nsys_csv_skips_status_preamble(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nsys.csv"
            path.write_text(
                "Generating SQLite file report.sqlite\n"
                "Processing [report.sqlite] with [cuda_gpu_kern_sum.py]...\n"
                "Time (%),Total Time (ns),Instances,Name\n"
                "100,250000,2,kernel_a\n",
                encoding="utf-8",
            )
            result = profile_feedback.parse_nsys_csv(path)
        self.assertEqual(result["kernel_count"], 2)
        self.assertEqual(result["gpu_total_us"], 250.0)

    def test_find_col_ignores_nullable_csv_column(self):
        self.assertEqual(
            profile_feedback._find_col([None, "Kernel Name"], ("Name", "Kernel Name")),
            "Kernel Name",
        )

    def test_parse_nsys_api_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api.csv"
            path.write_text(
                '"Time (%)","Total Time (ns)","Num Calls","Name"\n'
                '"60","60000","3","cudaLaunchKernel"\n'
                '"40","40000","2","cudaDeviceSynchronize"\n',
                encoding="utf-8",
            )
            result = profile_feedback.parse_nsys_api_csv(path)
        self.assertEqual(result["cuda_api_total_us"], 100.0)
        self.assertEqual(result["cuda_api_calls"], 5)

    def test_doctor_missing_tools_fallback(self):
        with (
            mock.patch.object(profile_feedback.socket, "gethostname", return_value="gn1001.twcc.ai"),
            mock.patch.object(profile_feedback, "_tool_path", return_value=None),
            mock.patch.object(
                profile_feedback,
                "_gpu_info",
                return_value={"available": True, "name": profile_feedback.EXPECTED_GPU},
            ),
        ):
            report = profile_feedback.doctor_report()
        self.assertTrue(report["compute_node"])
        self.assertFalse(report["tools"]["ncu"]["available"])
        self.assertFalse(report["hardware_counters"]["probed"])

    def test_login_node_guard(self):
        with mock.patch.object(profile_feedback.socket, "gethostname", return_value="un-ln01.twcc.ai"):
            with self.assertRaises(RuntimeError):
                profile_feedback._require_compute_node()

    def test_diagnosis_boundaries(self):
        launch = profile_feedback.classify_profile(
            {"kernel_count": 10, "kernels": [{"total_us": 190}]}, None
        )
        self.assertEqual(launch["primary"], "launch-bound")

        memory = profile_feedback.classify_profile(
            {"kernel_count": 1, "kernels": [{"total_us": 100}]},
            {"kernels": [{"memory_pct": 70, "compute_pct": 49, "occupancy_pct": 80}]},
        )
        self.assertEqual(memory["primary"], "memory-bound")

        compute = profile_feedback.classify_profile(
            {"kernel_count": 1, "kernels": [{"total_us": 100}]},
            {"kernels": [{"memory_pct": 49, "compute_pct": 70, "occupancy_pct": 80}]},
        )
        self.assertEqual(compute["primary"], "compute-bound")

        latency = profile_feedback.classify_profile(
            {"kernel_count": 1, "kernels": [{"total_us": 100}]},
            {"kernels": [{"memory_pct": 20, "compute_pct": 20, "occupancy_pct": 39, "eligible_warps": 0.5}]},
        )
        self.assertEqual(latency["primary"], "occupancy/latency-bound")

        conflict = profile_feedback.classify_profile(
            {"kernel_count": 1, "kernels": [{"total_us": 100}]},
            {"kernels": [{"memory_pct": 60, "compute_pct": 60, "branch_efficiency_pct": 79, "shared_conflicts": 1}]},
        )
        self.assertEqual(conflict["primary"], "divergence/conflict-risk")

    def test_render_feedback_without_deep_metrics(self):
        profile = {
            "level": 2,
            "problem_id": 40,
            "evaluation": {
                "correct": True,
                "kernel_ms": 39.0,
                "speedup_eager": 1.04,
                "speedup_compile": 0.97,
            },
            "paths": {
                "candidate": {
                    "timeline": {
                        "kernel_count": 2,
                        "gpu_total_us": 39000,
                        "kernels": [{"name": "gemm", "total_us": 38000}],
                    }
                }
            },
            "deep_profile": {"available": False},
            "diagnosis": {"primary": "insufficient-counter-data", "evidence": []},
            "recommendations": ["Keep measuring."],
        }
        text = profile_feedback.render_feedback(profile)
        self.assertIn("L2/P40", text)
        self.assertIn("insufficient-counter-data", text)


if __name__ == "__main__":
    unittest.main()
