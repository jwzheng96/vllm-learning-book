import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / ".claude" / "skills" / "vllm-doctor"


class DoctorSkillSafetyTests(unittest.TestCase):
    def test_frontmatter_uses_standard_fields_and_trigger_description(self):
        text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(text.split("---", 2)[1])
        self.assertEqual(set(frontmatter), {"name", "description"})
        self.assertTrue(frontmatter["description"].startswith("Use when "))

    def test_every_cluster_mutation_requires_explicit_approval(self):
        text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn(
            "Every cluster mutation requires explicit user approval", text
        )
        self.assertNotIn("L1/L2 直接做", text)
        self.assertNotIn("L2（受控扰动）", text)

    def test_remediation_scripts_are_plan_only(self):
        for name in ("remediate_01.sh", "remediate_02.sh"):
            with self.subTest(name=name):
                text = (SKILL_DIR / "scripts" / name).read_text(encoding="utf-8")
                self.assertNotIn("--apply-l2", text)
                self.assertNotIn('eval "$cmd"', text)

    def test_remediation_assets_do_not_embed_universal_mutation_values(self):
        assets = "\n".join(
            path.read_text(encoding="utf-8")
            for path in SKILL_DIR.rglob("*")
            if path.is_file() and path.suffix in {".md", ".sh", ".py"}
        )
        for unsafe in (
            "L2（直接做）",
            "MAX_NUM_SEQS_NEW:-32",
            "ADMISSION_KV_THRESHOLD=0.85",
            "NCCL_TIMEOUT=60",
            "NCCL_BLOCKING_WAIT=1",
            "gpu_memory_utilization ≤ 0.85",
        ):
            with self.subTest(unsafe=unsafe):
                self.assertNotIn(unsafe, assets)

    def test_golden3_uses_current_vllm_metric_names(self):
        text = (SKILL_DIR / "scripts" / "golden3.sh").read_text(encoding="utf-8")
        self.assertIn("vllm:kv_cache_usage_perc", text)
        self.assertIn("vllm:prefix_cache_hits_total", text)
        self.assertIn("vllm:prefix_cache_queries_total", text)
        self.assertNotIn("vllm:gpu_cache_usage_perc", text)
        self.assertNotIn("vllm:gpu_prefix_cache_hit_rate", text)
        self.assertNotIn("vllm:request_failed_total", text)

    def test_all_skill_assets_avoid_removed_vllm_metric_names(self):
        stale_names = (
            "vllm:gpu_cache_usage_perc",
            "vllm:gpu_prefix_cache_hit_rate",
            "vllm:request_failed_total",
        )
        for path in SKILL_DIR.rglob("*"):
            if not path.is_file() or path.suffix not in {".md", ".sh", ".py"}:
                continue
            text = path.read_text(encoding="utf-8")
            for name in stale_names:
                with self.subTest(path=path.relative_to(SKILL_DIR), name=name):
                    self.assertNotIn(name, text)

    def test_kv_pressure_without_oom_evidence_routes_to_preemption(self):
        payload = {
            "ttft_p99_ms": 9000,
            "queue": 80,
            "kv_usage": 0.95,
            "throughput": 100,
            "running": 50,
            "prefix_cache_hit_rate": 0.8,
            "preempt_rate_per_sec": 0.6,
            "request_failed_rate": 0,
            "format_compliance_rate": 1,
            "oom_killed": 0,
        }
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "scripts" / "triage.py")],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "VLLM_DOCTOR_USE_EXAMPLE_THRESHOLDS": "1",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["playbook"], "01-preempt-cascade")

    def test_live_route_requires_explicit_threshold_config(self):
        payload = {
            "ttft_p99_ms": 9000,
            "queue": 80,
            "kv_usage": 0.95,
            "throughput": 100,
            "running": 50,
            "prefix_cache_hit_rate": 0.8,
            "preempt_rate_per_sec": 0.6,
        }
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("VLLM_DOCTOR_")
            and key
            not in {
                "TTFT_SLO_MS",
                "QUEUE_HIGH",
                "KV_HIGH",
                "PREEMPT_HIGH_PER_SEC",
                "PREFIX_CACHE_DROP_FROM",
                "RUNNING_LOW",
            }
        }
        result = subprocess.run(
            [sys.executable, str(SKILL_DIR / "scripts" / "triage.py")],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        routing = json.loads(result.stdout)
        self.assertEqual(routing["playbook"], "none")
        self.assertIn("missing thresholds", routing["reason"])

    def test_verify_does_not_mark_missing_evidence_resolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "doctor-missing.json"
            fixture.write_text('{"ttft_p99_ms": null}\n', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SKILL_DIR / "scripts" / "triage.py"),
                    "--verify",
                    str(fixture),
                ],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "INSUFFICIENT_EVIDENCE")

    def test_verify_only_reports_no_active_route(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "doctor-healthy.json"
            fixture.write_text(
                json.dumps(
                    {
                        "ttft_p99_ms": 100,
                        "queue": 0,
                        "kv_usage": 0.2,
                        "throughput": 10,
                        "running": 10,
                        "prefix_cache_hit_rate": 0.8,
                        "preempt_rate_per_sec": 0,
                        "request_failed_rate": None,
                        "format_compliance_rate": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SKILL_DIR / "scripts" / "triage.py"),
                    "--verify",
                    str(fixture),
                ],
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "VLLM_DOCTOR_USE_EXAMPLE_THRESHOLDS": "1",
                },
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "NO_ACTIVE_ROUTE")


if __name__ == "__main__":
    unittest.main()
