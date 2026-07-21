import os
import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class WorkflowTests(unittest.TestCase):
    def test_all_workflows_parse_and_pin_checkout(self):
        for name in (
            "validate.yml",
            "sync-upstream.yml",
            "pages.yml",
            "gpu-validation.yml",
        ):
            with self.subTest(name=name):
                path = ROOT / ".github" / "workflows" / name
                text = path.read_text(encoding="utf-8")
                data = yaml.load(text, Loader=yaml.BaseLoader)
                self.assertIsInstance(data, dict)
                self.assertIn("on", data)
                self.assertIn("jobs", data)
                self.assertIn("actions/checkout@v4", text)
                self.assertIn("submodules: true", text)

    def test_sync_workflow_cannot_merge_or_deploy(self):
        path = ROOT / ".github" / "workflows" / "sync-upstream.yml"
        text = path.read_text(encoding="utf-8")
        self.assertIn("--force-with-lease", text)
        self.assertIn("merge-base --is-ancestor", text)
        self.assertNotIn("gh pr merge", text)
        self.assertNotIn("actions/deploy-pages", text)
        self.assertNotIn("HEAD:refs/heads/main", text)

    def test_ci_and_pages_use_full_semantic_gate(self):
        for name in ("validate.yml", "sync-upstream.yml", "pages.yml"):
            with self.subTest(name=name):
                text = (ROOT / ".github" / "workflows" / name).read_text(
                    encoding="utf-8"
                )
                self.assertIn("--profile full", text)
                self.assertNotIn("--profile contracts", text)

    def test_pages_is_gated_by_committed_full_review(self):
        path = ROOT / ".github" / "workflows" / "pages.yml"
        text = path.read_text(encoding="utf-8")
        validate = text.index("--profile full --require-committed")
        build = text.index("python3 build_html.py")
        upload = text.index("uses: actions/upload-pages-artifact")
        self.assertLess(validate, build)
        self.assertLess(build, upload)

    def test_gpu_validation_is_manual_fail_closed_and_archives_evidence(self):
        path = ROOT / ".github" / "workflows" / "gpu-validation.yml"
        text = path.read_text(encoding="utf-8")
        data = yaml.load(text, Loader=yaml.BaseLoader)

        self.assertEqual(set(data["on"]), {"workflow_dispatch"})
        inputs = data["on"]["workflow_dispatch"]["inputs"]
        self.assertIn("model_id", inputs)
        self.assertEqual(
            inputs["tensor_parallel_size"]["options"], ["1", "2", "4", "8"]
        )

        self.assertEqual(len(data["jobs"]), 1)
        job = next(iter(data["jobs"].values()))
        self.assertEqual(
            job["runs-on"], ["self-hosted", "linux", "x64", "nvidia-gpu"]
        )

        run_steps = [step for step in job["steps"] if "run" in step]
        self.assertEqual(len(run_steps), 1)
        self.assertIn("scripts/gpu-validation.sh", run_steps[0]["run"])
        self.assertIn("secrets.VLLM_API_KEY", run_steps[0]["env"]["VLLM_API_KEY"])
        self.assertNotIn("pip install", text)
        self.assertNotIn("apt-get", text)

        upload_steps = [
            step
            for step in job["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        ]
        self.assertEqual(len(upload_steps), 1)
        self.assertEqual(upload_steps[0]["if"], "always()")

    def test_gpu_script_rejects_option_like_model_id(self):
        script = ROOT / "scripts" / "gpu-validation.sh"
        result = subprocess.run(
            ["bash", str(script), "--help", "1"],
            env={**os.environ, "VLLM_API_KEY": "test-only-secret"},
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("MODEL_ID must match", result.stderr)


if __name__ == "__main__":
    unittest.main()
