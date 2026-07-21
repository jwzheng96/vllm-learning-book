import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools.source_sync.cli import main

from tests.source_sync.helpers import VALIDATED_AT, create_tutorial_repo, git


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo, self.source_sha, _ = create_tutorial_repo(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = main(list(args), repo_root=self.repo)
        return status, output.getvalue()

    def test_validate_prints_ok_and_returns_zero(self):
        status, output = self.run_cli("validate", "--profile", "contracts")
        self.assertEqual(status, 0)
        self.assertIn("OK: source contracts are valid", output)

    def test_validate_prints_each_error_and_returns_one(self):
        chapter = self.repo / "01-overview" / "01-intro.md"
        chapter.write_text(chapter.read_text() + "\n`vllm/example.py:3`\n")
        status, output = self.run_cli("validate", "--profile", "contracts")
        self.assertEqual(status, 1)
        self.assertIn("ERROR:", output)
        self.assertIn("vllm/example.py:3", output)

    def test_refresh_is_idempotent_and_rewrites_only_managed_state(self):
        chapter = self.repo / "01-overview" / "01-intro.md"
        chapter.write_text(
            chapter.read_text().replace(f"/blob/{self.source_sha}/", "/blob/" + "0" * 40 + "/")
        )
        report = self.repo / "impact.md"
        report.write_text("# Existing report\n", encoding="utf-8")
        args = (
            "refresh",
            "--candidate-sha",
            self.source_sha,
            "--validated-at",
            VALIDATED_AT,
            "--report",
            str(report),
        )
        first_status, first_output = self.run_cli(*args)
        first = {
            path: (self.repo / path).read_bytes()
            for path in ("README.md", "source.lock.json", "01-overview/01-intro.md")
        }
        second_status, second_output = self.run_cli(*args)
        second = {
            path: (self.repo / path).read_bytes()
            for path in first
        }
        self.assertEqual((first_status, second_status), (0, 0))
        self.assertEqual(first, second)
        self.assertIn(self.source_sha, first_output)
        self.assertIn(self.source_sha, second_output)
        self.assertEqual(report.read_text(), "# Existing report\n")

    def test_require_committed_detects_unstaged_submodule_move(self):
        submodule = self.repo / "vllm"
        (submodule / "new.py").write_text("value = 1\n")
        git(submodule, "add", "new.py")
        git(submodule, "commit", "-m", "new source")
        status, output = self.run_cli(
            "validate", "--profile", "contracts", "--require-committed"
        )
        self.assertEqual(status, 1)
        self.assertIn("committed gitlink does not match", output)

    def test_refresh_renders_candidate_lag_from_impact_report(self):
        submodule = self.repo / "vllm"
        (submodule / "new.py").write_text("value = 1\n")
        git(submodule, "add", "new.py")
        git(submodule, "commit", "-m", "new source")
        candidate = git(submodule, "rev-parse", "HEAD")
        report = self.repo / "impact.md"
        report.write_text(
            "# vLLM Upstream Impact Report\n\n"
            "## Summary\n\n"
            f"- Baseline: `{self.source_sha}`\n"
            f"- Candidate: `{candidate}`\n",
            encoding="utf-8",
        )

        cli_status, _ = self.run_cli(
            "refresh",
            "--candidate-sha",
            candidate,
            "--validated-at",
            VALIDATED_AT,
            "--report",
            str(report),
        )

        self.assertEqual(cli_status, 0)
        readme = (self.repo / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"Latest candidate: `{candidate}`", readme)
        self.assertIn("Candidate lag: `1` commit", readme)


if __name__ == "__main__":
    unittest.main()
