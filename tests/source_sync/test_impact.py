import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.source_sync.impact import (
    _changed_files,
    build_impact,
    render_impact_markdown,
)

from tests.source_sync.helpers import commit_all, create_tutorial_repo


class ImpactTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo, self.baseline, _ = create_tutorial_repo(self.root)
        submodule = self.repo / "vllm"
        (submodule / "vllm" / "example.py").write_text(
            "class Engine:\n"
            "    def run(self):\n"
            "        budget = 16\n"
            "        return budget\n",
            encoding="utf-8",
        )
        engine = submodule / "vllm" / "v1" / "engine"
        engine.mkdir(parents=True)
        (engine / "core.py").write_text("class EngineCore:\n    pass\n")
        (submodule / "csrc" / "uncovered.cu").write_text("// changed\n")
        self.candidate = commit_all(submodule, "candidate")

        chapter_dir = self.repo / "01-overview"
        (chapter_dir / "02-engine.md").write_text("# Engine\n", encoding="utf-8")
        with (self.repo / "curriculum.toml").open("a", encoding="utf-8") as stream:
            stream.write(
                "\n[[chapter]]\n"
                'path = "01-overview/02-engine.md"\n'
                'title = "Engine"\n'
                'level = "intermediate"\n'
                'tracks = ["internals"]\n'
                'environments = ["no-gpu"]\n'
                'source_areas = ["vllm/v1/engine/**"]\n'
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_maps_direct_and_source_area_changes(self):
        result = build_impact(self.repo, self.baseline, self.candidate)
        self.assertEqual(
            tuple(path.as_posix() for path in result.affected_chapters),
            ("01-overview/01-intro.md", "01-overview/02-engine.md"),
        )
        self.assertIn("csrc/uncovered.cu", result.uncovered_files)
        self.assertIn("vllm/example.py", result.changed_files)

    def test_changed_file_scan_disables_rename_detection(self):
        source_root = self.repo / "vllm"
        with patch(
            "tools.source_sync.impact.run_git", return_value="vllm/example.py"
        ) as run_git:
            changed = _changed_files(source_root, self.baseline, self.candidate)

        self.assertEqual(changed, ("vllm/example.py",))
        run_git.assert_called_once_with(
            source_root,
            "diff",
            "--no-renames",
            "--name-only",
            f"{self.baseline}..{self.candidate}",
            "--",
        )

    def test_renders_deterministic_review_report(self):
        report = render_impact_markdown(
            build_impact(self.repo, self.baseline, self.candidate)
        )
        headings = [
            "## Summary",
            "## Changed Files",
            "## Affected Chapters",
            "## Unresolved Contracts",
            "## Uncovered Source Files",
            "## Human Review Checklist",
        ]
        positions = [report.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("- [ ] `01-overview/02-engine.md`", report)
        self.assertIn("Defaults and CLI flags", report)


if __name__ == "__main__":
    unittest.main()
