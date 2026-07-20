import shutil
import tempfile
import unittest
from pathlib import Path

from tools.source_sync.inventory import (
    discover_chapters,
    load_curriculum,
    load_reviews,
    validate_inventory,
)
from tools.source_sync.models import SourceLock


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "source_sync"


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        chapter_dir = self.root / "01-overview"
        chapter_dir.mkdir()
        (chapter_dir / "01-intro.md").write_text(
            "# Introduction\n", encoding="utf-8"
        )
        shutil.copy(
            FIXTURE_ROOT / "curriculum.toml", self.root / "curriculum.toml"
        )
        shutil.copy(
            FIXTURE_ROOT / "content-review.toml",
            self.root / "content-review.toml",
        )
        self.lock = SourceLock(
            schema_version=1,
            repository="https://github.com/vllm-project/vllm",
            branch="main",
            commit="4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc",
            committed_at="2026-07-17T11:19:05Z",
            validated_at="2026-07-20T15:00:00Z",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_valid_curriculum_and_reviews(self):
        chapters = load_curriculum(self.root / "curriculum.toml")
        reviews = load_reviews(self.root / "content-review.toml")
        self.assertEqual(chapters[0].path.as_posix(), "01-overview/01-intro.md")
        self.assertEqual(chapters[0].tracks, ("quickstart", "internals"))
        self.assertEqual(reviews[0].status, "pending")

    def test_discovers_only_top_level_chapter_markdown(self):
        (self.root / "README.md").write_text("# Index\n", encoding="utf-8")
        templates = self.root / "01-overview" / "templates"
        templates.mkdir()
        (templates / "report.md").write_text("# Template\n", encoding="utf-8")
        self.assertEqual(
            tuple(path.as_posix() for path in discover_chapters(self.root)),
            ("01-overview/01-intro.md",),
        )

    def test_rejects_duplicate_chapter_paths(self):
        original = (self.root / "curriculum.toml").read_text(encoding="utf-8")
        (self.root / "curriculum.toml").write_text(
            original + "\n" + original,
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate chapter path"):
            load_curriculum(self.root / "curriculum.toml")

    def test_reports_discovered_chapter_missing_from_inventory(self):
        (self.root / "01-overview" / "02-extra.md").write_text(
            "# Extra\n", encoding="utf-8"
        )
        errors = validate_inventory(self.root, self.lock, "contracts")
        self.assertIn(
            "chapter missing from curriculum.toml: 01-overview/02-extra.md",
            errors,
        )

    def test_reports_inventory_path_missing_from_disk(self):
        (self.root / "01-overview" / "01-intro.md").unlink()
        errors = validate_inventory(self.root, self.lock, "contracts")
        self.assertIn(
            "curriculum chapter does not exist: 01-overview/01-intro.md",
            errors,
        )

    def test_rejects_invalid_chapter_enums_and_empty_source_areas(self):
        text = (self.root / "curriculum.toml").read_text(encoding="utf-8")
        text = text.replace('level = "beginner"', 'level = "expert"')
        text = text.replace(
            'tracks = ["quickstart", "internals"]', 'tracks = ["unknown"]'
        )
        text = text.replace(
            'environments = ["no-gpu"]', 'environments = ["tpu"]'
        )
        text = text.replace(
            'source_areas = ["vllm/v1/engine/**"]', "source_areas = []"
        )
        (self.root / "curriculum.toml").write_text(text, encoding="utf-8")
        errors = validate_inventory(self.root, self.lock, "contracts")
        self.assertIn("invalid level 'expert'", "\n".join(errors))
        self.assertIn("invalid track 'unknown'", "\n".join(errors))
        self.assertIn("invalid environment 'tpu'", "\n".join(errors))
        self.assertIn("source_areas must not be empty", "\n".join(errors))

    def test_contracts_profile_accepts_pending_review(self):
        self.assertEqual(
            validate_inventory(self.root, self.lock, "contracts"), ()
        )

    def test_full_profile_rejects_pending_review(self):
        errors = validate_inventory(self.root, self.lock, "full")
        self.assertIn(
            "review must be complete for current source lock",
            "\n".join(errors),
        )

    def test_full_profile_accepts_current_complete_review(self):
        text = (self.root / "content-review.toml").read_text(encoding="utf-8")
        text = text.replace('status = "pending"', 'status = "reviewed"')
        text = text.replace(
            'reviewed_commit = ""',
            'reviewed_commit = "4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc"',
        )
        text = text.replace(
            'reviewed_at = ""', 'reviewed_at = "2026-07-20T15:00:00Z"'
        )
        text = text.replace(" = false", " = true")
        (self.root / "content-review.toml").write_text(text, encoding="utf-8")
        self.assertEqual(validate_inventory(self.root, self.lock, "full"), ())

    def test_rejects_unknown_profile(self):
        with self.assertRaisesRegex(ValueError, "review profile"):
            validate_inventory(self.root, self.lock, "relaxed")


if __name__ == "__main__":
    unittest.main()
