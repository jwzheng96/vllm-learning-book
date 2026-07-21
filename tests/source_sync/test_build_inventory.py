import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_html
import build_pdf_epub
from tools.source_sync.inventory import (
    discover_chapter_files,
    load_curriculum,
)


ROOT = Path(__file__).resolve().parents[2]


def write_curriculum(root: Path, paths: list[str]) -> None:
    records = []
    for path in paths:
        records.extend(
            [
                "[[chapter]]",
                f'path = "{path}"',
                f'title = "{Path(path).stem}"',
                'level = "beginner"',
                'tracks = ["quickstart"]',
                'environments = ["no-gpu"]',
                'source_areas = ["vllm/v1/engine/**"]',
                "",
            ]
        )
    (root / "curriculum.toml").write_text(
        "\n".join(records), encoding="utf-8"
    )


class BuildInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.paths = [
            "01-overview/02-second.md",
            "01-overview/01-first.md",
        ]
        for path in self.paths:
            chapter = self.root / path
            chapter.parent.mkdir(parents=True, exist_ok=True)
            chapter.write_text(f"# {chapter.stem}\nbody\n", encoding="utf-8")
        (self.root / "README.md").write_text("# Book\n", encoding="utf-8")
        write_curriculum(self.root, self.paths)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_repository_inventory_matches_shared_discovery(self):
        self.assertEqual(
            [
                path.relative_to(ROOT).as_posix()
                for path in discover_chapter_files(ROOT)
            ],
            [
                chapter.path.as_posix()
                for chapter in load_curriculum(ROOT / "curriculum.toml")
            ],
        )

    def test_both_builders_follow_inventory_order(self):
        with patch.object(build_html, "SRC", self.root):
            html_paths = [
                path.relative_to(self.root).as_posix()
                for _, path in build_html.discover_files()[1:]
            ]
        with patch.object(build_pdf_epub, "SRC", self.root):
            publication_paths = [
                path.relative_to(self.root).as_posix()
                for path in build_pdf_epub.discover_files()[1:]
            ]
        self.assertEqual(html_paths, self.paths)
        self.assertEqual(publication_paths, self.paths)

    def test_missing_listed_file_is_rejected(self):
        (self.root / self.paths[0]).unlink()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            discover_chapter_files(self.root)

    def test_unlisted_chapter_is_rejected(self):
        extra = self.root / "01-overview" / "03-unlisted.md"
        extra.write_text("# unlisted\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing from curriculum"):
            discover_chapter_files(self.root)

    def test_publication_subtitle_uses_actual_count_and_lines(self):
        files = [self.root / "README.md", *discover_chapter_files(self.root)]
        combined = build_pdf_epub.combine_files(files)
        expected_lines = sum(
            len(path.read_text(encoding="utf-8").splitlines()) for path in files
        )
        self.assertIn("源码教程 · 2 章", combined)
        self.assertIn(f"{expected_lines} 行", combined)
        self.assertNotIn("15K+", combined)


if __name__ == "__main__":
    unittest.main()
