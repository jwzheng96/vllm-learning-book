import shutil
import tempfile
import unittest
from pathlib import Path

from tools.source_sync.markdown import (
    find_unmanaged_line_references,
    refresh_document,
    scan_document,
)
from tools.source_sync.models import SourceLock


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "source_sync"


class MarkdownContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_root = self.root / "source"
        shutil.copytree(FIXTURE_ROOT / "source", self.source_root)
        self.chapter_path = self.root / "chapter.md"
        self.original = (FIXTURE_ROOT / "chapter.md").read_text(encoding="utf-8")
        self.chapter_path.write_text(self.original, encoding="utf-8")
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

    def test_scans_directive_and_following_link(self):
        references = scan_document(self.chapter_path, self.original)
        self.assertEqual(len(references), 1)
        reference = references[0]
        self.assertEqual(reference.directive.path.as_posix(), "vllm/example.py")
        self.assertEqual(reference.directive.symbol, "Engine.run")
        self.assertEqual(reference.directive.anchor, "budget = 8")
        self.assertEqual(reference.current_url, (
            "https://github.com/vllm-project/vllm/blob/old/"
            "vllm/example.py#L1"
        ))
        self.assertEqual(
            self.original[reference.url_start:reference.url_end],
            reference.current_url,
        )

    def test_refresh_is_idempotent_and_changes_only_url(self):
        first = refresh_document(
            self.chapter_path, self.original, self.source_root, self.lock
        )
        second = refresh_document(
            self.chapter_path, first, self.source_root, self.lock
        )
        self.assertEqual(first, second)
        self.assertIn(
            "/blob/4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc/"
            "vllm/example.py#L8",
            first,
        )
        old_url = scan_document(self.chapter_path, self.original)[0].current_url
        new_url = scan_document(self.chapter_path, first)[0].current_url
        self.assertEqual(first, self.original.replace(old_url, new_url))

    def test_rejects_invalid_json(self):
        text = (
            '<!-- vllm-source: {"path":} -->\n'
            '[source](https://example.com/old)\n'
        )
        with self.assertRaisesRegex(ValueError, "invalid vllm-source JSON"):
            scan_document(self.chapter_path, text)

    def test_rejects_unknown_directive_key(self):
        text = (
            '<!-- vllm-source: {"path":"vllm/example.py","line":8} -->\n'
            '[source](https://example.com/old)\n'
        )
        with self.assertRaisesRegex(ValueError, "unknown vllm-source keys: line"):
            scan_document(self.chapter_path, text)

    def test_requires_link_on_next_physical_line(self):
        text = (
            '<!-- vllm-source: {"path":"vllm/example.py"} -->\n'
            '\n'
            '[source](https://example.com/old)\n'
        )
        with self.assertRaisesRegex(ValueError, "next physical line"):
            scan_document(self.chapter_path, text)

    def test_reports_unmanaged_prose_source_line(self):
        text = (
            "Read `vllm/example.py:8`.\n\n"
            "```text\ntrace.py:8\n```\n"
        )
        errors = find_unmanaged_line_references(self.chapter_path, text)
        self.assertEqual(
            errors,
            (f"{self.chapter_path}:1: vllm/example.py:8",),
        )

    def test_reports_unmanaged_source_line_inside_fence(self):
        text = "```text\nvllm/example.py:8\n```\n"
        errors = find_unmanaged_line_references(self.chapter_path, text)
        self.assertEqual(
            errors,
            (f"{self.chapter_path}:2: vllm/example.py:8",),
        )

    def test_managed_link_is_not_reported_as_legacy(self):
        refreshed = refresh_document(
            self.chapter_path, self.original, self.source_root, self.lock
        )
        self.assertEqual(
            find_unmanaged_line_references(self.chapter_path, refreshed), ()
        )


if __name__ == "__main__":
    unittest.main()
