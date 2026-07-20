import shutil
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from tools.source_sync.models import SourceDirective, SourceLock
from tools.source_sync.resolver import resolve_python_symbol, resolve_reference


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "source_sync" / "source"
)


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source_root = Path(self.temp_dir.name) / "source"
        shutil.copytree(FIXTURE_ROOT, self.source_root)
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

    def test_resolves_method_start_line(self):
        result = resolve_reference(
            self.source_root,
            SourceDirective(
                PurePosixPath("vllm/example.py"), symbol="Engine.run"
            ),
            self.lock,
        )
        self.assertEqual((result.start_line, result.end_line), (7, 7))

    def test_resolves_async_method_symbol_range(self):
        text = (self.source_root / "vllm" / "example.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(resolve_python_symbol(text, "Engine.async_run"), (11, 12))

    def test_anchor_is_scoped_to_symbol(self):
        result = resolve_reference(
            self.source_root,
            SourceDirective(
                PurePosixPath("vllm/example.py"),
                symbol="Engine.run",
                anchor="budget = 8",
                span=2,
            ),
            self.lock,
        )
        self.assertEqual((result.start_line, result.end_line), (8, 9))

    def test_rejects_zero_and_multiple_anchor_matches(self):
        for anchor in ("missing = True", "return budget"):
            with self.subTest(anchor=anchor):
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    resolve_reference(
                        self.source_root,
                        SourceDirective(
                            PurePosixPath("vllm/example.py"), anchor=anchor
                        ),
                        self.lock,
                    )

    def test_rejects_missing_symbol(self):
        with self.assertRaisesRegex(ValueError, "symbol not found"):
            resolve_reference(
                self.source_root,
                SourceDirective(
                    PurePosixPath("vllm/example.py"), symbol="Engine.missing"
                ),
                self.lock,
            )

    def test_rejects_symlink_escape(self):
        outside = self.source_root.parent / "outside.py"
        outside.write_text("secret = True\n", encoding="utf-8")
        (self.source_root / "escape.py").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside source root"):
            resolve_reference(
                self.source_root,
                SourceDirective(PurePosixPath("escape.py")),
                self.lock,
            )

    def test_builds_immutable_line_url(self):
        result = resolve_reference(
            self.source_root,
            SourceDirective(
                PurePosixPath("csrc/example.cu"),
                anchor="const int block_size = 16;",
            ),
            self.lock,
        )
        self.assertEqual(
            result.url,
            "https://github.com/vllm-project/vllm/blob/"
            "4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc/"
            "csrc/example.cu#L2",
        )

    def test_builds_directory_tree_url(self):
        result = resolve_reference(
            self.source_root,
            SourceDirective(PurePosixPath("vllm")),
            self.lock,
        )
        self.assertIsNone(result.start_line)
        self.assertEqual(
            result.url,
            "https://github.com/vllm-project/vllm/tree/"
            "4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc/vllm",
        )


if __name__ == "__main__":
    unittest.main()
