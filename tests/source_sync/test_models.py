import unittest
from pathlib import PurePosixPath

from tools.source_sync.models import SourceDirective, SourceLock


class SourceLockTests(unittest.TestCase):
    def test_rejects_non_full_sha(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            SourceLock(
                schema_version=1,
                repository="https://github.com/vllm-project/vllm",
                branch="main",
                commit="4c6e2e4",
                committed_at="2026-07-17T11:19:05Z",
                validated_at="2026-07-20T15:00:00Z",
            )

    def test_accepts_full_lowercase_sha(self):
        lock = SourceLock(
            schema_version=1,
            repository="https://github.com/vllm-project/vllm",
            branch="main",
            commit="4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc",
            committed_at="2026-07-17T11:19:05Z",
            validated_at="2026-07-20T15:00:00Z",
        )
        self.assertEqual(lock.short_commit, "4c6e2e4")


class SourceDirectiveTests(unittest.TestCase):
    def test_requires_a_relative_posix_path(self):
        for bad_path in ("/tmp/source.py", "../source.py", "vllm/../secret"):
            with self.subTest(path=bad_path):
                with self.assertRaisesRegex(ValueError, "relative source path"):
                    SourceDirective(path=PurePosixPath(bad_path))

    def test_rejects_non_positive_span(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SourceDirective(path=PurePosixPath("vllm/a.py"), span=0)


if __name__ == "__main__":
    unittest.main()
