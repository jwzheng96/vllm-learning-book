import json
import tempfile
import unittest
from pathlib import Path

from tools.source_sync.versions import (
    gitlink_head,
    load_source_lock,
    render_version_block,
    submodule_head,
    validate_repository,
)

from tests.source_sync.helpers import create_tutorial_repo, git


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo, self.source_sha, _ = create_tutorial_repo(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_exact_source_lock_schema(self):
        lock = load_source_lock(self.repo / "source.lock.json")
        self.assertEqual(lock.commit, self.source_sha)
        data = json.loads((self.repo / "source.lock.json").read_text())
        data["extra"] = True
        (self.repo / "source.lock.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "unknown source lock keys: extra"):
            load_source_lock(self.repo / "source.lock.json")

    def test_reads_submodule_and_committed_gitlink_sha(self):
        self.assertEqual(submodule_head(self.repo), self.source_sha)
        self.assertEqual(gitlink_head(self.repo), self.source_sha)

    def test_version_block_has_no_trailing_whitespace(self):
        block = render_version_block(
            load_source_lock(self.repo / "source.lock.json")
        )
        self.assertTrue(all(line == line.rstrip() for line in block.splitlines()))

    def test_reports_uninitialized_submodule(self):
        git(self.repo, "submodule", "deinit", "-f", "vllm")
        with self.assertRaisesRegex(ValueError, "not initialized"):
            submodule_head(self.repo)

    def test_valid_repository_has_no_contract_errors(self):
        self.assertEqual(
            validate_repository(
                self.repo, profile="contracts", require_committed=True
            ),
            (),
        )

    def test_reports_stale_managed_url(self):
        chapter = self.repo / "01-overview" / "01-intro.md"
        chapter.write_text(
            chapter.read_text().replace(f"/blob/{self.source_sha}/", "/blob/" + "0" * 40 + "/")
        )
        errors = validate_repository(self.repo, profile="contracts")
        self.assertIn("managed source URL is stale", "\n".join(errors))

    def test_reports_lock_submodule_mismatch(self):
        data = json.loads((self.repo / "source.lock.json").read_text())
        data["commit"] = "0" * 40
        (self.repo / "source.lock.json").write_text(json.dumps(data))
        errors = validate_repository(self.repo, profile="contracts")
        self.assertIn("submodule HEAD does not match source lock", "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
