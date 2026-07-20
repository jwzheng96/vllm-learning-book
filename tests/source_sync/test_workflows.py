import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class WorkflowTests(unittest.TestCase):
    def test_all_workflows_parse_and_pin_checkout(self):
        for name in ("validate.yml", "sync-upstream.yml", "pages.yml"):
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

    def test_pages_is_gated_by_committed_contracts(self):
        path = ROOT / ".github" / "workflows" / "pages.yml"
        text = path.read_text(encoding="utf-8")
        validate = text.index("--profile contracts --require-committed")
        build = text.index("python3 build_html.py")
        upload = text.index("uses: actions/upload-pages-artifact")
        self.assertLess(validate, build)
        self.assertLess(build, upload)


if __name__ == "__main__":
    unittest.main()
