# vLLM Source Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed source contract system that updates the local vLLM main branch and the tutorial submodule to official main, resolves semantic source anchors into immutable line links, reports affected chapters, and proposes weekly upgrade PRs.

**Architecture:** A Python 3.9-compatible package under `tools/source_sync/` owns four boundaries: semantic resolution, Markdown rewriting, repository metadata validation, and upstream impact analysis. The tutorial keeps a pinned submodule plus `source.lock.json`; machine-readable chapter and review inventories make semantic review visible. GitHub workflows call the same CLI used locally and never publish a candidate that fails validation.

**Tech Stack:** Python 3.9+, standard library (`ast`, `argparse`, `dataclasses`, `json`, `pathlib`, `subprocess`), `tomli` compatibility package, PyYAML for workflow syntax tests, `unittest`, Git, GitHub Actions, GitHub CLI.

## Global Constraints

- All project commands, tests, formatters, builds, installs, and mutating Git commands run on `rlocal`; never run them on the sshfs client.
- vLLM source commands run in `/Users/zjw/Documents/LLM/inference-engine/vllm/vllm`; tutorial commands run in `/Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning`.
- File edits use `apply_patch` against the mounted workspace; generated artifacts are produced only by project commands on `rlocal`.
- Do not modify upstream vLLM source. Only fetch official `origin/main`, fast-forward the local `main` ref, and move the tutorial submodule gitlink.
- Preserve `bugfix/parallel-config-size-validation` and commits `004b8601c`, `d1cd0162d`, and `090fd61d8`; verify all three remain reachable after the main ref update.
- Track `https://github.com/vllm-project/vllm` branch `main`; the final SHA is resolved at execution time and must be a 40-character lowercase hexadecimal commit reachable from official `main`.
- Python tooling must execute on the existing Python 3.9.6 environment and on GitHub Actions Python 3.12.
- Do not use fuzzy symbol matching, occurrence numbers, `eval`, or paths outside the submodule root.
- Do not auto-merge source update PRs and do not publish failed candidates.
- Do not hand-edit `_site/` or `vllm-learning-html/`.
- The source-sync design is `docs/superpowers/specs/2026-07-20-source-sync-design.md` and is authoritative for behavior.

---

## File Map

- `requirements-docs.txt`: deterministic documentation and source-sync Python dependencies.
- `tools/__init__.py`: marks repository tools as an importable package.
- `tools/source_sync/__init__.py`: exports the public source-sync API.
- `tools/source_sync/models.py`: immutable source lock, directive, resolution, chapter, review, and impact data models.
- `tools/source_sync/resolver.py`: path confinement, Python AST symbol lookup, exact anchor lookup, and GitHub URL generation.
- `tools/source_sync/markdown.py`: directive scanning, managed-link rewriting, and unmanaged line-reference detection.
- `tools/source_sync/inventory.py`: `curriculum.toml` and `content-review.toml` loading and repository coverage validation.
- `tools/source_sync/versions.py`: submodule/lock/README consistency and safe Git command wrappers.
- `tools/source_sync/impact.py`: upstream diff-to-chapter mapping and Markdown report rendering.
- `tools/source_sync/cli.py`: `validate`, `refresh`, `impact`, and `check-upstream` command-line interface.
- `tools/source_sync/__main__.py`: supports `python3 -m tools.source_sync`.
- `tests/source_sync/`: standard-library unit and integration tests.
- `tests/fixtures/source_sync/`: tiny synthetic source tree and Markdown fixtures.
- `source.lock.json`: explicit repository, branch, source commit, upstream time, and validation time.
- `curriculum.toml`: one inventory record for every chapter.
- `content-review.toml`: one review record for every chapter, initially `pending` and later completed by the curriculum plan.
- `docs/source-sync.md`: author-facing directive, CLI, failure, and upgrade-runbook documentation.
- `.github/workflows/validate.yml`: pull-request source-contract and documentation build checks.
- `.github/workflows/sync-upstream.yml`: weekly/manual official-main candidate PR workflow.
- `.github/workflows/pages.yml`: deploys only after full validation and submodule checkout.

---

### Task 1: Dependency Floor and Immutable Data Contracts

**Files:**
- Create: `requirements-docs.txt`
- Create: `tools/__init__.py`
- Create: `tools/source_sync/__init__.py`
- Create: `tools/source_sync/models.py`
- Create: `tests/__init__.py`
- Create: `tests/source_sync/__init__.py`
- Create: `tests/source_sync/test_models.py`

**Interfaces:**
- Consumes: Python 3.9.6 on `rlocal` and Python 3.12 in CI.
- Produces: `SourceLock`, `SourceDirective`, `ResolvedReference`, `MarkdownReference`, `Chapter`, `Review`, and `ImpactResult` dataclasses imported by every later task.

- [ ] **Step 1: Add the failing model tests**

Create `tests/source_sync/test_models.py` with these exact cases:

```python
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
```

- [ ] **Step 2: Run the test to prove the package is missing**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_models -v'
```

Expected: `ModuleNotFoundError: No module named 'tools.source_sync'`.

- [ ] **Step 3: Add dependencies and implement the data contracts**

Create `requirements-docs.txt`:

```text
Markdown>=3.5,<4
Pygments>=2.17,<3
tomli>=2.0,<3; python_version < "3.11"
PyYAML>=6,<7
```

Implement `tools/source_sync/models.py` with frozen dataclasses and validation. The public shapes must be:

```python
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SourceLock:
    schema_version: int
    repository: str
    branch: str
    commit: str
    committed_at: str
    validated_at: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("source lock schema_version must be 1")
        if not FULL_SHA_RE.fullmatch(self.commit):
            raise ValueError("source lock commit must be a 40-character lowercase SHA")

    @property
    def short_commit(self) -> str:
        return self.commit[:7]


@dataclass(frozen=True)
class SourceDirective:
    path: PurePosixPath
    symbol: Optional[str] = None
    anchor: Optional[str] = None
    span: int = 1

    def __post_init__(self) -> None:
        path_text = self.path.as_posix()
        if self.path.is_absolute() or ".." in self.path.parts or path_text in ("", "."):
            raise ValueError("path must be a confined relative source path")
        if not isinstance(self.span, int) or isinstance(self.span, bool) or self.span < 1:
            raise ValueError("span must be a positive integer")


@dataclass(frozen=True)
class ResolvedReference:
    directive: SourceDirective
    source_path: Path
    start_line: Optional[int]
    end_line: Optional[int]
    url: str


@dataclass(frozen=True)
class MarkdownReference:
    markdown_path: Path
    directive: SourceDirective
    directive_start: int
    link_start: int
    url_start: int
    url_end: int
    current_url: str


@dataclass(frozen=True)
class Chapter:
    path: PurePosixPath
    title: str
    level: str
    tracks: Tuple[str, ...]
    environments: Tuple[str, ...]
    source_areas: Tuple[str, ...]


@dataclass(frozen=True)
class Review:
    path: PurePosixPath
    status: str
    reviewed_commit: str
    reviewed_at: str
    source_contracts: bool
    commands_checked: bool
    metrics_checked: bool
    diagrams_checked: bool
    hardware_verified: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ImpactResult:
    baseline: str
    candidate: str
    changed_files: Tuple[str, ...]
    affected_chapters: Tuple[PurePosixPath, ...]
    uncovered_files: Tuple[str, ...]
    unresolved: Tuple[str, ...] = field(default_factory=tuple)
```

Export these names from `tools/source_sync/__init__.py`; keep `tools/__init__.py`, `tests/__init__.py`, and `tests/source_sync/__init__.py` empty.

- [ ] **Step 4: Install docs dependencies and run the model tests**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m pip install -r requirements-docs.txt && python3 -m unittest tests.source_sync.test_models -v'
```

Expected: 4 tests pass on Python 3.9.6.

- [ ] **Step 5: Commit the data-contract slice**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add requirements-docs.txt tools/__init__.py tools/source_sync/__init__.py tools/source_sync/models.py tests/__init__.py tests/source_sync/__init__.py tests/source_sync/test_models.py && git commit -m "feat: define source sync data contracts"'
```

Expected: one commit containing only the listed files.

---

### Task 2: Confined Semantic Source Resolver

**Files:**
- Create: `tools/source_sync/resolver.py`
- Create: `tests/source_sync/test_resolver.py`
- Create: `tests/fixtures/source_sync/source/vllm/example.py`
- Create: `tests/fixtures/source_sync/source/csrc/example.cu`

**Interfaces:**
- Consumes: `SourceDirective`, `SourceLock`.
- Produces: `resolve_reference(source_root: Path, directive: SourceDirective, lock: SourceLock) -> ResolvedReference` and `resolve_python_symbol(text: str, symbol: str) -> tuple[int, int]`.

- [ ] **Step 1: Create resolver fixtures and failing tests**

Create `tests/fixtures/source_sync/source/vllm/example.py` with exact line placement:

```python
def top_level():
    budget = 8
    return budget


class Engine:
    def run(self):
        budget = 8
        return budget

    async def async_run(self):
        return budget
```

Create `tests/fixtures/source_sync/source/csrc/example.cu`:

```cuda
// Source-sync fixture.
const int block_size = 16;
```

Add tests that assert:

```python
class ResolverTests(unittest.TestCase):
    def test_resolves_method_start_line(self):
        result = resolve_reference(
            self.source_root,
            SourceDirective(PurePosixPath("vllm/example.py"), symbol="Engine.run"),
            self.lock,
        )
        self.assertEqual((result.start_line, result.end_line), (7, 7))

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
                        SourceDirective(PurePosixPath("vllm/example.py"), anchor=anchor),
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
```

- [ ] **Step 2: Run the resolver tests and verify failure**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_resolver -v'
```

Expected: import failure for `tools.source_sync.resolver`.

- [ ] **Step 3: Implement exact symbol and anchor resolution**

Implement `resolve_python_symbol` by walking `ast.Module.body`, then each matching `ClassDef`, `FunctionDef`, or `AsyncFunctionDef` body for the dot-separated name. Reject missing and duplicate names. Use `lineno` and `end_lineno`, but link to only the definition line unless an anchor is present.

Implement `resolve_reference` with this sequence:

```python
def resolve_reference(source_root, directive, lock):
    root = source_root.resolve()
    source_path = (root / directive.path.as_posix()).resolve()
    if os.path.commonpath((str(root), str(source_path))) != str(root):
        raise ValueError(f"source path is outside source root: {directive.path}")
    if not source_path.exists():
        raise ValueError(f"source path does not exist: {directive.path}")
    if source_path.is_dir():
        if directive.symbol or directive.anchor:
            raise ValueError("directories cannot have symbol or anchor fields")
        return ResolvedReference(directive, source_path, None, None, _tree_url(lock, directive.path))

    lines = source_path.read_text(encoding="utf-8").splitlines()
    scope_start, scope_end = 1, len(lines)
    if directive.symbol:
        if source_path.suffix != ".py":
            raise ValueError("symbol is supported only for Python files")
        scope_start, scope_end = resolve_python_symbol("\n".join(lines), directive.symbol)

    start_line = scope_start
    if directive.anchor:
        needle = directive.anchor.strip()
        matches = [
            number
            for number in range(scope_start, scope_end + 1)
            if lines[number - 1].strip() == needle
        ]
        if len(matches) != 1:
            raise ValueError(
                f"anchor must occur exactly once in scope; found {len(matches)}: {needle}"
            )
        start_line = matches[0]
    end_line = start_line + directive.span - 1
    if end_line > scope_end:
        raise ValueError("span exceeds the resolved source scope")
    return ResolvedReference(
        directive,
        source_path,
        start_line,
        end_line,
        _blob_url(lock, directive.path, start_line, end_line),
    )
```

Use `urllib.parse.urlsplit(lock.repository)` to derive `https://github.com/vllm-project/vllm`, reject non-HTTPS/non-GitHub repositories, and generate `#Lx` or `#Lx-Ly` fragments.

- [ ] **Step 4: Run resolver tests**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_resolver -v'
```

Expected: all resolver tests pass, including symlink confinement.

- [ ] **Step 5: Commit the resolver slice**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add tools/source_sync/resolver.py tests/source_sync/test_resolver.py tests/fixtures/source_sync/source/vllm/example.py tests/fixtures/source_sync/source/csrc/example.cu && git commit -m "feat: resolve semantic source anchors"'
```

---

### Task 3: Markdown Contract Scanner and Idempotent Rewriter

**Files:**
- Create: `tools/source_sync/markdown.py`
- Create: `tests/source_sync/test_markdown.py`
- Create: `tests/fixtures/source_sync/chapter.md`

**Interfaces:**
- Consumes: `SourceDirective`, `SourceLock`, `resolve_reference()`.
- Produces: `scan_document(path: Path, text: str) -> tuple[MarkdownReference, ...]`, `refresh_document(path: Path, text: str, source_root: Path, lock: SourceLock) -> str`, and `find_unmanaged_line_references(path: Path, text: str) -> tuple[str, ...]`.

- [ ] **Step 1: Add failing Markdown contract tests**

Test a directive followed immediately by a Markdown link, invalid JSON, a directive without a following link, duplicate refresh, a fenced-code exemption, and an unmanaged prose reference. The idempotence assertion must be:

```python
first = refresh_document(path, original, self.source_root, self.lock)
second = refresh_document(path, first, self.source_root, self.lock)
self.assertEqual(first, second)
self.assertIn(
    "/blob/4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc/"
    "vllm/example.py#L8",
    first,
)
```

The unmanaged-reference fixture must contain prose `` `vllm/example.py:8` `` and fenced code `trace.py:8`; only the prose reference is reported because the second value is not a vLLM/csrc source path. A vLLM path inside a fenced code block is still a source citation and must be reported unless converted to a semantic contract outside the fence.

- [ ] **Step 2: Verify the Markdown tests fail**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_markdown -v'
```

Expected: import failure for `tools.source_sync.markdown`.

- [ ] **Step 3: Implement strict directive parsing and URL-only rewriting**

Use a line-oriented scanner rather than a single regex over the full document:

```python
DIRECTIVE_PREFIX = "<!-- vllm-source: "
DIRECTIVE_SUFFIX = " -->"
MARKDOWN_LINK_RE = re.compile(r"^(?P<prefix>\s*\[[^\n]+\]\()(?P<url>[^)]+)(?P<suffix>\)\s*)$")
LEGACY_LINE_RE = re.compile(r"(?<![A-Za-z0-9_])(?:vllm/|csrc/)[A-Za-z0-9_./{}*-]+:\d+(?:[-,]\s*\d+)*")
```

For every directive line:

1. Parse only the JSON between the fixed prefix and suffix with `json.loads`.
2. Reject unknown keys; accepted keys are exactly `path`, `symbol`, `anchor`, and `span`.
3. Require the next physical line to be one Markdown link and record URL byte offsets.
4. Convert `path` to `PurePosixPath` and construct `SourceDirective`.
5. Preserve the directive and link label byte-for-byte; replace only the link URL with the resolver URL.

`find_unmanaged_line_references` must mask managed link lines before applying `LEGACY_LINE_RE`, but must scan both prose and fenced code so source citations cannot hide in examples. It must return `path:line: matched text` strings sorted by source position.

- [ ] **Step 4: Run Markdown tests and the combined unit suite**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_markdown -v && python3 -m unittest discover -s tests/source_sync -v'
```

Expected: all model, resolver, and Markdown tests pass.

- [ ] **Step 5: Commit the Markdown contract slice**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add tools/source_sync/markdown.py tests/source_sync/test_markdown.py tests/fixtures/source_sync/chapter.md && git commit -m "feat: refresh managed source links"'
```

---

### Task 4: Curriculum Inventory and Review Ledger Validation

**Files:**
- Create: `tools/source_sync/inventory.py`
- Create: `tests/source_sync/test_inventory.py`
- Create: `tests/fixtures/source_sync/curriculum.toml`
- Create: `tests/fixtures/source_sync/content-review.toml`

**Interfaces:**
- Consumes: `Chapter`, `Review`, `SourceLock`.
- Produces: `load_curriculum(path: Path) -> tuple[Chapter, ...]`, `load_reviews(path: Path) -> tuple[Review, ...]`, `discover_chapters(repo_root: Path) -> tuple[PurePosixPath, ...]`, and `validate_inventory(repo_root: Path, lock: SourceLock, profile: str) -> tuple[str, ...]`.

- [ ] **Step 1: Add failing inventory tests**

Tests must prove:

- Python 3.9 imports `tomli`; Python 3.11+ imports `tomllib` through the same `_toml` alias.
- duplicate chapter paths fail;
- a Markdown chapter missing from `curriculum.toml` fails;
- a TOML record pointing to a missing Markdown file fails;
- invalid `level`, `track`, or `environment` fails;
- empty `source_areas` fails;
- `profile="contracts"` accepts `status="pending"` with an empty `reviewed_commit`;
- `profile="full"` requires `status="reviewed"`, current lock SHA, and all four review booleans true.

Use these allowed values:

```python
LEVELS = frozenset({"beginner", "intermediate", "advanced"})
TRACKS = frozenset({"quickstart", "internals", "production", "interview"})
ENVIRONMENTS = frozenset({"no-gpu", "cpu", "nvidia-gpu", "multi-gpu"})
REVIEW_PROFILES = frozenset({"contracts", "full"})
```

- [ ] **Step 2: Run inventory tests and verify failure**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_inventory -v'
```

Expected: import failure for `tools.source_sync.inventory`.

- [ ] **Step 3: Implement strict TOML loading and one-to-one coverage**

Use:

```python
try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml
```

`discover_chapters` must scan only `01-overview` through `09-advanced-features`, include every `*.md`, exclude `README.md`, `docs/`, `_site/`, and the submodule, and return sorted POSIX paths.

`validate_inventory` returns every error instead of stopping at the first. It compares discovered paths, curriculum paths, and review paths as sets; validates enums and non-empty source areas; then applies profile-specific review rules. Sort errors by chapter path so CI output is deterministic.

- [ ] **Step 4: Run inventory and combined tests**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_inventory -v && python3 -m unittest discover -s tests/source_sync -v'
```

Expected: all tests pass on Python 3.9.6.

- [ ] **Step 5: Commit inventory validation**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add tools/source_sync/inventory.py tests/source_sync/test_inventory.py tests/fixtures/source_sync/curriculum.toml tests/fixtures/source_sync/content-review.toml && git commit -m "feat: validate curriculum review coverage"'
```

---

### Task 5: Version Consistency, Impact Mapping, and CLI

**Files:**
- Create: `tools/source_sync/versions.py`
- Create: `tools/source_sync/impact.py`
- Create: `tools/source_sync/cli.py`
- Create: `tools/source_sync/__main__.py`
- Create: `tests/source_sync/test_versions.py`
- Create: `tests/source_sync/test_impact.py`
- Create: `tests/source_sync/test_cli.py`

**Interfaces:**
- Consumes: all previous source-sync APIs, repository Git state, `curriculum.toml`, and `content-review.toml`.
- Produces: CLI commands `validate`, `refresh`, `impact`, and `check-upstream`; `validate_repository(repo_root: Path, profile: str, require_committed: bool) -> tuple[str, ...]`; `build_impact(...) -> ImpactResult`; `render_impact_markdown(result: ImpactResult) -> str`.

- [ ] **Step 1: Add failing version, impact, and CLI tests**

Use temporary Git repositories in tests. Cover:

1. `load_source_lock` rejects missing/extra JSON fields and creates `SourceLock` for the exact schema.
2. `submodule_head` returns a full SHA and reports an uninitialized submodule clearly.
3. `gitlink_head` parses mode `160000` and rejects a normal directory.
4. `validate_repository(..., require_committed=True)` reports lock/submodule/gitlink disagreement.
5. `build_impact` maps a changed file to chapters through both a direct directive and `source_areas` glob.
6. changed vLLM files with no matching chapter appear in `uncovered_files`.
7. CLI exits `0` with `OK` on success and `1` with one `ERROR:` line per failure.
8. `refresh` is idempotent and updates only managed URLs, lock data, the delimited README version block, and its requested report path.

The README version block delimiters are exact:

```markdown
<!-- vllm-version:start -->
... generated content ...
<!-- vllm-version:end -->
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_versions tests.source_sync.test_impact tests.source_sync.test_cli -v'
```

Expected: import failures for the three missing modules.

- [ ] **Step 3: Implement safe Git wrappers and version validation**

All Git calls use one helper:

```python
def run_git(cwd, *args):
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()
```

`validate_repository` must aggregate:

- lock schema and official repository/branch;
- submodule HEAD equals lock commit;
- optional committed gitlink equals lock commit;
- every directive resolves and every managed URL equals the resolved URL;
- no unmanaged prose line reference remains;
- inventory/review profile validation;
- README version block contains the lock full SHA, short SHA, upstream time, and validation time.

- [ ] **Step 4: Implement deterministic impact mapping and report rendering**

Use `git -C vllm diff --name-only BASELINE_SHA..CANDIDATE_SHA` for changed files. A chapter is affected when a changed file is directly cited or matches a `source_areas` POSIX glob. Render headings in this exact order: Summary, Changed Files, Affected Chapters, Unresolved Contracts, Uncovered Source Files, Human Review Checklist. Sort every list.

The review checklist contains one unchecked item per affected chapter and these fixed global items:

```markdown
- [ ] Defaults and CLI flags still match the candidate source.
- [ ] Metrics names and meanings still match the candidate source.
- [ ] Mermaid flows still match the candidate request path.
- [ ] Commands and rollback instructions still use supported interfaces.
- [ ] No GPU result is labeled as measured without a current hardware record.
```

- [ ] **Step 5: Implement the CLI**

The parser must expose:

```text
python3 -m tools.source_sync validate [--profile contracts|full] [--require-committed]
python3 -m tools.source_sync refresh --candidate-sha SHA --validated-at ISO8601 [--report PATH]
python3 -m tools.source_sync impact --baseline SHA --candidate SHA --output PATH
python3 -m tools.source_sync check-upstream
```

Defaults: `validate --profile full`; repository root is derived from `tools/source_sync/cli.py`, not the current shell directory. `refresh` validates the SHA with `FULL_SHA_RE`, requires submodule HEAD to equal it, obtains upstream commit time with `git show -s --format=%cI`, rewrites source links, writes `source.lock.json` atomically through a sibling temporary file, refreshes the README block, and runs contracts validation before returning.

`check-upstream` runs `git ls-remote https://github.com/vllm-project/vllm.git refs/heads/main`, prints JSON containing `baseline`, `candidate`, and `changed`, and never writes files.

- [ ] **Step 6: Run CLI tests and the complete source-sync suite**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_versions tests.source_sync.test_impact tests.source_sync.test_cli -v && python3 -m unittest discover -s tests/source_sync -v'
```

Expected: all tests pass with no network access.

- [ ] **Step 7: Commit the CLI slice**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add tools/source_sync/versions.py tools/source_sync/impact.py tools/source_sync/cli.py tools/source_sync/__main__.py tests/source_sync/test_versions.py tests/source_sync/test_impact.py tests/source_sync/test_cli.py && git commit -m "feat: add source sync validation cli"'
```

---

### Task 6: Safely Update Official Main and Bootstrap Repository Contracts

**Files:**
- Create: `source.lock.json`
- Create: `curriculum.toml`
- Create: `content-review.toml`
- Create: `artifacts/source-sync/latest-impact.md`
- Modify: `README.md`
- Modify: `.gitmodules`
- Modify: `01-overview/02-architecture.md`
- Modify: `01-overview/05-process-and-ipc-internals.md`
- Modify: `02-core-concepts/04-prefix-caching.md`
- Modify: `03-code-walkthrough/02b-scheduling-policies.md`
- Modify: `03-code-walkthrough/03-kv-cache-manager.md`
- Modify: `03-code-walkthrough/05-attention-backends.md`
- Modify: `03-code-walkthrough/07-model-architectures.md`
- Modify: `04-optimizations/04-compilation-internals.md`
- Modify: `05-distributed/03-expert-parallel-deep-dive.md`
- Modify: `05-distributed/04-context-parallel.md`
- Modify: `08-production-deployment/08-monitoring-cookbook.md`
- Modify: `08-production-deployment/10-gpu-utilization-and-tail-latency.md`
- Modify: `09-advanced-features/01-sampling-and-logits.md`
- Modify: `09-advanced-features/02-structured-output.md`
- Modify: `09-advanced-features/03-multimodal.md`
- Modify: `09-advanced-features/04-lora-serving.md`
- Modify: `09-advanced-features/05-embedding-and-pooling.md`
- Modify: submodule gitlink `vllm`
- Modify outside tutorial repository: local ref `vllm/.git/refs/heads/main`

**Interfaces:**
- Consumes: official upstream `main`, existing submodule baseline `27b85d2084c48f9b12f8cfd6638a56fe9b257635`, all source-sync CLI commands.
- Produces: one exact candidate SHA shared by the independent vLLM `main`, submodule, lock file, generated source URLs, README block, and impact report; a 50-chapter inventory with pending review ledger.

- [ ] **Step 1: Capture preservation evidence before mutation**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm && test "$(git branch --show-current)" = "bugfix/parallel-config-size-validation" && git rev-parse 004b8601c d1cd0162d 090fd61d8 && git status --short --branch'
```

Expected: the named branch is current, all three SHAs resolve, and no unrelated working-tree changes are present. If unrelated changes exist, stop and preserve them before continuing.

- [ ] **Step 2: Fetch and atomically fast-forward only the local vLLM main ref**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm && git fetch https://github.com/vllm-project/vllm.git main && candidate_sha="$(git rev-parse FETCH_HEAD)" && printf "%s\n" "$candidate_sha" | grep -Eq "^[0-9a-f]{40}$" && old_main="$(git rev-parse main)" && git merge-base --is-ancestor "$old_main" "$candidate_sha" && git update-ref refs/heads/main "$candidate_sha" "$old_main" && test "$(git branch --show-current)" = "bugfix/parallel-config-size-validation" && git rev-parse 004b8601c d1cd0162d 090fd61d8'
```

Expected: `main` fast-forwards to official `origin/main`, the active branch does not change, and the three user commits remain reachable. Do not replace this with reset, checkout, rebase, or force push.

- [ ] **Step 3: Move the tutorial submodule to the same official commit**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git submodule update --init vllm && git -C vllm fetch https://github.com/vllm-project/vllm.git main && candidate_sha="$(git -C ../vllm rev-parse main)" && test "$(git -C vllm cat-file -t "$candidate_sha")" = commit && git -C vllm checkout --detach "$candidate_sha" && test "$(git -C vllm rev-parse HEAD)" = "$candidate_sha"'
```

Expected: submodule HEAD equals the independent repository's updated `main` SHA.

- [ ] **Step 4: Add an explicit HTTPS submodule branch contract**

Update `.gitmodules` to:

```ini
[submodule "vllm"]
	path = vllm
	url = https://github.com/vllm-project/vllm.git
	branch = main
```

This removes SSH-key dependence from public CI while keeping the gitlink pinned.

- [ ] **Step 5: Bootstrap complete chapter and pending-review inventories**

Create one `[[chapter]]` record for each of the 50 discovered chapter paths. Titles come from the first H1; levels, tracks, environments, and source areas are explicit, not inferred at validation time. Use these section defaults as the starting contract and narrow them per chapter where the filename names a subsystem:

```toml
# 01-overview
level = "beginner"
tracks = ["quickstart", "internals", "interview"]
environments = ["no-gpu", "nvidia-gpu"]
source_areas = ["vllm/entrypoints/**", "vllm/v1/engine/**", "vllm/config/**"]

# 02-core-concepts and 03-code-walkthrough
level = "intermediate"
tracks = ["internals", "interview"]
environments = ["no-gpu", "nvidia-gpu"]

# 04-optimizations and 05-distributed
level = "advanced"
tracks = ["internals", "production", "interview"]
environments = ["no-gpu", "nvidia-gpu", "multi-gpu"]

# 06-interview
level = "intermediate"
tracks = ["interview"]
environments = ["no-gpu"]

# 07-hands-on
level = "intermediate"
tracks = ["quickstart", "production", "interview"]
environments = ["no-gpu", "nvidia-gpu"]

# 08-production-deployment
level = "advanced"
tracks = ["production", "interview"]
environments = ["no-gpu", "nvidia-gpu", "multi-gpu"]

# 09-advanced-features
level = "advanced"
tracks = ["internals", "production", "interview"]
environments = ["no-gpu", "nvidia-gpu"]
```

For `02` through `09`, derive `source_areas` from the exact module paths already cited in the chapter plus the authoritative architecture map in root `AGENTS.md`; never use a repository-wide `vllm/**` catch-all. Add one `[[review]]` per chapter with:

```toml
status = "pending"
reviewed_commit = ""
reviewed_at = ""
source_contracts = false
commands_checked = false
metrics_checked = false
diagrams_checked = false
hardware_verified = []
notes = []
```

Run the inventory test immediately after writing:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts'
```

Expected at this intermediate point: failures are limited to missing lock/version data and unmanaged source lines, not inventory coverage.

- [ ] **Step 6: Create the candidate lock and impact report**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && candidate_sha="$(git -C vllm rev-parse HEAD)" && validated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" && python3 -m tools.source_sync impact --baseline 27b85d2084c48f9b12f8cfd6638a56fe9b257635 --candidate "$candidate_sha" --output artifacts/source-sync/latest-impact.md && python3 -m tools.source_sync refresh --candidate-sha "$candidate_sha" --validated-at "$validated_at" --report artifacts/source-sync/latest-impact.md'
```

Expected: `source.lock.json`, README version block, and impact report contain the same full candidate SHA. `refresh` may still report unmanaged legacy references until the next step.

- [ ] **Step 7: Migrate every explicit source line reference**

Run a contracts validation to obtain the exact list:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts'
```

For each reported prose reference:

1. Normalize paths relative to the submodule root; change accidental `vllm/vllm/...` to `vllm/...` only after confirming the target.
2. For Python, use the narrowest stable qualified symbol and, when a specific statement matters, an exact `anchor` inside that symbol.
3. For C++/CUDA/configuration files, use an exact unique `anchor` and the smallest explanatory `span`.
4. Insert the one-line JSON directive immediately before a normal Markdown link.
5. Replace comma-separated manual line lists with separate semantic contracts when the lines represent different concepts.
6. Delete prose claims that refer to removed code instead of anchoring to unrelated replacements; record the chapter in the impact report for content review.

The migration must include every validation-reported file, including the 18 files identified in the initial inventory, and continue until:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && candidate_sha="$(git -C vllm rev-parse HEAD)" && validated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" && python3 -m tools.source_sync refresh --candidate-sha "$candidate_sha" --validated-at "$validated_at" --report artifacts/source-sync/latest-impact.md && python3 -m tools.source_sync validate --profile contracts'
```

Expected: `OK: source contracts are valid`; full validation still reports pending chapter reviews, which the curriculum plan owns.

- [ ] **Step 8: Verify preservation and commit the repository baseline**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && git diff --check && git status --short && test "$(git -C ../vllm branch --show-current)" = "bugfix/parallel-config-size-validation" && git -C ../vllm rev-parse 004b8601c d1cd0162d 090fd61d8'
```

Expected: contracts pass, diff check passes, only planned files and the submodule are modified, and user commits remain reachable.

Commit:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add .gitmodules README.md source.lock.json curriculum.toml content-review.toml artifacts/source-sync/latest-impact.md vllm 01-overview 02-core-concepts 03-code-walkthrough 04-optimizations 05-distributed 06-interview 07-hands-on 08-production-deployment 09-advanced-features && git commit -m "docs: align source contracts with vLLM main"'
```

---

### Task 7: Author Runbook and Fail-Closed GitHub Workflows

**Files:**
- Create: `docs/source-sync.md`
- Create: `.github/workflows/validate.yml`
- Create: `.github/workflows/sync-upstream.yml`
- Create: `tests/source_sync/test_workflows.py`
- Modify: `.github/workflows/pages.yml`
- Modify: `DEPLOY.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: source-sync CLI, docs requirements, submodule contract.
- Produces: contributor commands, PR validation, weekly/manual candidate PR generation, and deployment gate.

- [ ] **Step 1: Write the author runbook**

`docs/source-sync.md` must contain these complete sections:

1. Why manual line numbers are forbidden.
2. Path-only, Python symbol, Python symbol plus anchor, and CUDA text-anchor examples.
3. `validate`, `refresh`, `impact`, and `check-upstream` command examples.
4. Error table for missing path, missing symbol, zero/multiple anchors, stale URL, uncovered changed file, and pending review.
5. Manual upstream refresh sequence and rollback using a new commit, never `reset --hard`.
6. Review checklist distinguishing source-contract validity from semantic chapter review.
7. GPU evidence rule and the meaning of an empty `hardware_verified` list.

Link this runbook from README and DEPLOY. README's generated version area must show the validated baseline SHA/time, the candidate SHA and lag from the latest impact report when present, and a badge/link to `sync-upstream.yml`; a failed workflow badge is the public signal that the candidate has not replaced the published baseline.

- [ ] **Step 2: Add a failing workflow contract test**

Create `tests/source_sync/test_workflows.py`. Load each workflow with `yaml.load(text, Loader=yaml.BaseLoader)` and assert:

```python
class WorkflowTests(unittest.TestCase):
    def test_all_workflows_parse_and_pin_checkout(self):
        for name in ("validate.yml", "sync-upstream.yml", "pages.yml"):
            with self.subTest(name=name):
                path = ROOT / ".github" / "workflows" / name
                data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
                self.assertIsInstance(data, dict)
                self.assertIn("on", data)
                self.assertIn("jobs", data)
                self.assertIn("actions/checkout@v4", path.read_text(encoding="utf-8"))
```

Run it before creating the two new workflows:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_workflows -v'
```

Expected: failure because `validate.yml` and `sync-upstream.yml` do not exist.

- [ ] **Step 3: Add pull-request validation**

Create `.github/workflows/validate.yml` with:

- triggers `pull_request` and `workflow_dispatch`;
- `permissions: contents: read`;
- `actions/checkout@v4` with `submodules: true` and `fetch-depth: 0`;
- Python 3.9 and 3.12 matrix for `python3 -m unittest discover -s tests/source_sync -v`;
- Python 3.12 job for `python3 -m tools.source_sync validate --profile contracts --require-committed`;
- `python3 build_html.py` with `VLLM_LEARNING_DST=${{ runner.temp }}/vllm-learning-site`;
- artifact upload of the generated site on failure and success for inspection.

Install only `python -m pip install -r requirements-docs.txt`.

- [ ] **Step 4: Add weekly/manual source candidate workflow**

Create `.github/workflows/sync-upstream.yml` with:

- schedule `17 3 * * 1` and `workflow_dispatch.inputs.candidate_sha` as an optional string;
- `permissions: contents: write, pull-requests: write`;
- one concurrency group `vllm-main-sync` with `cancel-in-progress: false`;
- checkout with submodules and full history;
- candidate resolution from the input or official HTTPS `git ls-remote`;
- strict full-SHA validation and `git -C vllm fetch origin main`;
- submodule detached checkout, impact generation, refresh, contracts validation, unit tests, and HTML build;
- validation steps configured to capture status while continuing long enough to write the impact report;
- automation branch `automation/vllm-main-sync`, updated only with `git push --force-with-lease` after confirming the remote branch name exactly;
- `gh pr create` or `gh pr edit` for one open PR titled `docs: sync tutorial to vLLM main`;
- PR body copied from `artifacts/source-sync/latest-impact.md` plus validation status;
- final nonzero exit when any gate failed, leaving a visible red candidate PR.

The workflow must never merge, push to `main`, push to upstream vLLM, or deploy Pages.

- [ ] **Step 5: Gate Pages on committed contracts**

Modify `.github/workflows/pages.yml`:

1. checkout with `submodules: true` and `fetch-depth: 0`;
2. install `requirements-docs.txt`;
3. run unit tests;
4. run `python3 -m tools.source_sync validate --profile contracts --require-committed` before build;
5. build into `_site` and upload only after all previous steps succeed.

The curriculum plan will change `contracts` to `full` after all chapter reviews are complete.

- [ ] **Step 6: Validate workflow syntax and documentation build**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest discover -s tests/source_sync -v && python3 -m unittest tests.source_sync.test_workflows -v && python3 -m tools.source_sync validate --profile contracts --require-committed && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test -s "$site_dir/index.html" && test -s "$site_dir/search-index.json"'
```

Expected: unit tests and contracts pass; a clean temporary site contains index and search index. If the mount has not flushed after an edit, rerun this same remote command once rather than running locally.

- [ ] **Step 7: Commit workflows and runbook**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add docs/source-sync.md .github/workflows/validate.yml .github/workflows/sync-upstream.yml .github/workflows/pages.yml DEPLOY.md README.md tests/source_sync/test_workflows.py && git commit -m "ci: validate and propose vLLM source updates"'
```

---

### Task 8: Source-Sync Acceptance Checkpoint

**Files:**
- Modify only if verification reveals a source-sync defect: files created or modified in Tasks 1-7.

**Interfaces:**
- Consumes: complete source-sync implementation.
- Produces: evidence that the infrastructure portion is ready for the curriculum refresh.

- [ ] **Step 1: Run all source-sync verification**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest discover -s tests/source_sync -v && python3 -m tools.source_sync validate --profile contracts --require-committed && first_hash="$(git hash-object README.md curriculum.toml content-review.toml source.lock.json | shasum)" && candidate_sha="$(git -C vllm rev-parse HEAD)" && validated_at="$(python3 -c "import json; print(json.load(open(\"source.lock.json\"))[\"validated_at\"])")" && python3 -m tools.source_sync refresh --candidate-sha "$candidate_sha" --validated-at "$validated_at" --report artifacts/source-sync/latest-impact.md && second_hash="$(git hash-object README.md curriculum.toml content-review.toml source.lock.json | shasum)" && test "$first_hash" = "$second_hash" && git diff --exit-code && git status --short --branch'
```

Expected: tests pass, contracts pass, refresh is byte-for-byte idempotent for managed metadata, no diff remains, and the tutorial branch is clean except for intentional commits not yet pushed.

- [ ] **Step 2: Re-check official-main freshness and branch preservation**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && official_sha="$(git ls-remote https://github.com/vllm-project/vllm.git refs/heads/main | awk "{print \$1}")" && test "${#official_sha}" -eq 40 && test "$(git -C vllm rev-parse HEAD)" = "$official_sha" && test "$(git -C ../vllm rev-parse main)" = "$official_sha" && test "$(git -C ../vllm branch --show-current)" = "bugfix/parallel-config-size-validation" && git -C ../vllm rev-parse 004b8601c d1cd0162d 090fd61d8'
```

Expected: official main, local main, and submodule match at the checkpoint; the active user branch and all user commits are intact. If official main advanced, repeat Task 6 impact/refresh/migration only for the new candidate before recording this checkpoint.

- [ ] **Step 3: Record checkpoint evidence without claiming curriculum completion**

Append a dated section to `artifacts/source-sync/latest-impact.md` containing the exact commands and results from Steps 1-2. State explicitly that full semantic chapter review remains pending under `2026-07-20-curriculum-refresh-implementation.md`.

Commit only if the report changed:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add artifacts/source-sync/latest-impact.md && git diff --cached --quiet || git commit -m "docs: record source sync checkpoint"'
```
