from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from .inventory import discover_chapters, validate_inventory
from .markdown import (
    find_unmanaged_line_references,
    refresh_document,
    scan_document,
)
from .models import SourceLock
from .resolver import resolve_reference

SOURCE_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "repository",
        "branch",
        "commit",
        "committed_at",
        "validated_at",
    }
)
VERSION_START = "<!-- vllm-version:start -->"
VERSION_END = "<!-- vllm-version:end -->"


def run_git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise ValueError(message or f"git command failed: {' '.join(args)}")
    return result.stdout.strip()


def load_source_lock(path: Path) -> SourceLock:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load source lock {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("source lock must contain one JSON object")
    missing = sorted(SOURCE_LOCK_KEYS - set(data))
    unknown = sorted(set(data) - SOURCE_LOCK_KEYS)
    if missing:
        raise ValueError(f"missing source lock keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown source lock keys: {', '.join(unknown)}")
    return SourceLock(**data)


def write_source_lock(path: Path, lock: SourceLock) -> None:
    data = {
        "schema_version": lock.schema_version,
        "repository": lock.repository,
        "branch": lock.branch,
        "commit": lock.commit,
        "committed_at": lock.committed_at,
        "validated_at": lock.validated_at,
    }
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def submodule_head(repo_root: Path, submodule: str = "vllm") -> str:
    path = repo_root / submodule
    if not path.exists() or not (path / ".git").exists():
        raise ValueError(f"submodule {submodule} is not initialized")
    try:
        superproject = run_git(path, "rev-parse", "--show-superproject-working-tree")
        if Path(superproject).resolve() != repo_root.resolve():
            raise ValueError("submodule belongs to a different superproject")
        return run_git(path, "rev-parse", "HEAD")
    except ValueError as exc:
        raise ValueError(f"submodule {submodule} is not initialized: {exc}") from exc


def gitlink_head(repo_root: Path, submodule: str = "vllm") -> str:
    output = run_git(repo_root, "ls-files", "--stage", "--", submodule)
    if not output:
        raise ValueError(f"committed gitlink is missing: {submodule}")
    first_line = output.splitlines()[0]
    fields = first_line.split()
    if len(fields) < 3 or fields[0] != "160000":
        raise ValueError(f"tracked path is not a submodule gitlink: {submodule}")
    commit = fields[1]
    if len(commit) != 40:
        raise ValueError(f"committed gitlink has an invalid SHA: {submodule}")
    return commit


def render_version_block(
    lock: SourceLock,
    *,
    candidate: str = "",
    lag_commits: Optional[int] = None,
    impact_report: str = "",
) -> str:
    lines = [
        VERSION_START,
        f"- Validated vLLM: `{lock.commit}`",
        f"- Upstream committed: `{lock.committed_at}`",
        f"- Validated: `{lock.validated_at}`",
    ]
    if candidate:
        lines.append(f"- Latest candidate: `{candidate}`")
    if lag_commits is not None:
        lines.append(f"- Candidate lag: `{lag_commits}` commits")
    if impact_report:
        lines.append(f"- Impact report: [{impact_report}]({impact_report})")
    lines.append(VERSION_END)
    return "\n".join(lines)


def refresh_readme_version(
    path: Path,
    lock: SourceLock,
    *,
    candidate: str = "",
    lag_commits: Optional[int] = None,
    impact_report: str = "",
) -> None:
    text = path.read_text(encoding="utf-8")
    block = render_version_block(
        lock,
        candidate=candidate,
        lag_commits=lag_commits,
        impact_report=impact_report,
    )
    start = text.find(VERSION_START)
    end = text.find(VERSION_END)
    if start >= 0 or end >= 0:
        if start < 0 or end < start:
            raise ValueError("README contains an incomplete vllm-version block")
        end += len(VERSION_END)
        refreshed = text[:start] + block + text[end:]
    else:
        first_newline = text.find("\n")
        if first_newline < 0:
            refreshed = text + "\n\n" + block + "\n"
        else:
            insertion = first_newline + 1
            refreshed = text[:insertion] + "\n" + block + "\n" + text[insertion:]
    path.write_text(refreshed, encoding="utf-8")


def markdown_paths(repo_root: Path) -> Tuple[Path, ...]:
    paths = []
    readme = repo_root / "README.md"
    if readme.is_file():
        paths.append(readme)
    paths.extend(repo_root / item.as_posix() for item in discover_chapters(repo_root))
    return tuple(paths)


def refresh_markdown(repo_root: Path, lock: SourceLock) -> None:
    source_root = repo_root / "vllm"
    for path in markdown_paths(repo_root):
        original = path.read_text(encoding="utf-8")
        refreshed = refresh_document(path, original, source_root, lock)
        if refreshed != original:
            path.write_text(refreshed, encoding="utf-8")


def validate_repository(
    repo_root: Path,
    profile: str = "full",
    require_committed: bool = False,
) -> Tuple[str, ...]:
    errors: List[str] = []
    try:
        lock = load_source_lock(repo_root / "source.lock.json")
    except ValueError as exc:
        return (str(exc),)

    source_root = repo_root / "vllm"
    source_ready = False
    source_filenames = None
    actual_submodule = None
    try:
        actual_submodule = submodule_head(repo_root)
        source_ready = True
        source_filenames = frozenset(
            item.name for item in source_root.rglob("*") if item.is_file()
        )
        if actual_submodule != lock.commit:
            errors.append(
                "submodule HEAD does not match source lock: "
                f"{actual_submodule} != {lock.commit}"
            )
    except ValueError as exc:
        errors.append(str(exc))

    if require_committed:
        try:
            committed = gitlink_head(repo_root)
            if committed != lock.commit:
                errors.append(
                    "committed gitlink does not match source lock: "
                    f"{committed} != {lock.commit}"
                )
            if actual_submodule is not None and committed != actual_submodule:
                errors.append(
                    "committed gitlink does not match submodule HEAD: "
                    f"{committed} != {actual_submodule}"
                )
        except ValueError as exc:
            errors.append(str(exc))

    errors.extend(validate_inventory(repo_root, lock, profile))

    readme = repo_root / "README.md"
    try:
        readme_text = readme.read_text(encoding="utf-8")
        start = readme_text.index(VERSION_START)
        end = readme_text.index(VERSION_END, start)
        block = readme_text[start : end + len(VERSION_END)]
        for expected in (lock.commit, lock.committed_at, lock.validated_at):
            if expected not in block:
                errors.append(f"README version block is stale: missing {expected}")
    except (OSError, ValueError) as exc:
        errors.append(f"README version block is invalid: {exc}")

    for path in markdown_paths(repo_root):
        try:
            text = path.read_text(encoding="utf-8")
            errors.extend(
                find_unmanaged_line_references(
                    path, text, source_filenames=source_filenames
                )
            )
            references = scan_document(path, text)
            if source_ready:
                for reference in references:
                    resolved = resolve_reference(
                        source_root, reference.directive, lock
                    )
                    if reference.current_url != resolved.url:
                        errors.append(
                            f"{path}: managed source URL is stale: "
                            f"{reference.current_url} != {resolved.url}"
                        )
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")

    return tuple(sorted(set(errors)))
