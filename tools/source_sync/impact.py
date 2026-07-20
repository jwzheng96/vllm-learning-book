from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath
from typing import Dict, List, Set, Tuple

from .inventory import load_curriculum
from .markdown import scan_document
from .models import ImpactResult
from .versions import run_git


def _changed_files(source_root, baseline: str, candidate: str) -> Tuple[str, ...]:
    output = run_git(
        source_root,
        "diff",
        "--no-renames",
        "--name-only",
        f"{baseline}..{candidate}",
        "--",
    )
    return tuple(sorted(line for line in output.splitlines() if line))


def _direct_paths(repo_root, chapter_path: PurePosixPath) -> Tuple[str, ...]:
    markdown_path = repo_root / chapter_path.as_posix()
    text = markdown_path.read_text(encoding="utf-8")
    return tuple(
        reference.directive.path.as_posix()
        for reference in scan_document(markdown_path, text)
    )


def _direct_match(changed_file: str, cited_path: str) -> bool:
    return changed_file == cited_path or changed_file.startswith(cited_path + "/")


def _area_match(changed_file: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(changed_file, pattern)


def build_impact(
    repo_root,
    baseline: str,
    candidate: str,
    unresolved: Tuple[str, ...] = (),
) -> ImpactResult:
    changed_files = _changed_files(repo_root / "vllm", baseline, candidate)
    chapters = load_curriculum(repo_root / "curriculum.toml")
    affected: Set[PurePosixPath] = set()
    covered: Set[str] = set()

    for chapter in chapters:
        direct_paths = _direct_paths(repo_root, chapter.path)
        for changed_file in changed_files:
            direct = any(
                _direct_match(changed_file, cited_path)
                for cited_path in direct_paths
            )
            area = any(
                _area_match(changed_file, pattern)
                for pattern in chapter.source_areas
            )
            if direct or area:
                affected.add(chapter.path)
                covered.add(changed_file)

    return ImpactResult(
        baseline=baseline,
        candidate=candidate,
        changed_files=changed_files,
        affected_chapters=tuple(sorted(affected, key=lambda path: path.as_posix())),
        uncovered_files=tuple(
            sorted(set(changed_files) - covered)
        ),
        unresolved=tuple(sorted(unresolved)),
    )


def _bullet_lines(values: Tuple[str, ...], empty: str = "- None") -> List[str]:
    if not values:
        return [empty]
    return [f"- `{value}`" for value in values]


def render_impact_markdown(result: ImpactResult) -> str:
    lines = [
        "# vLLM Upstream Impact Report",
        "",
        "## Summary",
        "",
        f"- Baseline: `{result.baseline}`",
        f"- Candidate: `{result.candidate}`",
        f"- Changed files: {len(result.changed_files)}",
        f"- Affected chapters: {len(result.affected_chapters)}",
        f"- Unresolved contracts: {len(result.unresolved)}",
        f"- Uncovered source files: {len(result.uncovered_files)}",
        "",
        "## Changed Files",
        "",
        *_bullet_lines(result.changed_files),
        "",
        "## Affected Chapters",
        "",
        *_bullet_lines(
            tuple(path.as_posix() for path in result.affected_chapters)
        ),
        "",
        "## Unresolved Contracts",
        "",
        *_bullet_lines(result.unresolved),
        "",
        "## Uncovered Source Files",
        "",
        *_bullet_lines(result.uncovered_files),
        "",
        "## Human Review Checklist",
        "",
    ]
    lines.extend(
        f"- [ ] `{path.as_posix()}`" for path in result.affected_chapters
    )
    lines.extend(
        [
            "- [ ] Defaults and CLI flags still match the candidate source.",
            "- [ ] Metrics names and meanings still match the candidate source.",
            "- [ ] Mermaid flows still match the candidate request path.",
            "- [ ] Commands and rollback instructions still use supported interfaces.",
            "- [ ] No GPU result is labeled as measured without a current hardware record.",
            "",
        ]
    )
    return "\n".join(lines)
