from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .impact import build_impact, render_impact_markdown
from .models import FULL_SHA_RE, SourceLock
from .versions import (
    load_source_lock,
    refresh_markdown,
    refresh_readme_version,
    run_git,
    submodule_head,
    validate_repository,
    write_source_lock,
)

OFFICIAL_REPOSITORY = "https://github.com/vllm-project/vllm"
OFFICIAL_BRANCH = "main"


def _impact_commits(path: Path) -> Optional[Tuple[str, str]]:
    baseline = ""
    candidate = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- Baseline: `") and line.endswith("`"):
            baseline = line[len("- Baseline: `") : -1]
        elif line.startswith("- Candidate: `") and line.endswith("`"):
            candidate = line[len("- Candidate: `") : -1]
    if FULL_SHA_RE.fullmatch(baseline) and FULL_SHA_RE.fullmatch(candidate):
        return baseline, candidate
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m tools.source_sync")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument(
        "--profile", choices=("contracts", "full"), default="full"
    )
    validate.add_argument("--require-committed", action="store_true")

    refresh = subparsers.add_parser("refresh")
    refresh.add_argument("--candidate-sha", required=True)
    refresh.add_argument("--validated-at", required=True)
    refresh.add_argument("--report", type=Path)

    impact = subparsers.add_parser("impact")
    impact.add_argument("--baseline", required=True)
    impact.add_argument("--candidate", required=True)
    impact.add_argument("--output", type=Path, required=True)

    subparsers.add_parser("check-upstream")
    return parser


def _validate(repo_root: Path, profile: str, require_committed: bool) -> int:
    errors = validate_repository(
        repo_root, profile=profile, require_committed=require_committed
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    if profile == "full":
        print("OK: full source and content review is valid")
    else:
        print("OK: source contracts are valid")
    return 0


def _refresh(
    repo_root: Path,
    candidate_sha: str,
    validated_at: str,
    report: Optional[Path],
) -> int:
    if not FULL_SHA_RE.fullmatch(candidate_sha):
        print("ERROR: candidate SHA must be 40 lowercase hexadecimal characters")
        return 1
    if report is not None and not report.exists():
        print(f"ERROR: requested impact report does not exist: {report}")
        return 1
    try:
        actual = submodule_head(repo_root)
        if actual != candidate_sha:
            raise ValueError(
                f"submodule HEAD {actual} does not equal candidate {candidate_sha}"
            )
        committed_at = run_git(
            repo_root / "vllm",
            "show",
            "-s",
            "--format=%cI",
            candidate_sha,
        )
        lock = SourceLock(
            schema_version=1,
            repository=OFFICIAL_REPOSITORY,
            branch=OFFICIAL_BRANCH,
            commit=candidate_sha,
            committed_at=committed_at,
            validated_at=validated_at,
        )
        write_source_lock(repo_root / "source.lock.json", lock)
        impact_candidate = ""
        impact_lag = None
        impact_path = ""
        if report is not None:
            commits = _impact_commits(report)
            if commits is not None:
                baseline, impact_candidate = commits
                if impact_candidate != candidate_sha:
                    raise ValueError(
                        "impact report candidate does not match refresh candidate"
                    )
                impact_lag = int(
                    run_git(
                        repo_root / "vllm",
                        "rev-list",
                        "--count",
                        f"{baseline}..{impact_candidate}",
                    )
                )
                try:
                    impact_path = report.resolve().relative_to(
                        repo_root.resolve()
                    ).as_posix()
                except ValueError:
                    impact_path = str(report)
        refresh_readme_version(
            repo_root / "README.md",
            lock,
            candidate=impact_candidate,
            lag_commits=impact_lag,
            impact_report=impact_path,
        )
        refresh_markdown(repo_root, lock)
        errors = validate_repository(repo_root, profile="contracts")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: refreshed source contracts for {candidate_sha}")
    return 0


def _impact(
    repo_root: Path, baseline: str, candidate: str, output: Path
) -> int:
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        if not FULL_SHA_RE.fullmatch(value):
            print(f"ERROR: {name} must be a full lowercase SHA")
            return 1
    try:
        result = build_impact(repo_root, baseline, candidate)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_impact_markdown(result), encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"OK: wrote impact report to {output}")
    return 0


def _check_upstream(repo_root: Path) -> int:
    try:
        baseline = load_source_lock(repo_root / "source.lock.json").commit
        result = subprocess.run(
            [
                "git",
                "ls-remote",
                f"{OFFICIAL_REPOSITORY}.git",
                "refs/heads/main",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise ValueError(result.stderr.strip() or "git ls-remote failed")
        candidate = result.stdout.split()[0]
        if not FULL_SHA_RE.fullmatch(candidate):
            raise ValueError("official main did not return a full SHA")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        json.dumps(
            {
                "baseline": baseline,
                "candidate": candidate,
                "changed": baseline != candidate,
            },
            sort_keys=True,
        )
    )
    return 0


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    repo_root: Optional[Path] = None,
) -> int:
    args = _parser().parse_args(argv)
    root = (
        repo_root.resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if args.command == "validate":
        return _validate(root, args.profile, args.require_committed)
    if args.command == "refresh":
        return _refresh(root, args.candidate_sha, args.validated_at, args.report)
    if args.command == "impact":
        return _impact(root, args.baseline, args.candidate, args.output)
    if args.command == "check-upstream":
        return _check_upstream(root)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
