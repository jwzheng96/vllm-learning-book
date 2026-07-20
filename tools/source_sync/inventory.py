from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml

from .models import Chapter, FULL_SHA_RE, Review, SourceLock

SECTIONS = (
    "01-overview",
    "02-core-concepts",
    "03-code-walkthrough",
    "04-optimizations",
    "05-distributed",
    "06-interview",
    "07-hands-on",
    "08-production-deployment",
    "09-advanced-features",
)
LEVELS = frozenset({"beginner", "intermediate", "advanced"})
TRACKS = frozenset({"quickstart", "internals", "production", "interview"})
ENVIRONMENTS = frozenset({"no-gpu", "cpu", "nvidia-gpu", "multi-gpu"})
REVIEW_PROFILES = frozenset({"contracts", "full"})
REVIEW_STATUSES = frozenset({"pending", "reviewed"})

_CHAPTER_KEYS = frozenset(
    {"path", "title", "level", "tracks", "environments", "source_areas"}
)
_REVIEW_KEYS = frozenset(
    {
        "path",
        "status",
        "reviewed_commit",
        "reviewed_at",
        "source_contracts",
        "commands_checked",
        "metrics_checked",
        "diagrams_checked",
        "hardware_verified",
        "notes",
    }
)


def _load_toml(path: Path) -> Mapping[str, object]:
    try:
        with path.open("rb") as stream:
            data = _toml.load(stream)
    except (OSError, _toml.TOMLDecodeError) as exc:
        raise ValueError(f"cannot load TOML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"TOML root must be a table: {path}")
    return data


def _record_list(
    data: Mapping[str, object], key: str, path: Path
) -> Sequence[Mapping[str, object]]:
    unknown_root = sorted(set(data) - {key})
    if unknown_root:
        raise ValueError(
            f"unknown top-level keys in {path}: {', '.join(unknown_root)}"
        )
    records = data.get(key)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain [[{key}]] records")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"every [[{key}]] entry in {path} must be a table")
    return records


def _exact_keys(
    record: Mapping[str, object], expected: frozenset, context: str
) -> None:
    missing = sorted(expected - set(record))
    unknown = sorted(set(record) - expected)
    if missing:
        raise ValueError(f"{context} missing keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown keys: {', '.join(unknown)}")


def _string(record: Mapping[str, object], key: str, context: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be a string")
    return value


def _string_tuple(
    record: Mapping[str, object], key: str, context: str
) -> Tuple[str, ...]:
    value = record[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{context}.{key} must be an array of strings")
    return tuple(value)


def _boolean(record: Mapping[str, object], key: str, context: str) -> bool:
    value = record[key]
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return value


def _chapter_path(value: str, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] not in SECTIONS
        or path.suffix != ".md"
    ):
        raise ValueError(f"{context} must be a top-level chapter Markdown path")
    return path


def load_curriculum(path: Path) -> Tuple[Chapter, ...]:
    records = _record_list(_load_toml(path), "chapter", path)
    chapters = []
    seen = set()
    for index, record in enumerate(records, start=1):
        context = f"chapter[{index}]"
        _exact_keys(record, _CHAPTER_KEYS, context)
        chapter_path = _chapter_path(_string(record, "path", context), context)
        if chapter_path in seen:
            raise ValueError(f"duplicate chapter path: {chapter_path}")
        seen.add(chapter_path)
        chapters.append(
            Chapter(
                path=chapter_path,
                title=_string(record, "title", context),
                level=_string(record, "level", context),
                tracks=_string_tuple(record, "tracks", context),
                environments=_string_tuple(record, "environments", context),
                source_areas=_string_tuple(record, "source_areas", context),
            )
        )
    return tuple(chapters)


def load_reviews(path: Path) -> Tuple[Review, ...]:
    records = _record_list(_load_toml(path), "review", path)
    reviews = []
    seen = set()
    for index, record in enumerate(records, start=1):
        context = f"review[{index}]"
        _exact_keys(record, _REVIEW_KEYS, context)
        review_path = _chapter_path(_string(record, "path", context), context)
        if review_path in seen:
            raise ValueError(f"duplicate review path: {review_path}")
        seen.add(review_path)
        reviews.append(
            Review(
                path=review_path,
                status=_string(record, "status", context),
                reviewed_commit=_string(record, "reviewed_commit", context),
                reviewed_at=_string(record, "reviewed_at", context),
                source_contracts=_boolean(record, "source_contracts", context),
                commands_checked=_boolean(record, "commands_checked", context),
                metrics_checked=_boolean(record, "metrics_checked", context),
                diagrams_checked=_boolean(record, "diagrams_checked", context),
                hardware_verified=_string_tuple(
                    record, "hardware_verified", context
                ),
                notes=_string_tuple(record, "notes", context),
            )
        )
    return tuple(reviews)


def discover_chapters(repo_root: Path) -> Tuple[PurePosixPath, ...]:
    paths = []
    for section in SECTIONS:
        directory = repo_root / section
        if not directory.is_dir():
            continue
        for markdown_path in directory.glob("*.md"):
            paths.append(PurePosixPath(section) / markdown_path.name)
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _first_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _review_time_is_valid(value: str) -> bool:
    if not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def validate_inventory(
    repo_root: Path, lock: SourceLock, profile: str
) -> Tuple[str, ...]:
    if profile not in REVIEW_PROFILES:
        raise ValueError(
            f"unknown review profile {profile!r}; expected contracts or full"
        )

    try:
        chapters = load_curriculum(repo_root / "curriculum.toml")
        reviews = load_reviews(repo_root / "content-review.toml")
    except ValueError as exc:
        return (str(exc),)

    errors: List[str] = []
    discovered = set(discover_chapters(repo_root))
    chapter_by_path: Dict[PurePosixPath, Chapter] = {
        chapter.path: chapter for chapter in chapters
    }
    review_by_path: Dict[PurePosixPath, Review] = {
        review.path: review for review in reviews
    }

    for path in sorted(discovered - set(chapter_by_path), key=str):
        errors.append(f"chapter missing from curriculum.toml: {path}")
    for path in sorted(set(chapter_by_path) - discovered, key=str):
        errors.append(f"curriculum chapter does not exist: {path}")
    for path in sorted(set(chapter_by_path) - set(review_by_path), key=str):
        errors.append(f"chapter missing from content-review.toml: {path}")
    for path in sorted(set(review_by_path) - set(chapter_by_path), key=str):
        errors.append(f"review has no curriculum chapter: {path}")

    for chapter in chapters:
        prefix = f"{chapter.path}:"
        if chapter.level not in LEVELS:
            errors.append(f"{prefix} invalid level {chapter.level!r}")
        if not chapter.tracks:
            errors.append(f"{prefix} tracks must not be empty")
        for track in chapter.tracks:
            if track not in TRACKS:
                errors.append(f"{prefix} invalid track {track!r}")
        if not chapter.environments:
            errors.append(f"{prefix} environments must not be empty")
        for environment in chapter.environments:
            if environment not in ENVIRONMENTS:
                errors.append(f"{prefix} invalid environment {environment!r}")
        if not chapter.source_areas:
            errors.append(f"{prefix} source_areas must not be empty")
        for area in chapter.source_areas:
            area_path = PurePosixPath(area)
            if area_path.is_absolute() or ".." in area_path.parts or not area:
                errors.append(f"{prefix} invalid source area {area!r}")
        disk_path = repo_root / chapter.path.as_posix()
        if disk_path.is_file():
            disk_title = _first_title(disk_path)
            if not disk_title:
                errors.append(f"{prefix} chapter has no H1 title")
            elif disk_title != chapter.title:
                errors.append(
                    f"{prefix} title mismatch: inventory={chapter.title!r}, "
                    f"markdown={disk_title!r}"
                )

    for review in reviews:
        prefix = f"{review.path}:"
        if review.status not in REVIEW_STATUSES:
            errors.append(f"{prefix} invalid review status {review.status!r}")
        if review.reviewed_commit and not FULL_SHA_RE.fullmatch(
            review.reviewed_commit
        ):
            errors.append(f"{prefix} reviewed_commit must be a full SHA")
        if profile == "full":
            checks = (
                review.status == "reviewed",
                review.reviewed_commit == lock.commit,
                _review_time_is_valid(review.reviewed_at),
                review.source_contracts,
                review.commands_checked,
                review.metrics_checked,
                review.diagrams_checked,
            )
            if not all(checks):
                errors.append(
                    f"{prefix} review must be complete for current source lock"
                )

    return tuple(sorted(errors))
