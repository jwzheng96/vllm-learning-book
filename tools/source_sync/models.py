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
            raise ValueError(
                "source lock commit must be a 40-character lowercase SHA"
            )

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
        if (
            self.path.is_absolute()
            or ".." in self.path.parts
            or path_text in ("", ".")
        ):
            raise ValueError("path must be a confined relative source path")
        if (
            not isinstance(self.span, int)
            or isinstance(self.span, bool)
            or self.span < 1
        ):
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
