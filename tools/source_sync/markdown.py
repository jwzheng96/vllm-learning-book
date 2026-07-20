from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Dict, List, Tuple

from .models import MarkdownReference, SourceDirective, SourceLock
from .resolver import resolve_reference

DIRECTIVE_PREFIX = "<!-- vllm-source: "
DIRECTIVE_SUFFIX = " -->"
DIRECTIVE_KEYS = frozenset({"path", "symbol", "anchor", "span"})
MARKDOWN_LINK_RE = re.compile(
    r"^(?P<prefix>\s*\[[^\n]*\]\()(?P<url>[^)\s]+)(?P<suffix>\)\s*)$"
)
LEGACY_LINE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:vllm/|csrc/)"
    r"[A-Za-z0-9_./{}*-]+"
    r":\d+(?:[-,]\s*\d+)*"
)


def _directive_data(path: Path, line_number: int, line: str) -> Dict[str, object]:
    stripped = line.strip()
    payload = stripped[len(DIRECTIVE_PREFIX) : -len(DIRECTIVE_SUFFIX)]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid vllm-source JSON: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}:{line_number}: vllm-source JSON must be an object"
        )
    unknown = sorted(set(data) - DIRECTIVE_KEYS)
    if unknown:
        raise ValueError(
            f"{path}:{line_number}: unknown vllm-source keys: "
            + ", ".join(unknown)
        )
    return data


def _source_directive(path: Path, line_number: int, data: Dict[str, object]) -> SourceDirective:
    source_path = data.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError(
            f"{path}:{line_number}: vllm-source path must be a non-empty string"
        )
    symbol = data.get("symbol")
    if symbol is not None and (not isinstance(symbol, str) or not symbol):
        raise ValueError(
            f"{path}:{line_number}: vllm-source symbol must be a non-empty string"
        )
    anchor = data.get("anchor")
    if anchor is not None and (not isinstance(anchor, str) or not anchor.strip()):
        raise ValueError(
            f"{path}:{line_number}: vllm-source anchor must be a non-empty string"
        )
    span = data.get("span", 1)
    return SourceDirective(
        path=PurePosixPath(source_path),
        symbol=symbol,
        anchor=anchor,
        span=span,
    )


def scan_document(path: Path, text: str) -> Tuple[MarkdownReference, ...]:
    lines = text.splitlines(keepends=True)
    starts: List[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    references: List[MarkdownReference] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not (
            stripped.startswith(DIRECTIVE_PREFIX)
            and stripped.endswith(DIRECTIVE_SUFFIX)
        ):
            continue
        line_number = index + 1
        data = _directive_data(path, line_number, line)
        directive = _source_directive(path, line_number, data)
        if index + 1 >= len(lines):
            raise ValueError(
                f"{path}:{line_number}: vllm-source directive requires a "
                "Markdown link on the next physical line"
            )
        link_line = lines[index + 1].rstrip("\r\n")
        link_match = MARKDOWN_LINK_RE.fullmatch(link_line)
        if link_match is None:
            raise ValueError(
                f"{path}:{line_number}: vllm-source directive requires a "
                "Markdown link on the next physical line"
            )
        link_start = starts[index + 1]
        url_start = link_start + link_match.start("url")
        url_end = link_start + link_match.end("url")
        references.append(
            MarkdownReference(
                markdown_path=path,
                directive=directive,
                directive_start=starts[index],
                link_start=link_start,
                url_start=url_start,
                url_end=url_end,
                current_url=link_match.group("url"),
            )
        )
    return tuple(references)


def refresh_document(
    path: Path,
    text: str,
    source_root: Path,
    lock: SourceLock,
) -> str:
    replacements = []
    for reference in scan_document(path, text):
        resolved = resolve_reference(source_root, reference.directive, lock)
        replacements.append((reference.url_start, reference.url_end, resolved.url))

    refreshed = text
    for start, end, url in reversed(replacements):
        refreshed = refreshed[:start] + url + refreshed[end:]
    return refreshed


def find_unmanaged_line_references(path: Path, text: str) -> Tuple[str, ...]:
    errors = []
    for match in LEGACY_LINE_RE.finditer(text):
        line_number = text.count("\n", 0, match.start()) + 1
        errors.append(f"{path}:{line_number}: {match.group(0)}")
    return tuple(errors)
