from __future__ import annotations

import ast
import os
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Tuple
from urllib.parse import quote, urlsplit

from .models import ResolvedReference, SourceDirective, SourceLock

_NAMED_NODES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _named_nodes(nodes: Iterable[ast.AST], name: str) -> Tuple[ast.AST, ...]:
    return tuple(
        node
        for node in nodes
        if isinstance(node, _NAMED_NODES) and node.name == name
    )


def resolve_python_symbol(text: str, symbol: str) -> Tuple[int, int]:
    if not symbol or any(not part for part in symbol.split(".")):
        raise ValueError(f"invalid Python symbol: {symbol!r}")

    tree = ast.parse(text)
    nodes = tree.body
    current: Optional[ast.AST] = None
    walked = []
    for part in symbol.split("."):
        walked.append(part)
        matches = _named_nodes(nodes, part)
        qualified = ".".join(walked)
        if not matches:
            raise ValueError(f"Python symbol not found: {qualified}")
        if len(matches) != 1:
            raise ValueError(f"Python symbol is ambiguous: {qualified}")
        current = matches[0]
        nodes = getattr(current, "body", ())

    assert current is not None
    start_line = current.lineno
    end_line = getattr(current, "end_lineno", None) or start_line
    return start_line, end_line


def _repository_base(lock: SourceLock) -> str:
    parsed = urlsplit(lock.repository)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise ValueError("source repository must be an HTTPS GitHub URL")
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if len([part for part in path.split("/") if part]) != 2:
        raise ValueError("source repository must identify one GitHub repository")
    return f"https://github.com{path}"


def _source_url(
    lock: SourceLock,
    path: PurePosixPath,
    *,
    directory: bool,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    kind = "tree" if directory else "blob"
    escaped_path = quote(path.as_posix(), safe="/")
    url = f"{_repository_base(lock)}/{kind}/{lock.commit}/{escaped_path}"
    if start_line is None:
        return url
    if end_line is None or end_line == start_line:
        return f"{url}#L{start_line}"
    return f"{url}#L{start_line}-L{end_line}"


def resolve_reference(
    source_root: Path,
    directive: SourceDirective,
    lock: SourceLock,
) -> ResolvedReference:
    root = source_root.resolve()
    source_path = (root / directive.path.as_posix()).resolve()
    if os.path.commonpath((str(root), str(source_path))) != str(root):
        raise ValueError(f"source path is outside source root: {directive.path}")
    if not source_path.exists():
        raise ValueError(f"source path does not exist: {directive.path}")

    if source_path.is_dir():
        if directive.symbol or directive.anchor:
            raise ValueError("directories cannot have symbol or anchor fields")
        return ResolvedReference(
            directive=directive,
            source_path=source_path,
            start_line=None,
            end_line=None,
            url=_source_url(lock, directive.path, directory=True),
        )

    if not directive.symbol and not directive.anchor:
        return ResolvedReference(
            directive=directive,
            source_path=source_path,
            start_line=None,
            end_line=None,
            url=_source_url(lock, directive.path, directory=False),
        )

    lines = source_path.read_text(encoding="utf-8").splitlines()
    scope_start, scope_end = 1, len(lines)
    if directive.symbol:
        if source_path.suffix != ".py":
            raise ValueError("symbol is supported only for Python files")
        scope_start, scope_end = resolve_python_symbol(
            "\n".join(lines), directive.symbol
        )

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
                "anchor must occur exactly once in scope; "
                f"found {len(matches)}: {needle}"
            )
        start_line = matches[0]

    end_line = start_line + directive.span - 1
    if end_line > scope_end:
        raise ValueError("span exceeds the resolved source scope")
    return ResolvedReference(
        directive=directive,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        url=_source_url(
            lock,
            directive.path,
            directory=False,
            start_line=start_line,
            end_line=end_line,
        ),
    )
