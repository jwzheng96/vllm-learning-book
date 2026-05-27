#!/usr/bin/env python3
"""Build a single combined PDF and EPUB from all vllm-learning markdown files.

Requirements (auto-checked):
- pandoc          (brew install pandoc)
- xelatex         (brew install --cask mactex-no-gui, or texlive-xetex on linux)

Source:  <script dir>/                          (this directory)
Output:  <script dir>/../vllm-learning-html/vllm-learning.{pdf,epub}

Override with the VLLM_LEARNING_SRC / VLLM_LEARNING_DST environment variables.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = Path(os.environ.get("VLLM_LEARNING_SRC", SCRIPT_DIR))
DST = Path(os.environ.get("VLLM_LEARNING_DST", SCRIPT_DIR / "_site"))

SECTIONS = [
    "01-overview",
    "02-core-concepts",
    "03-code-walkthrough",
    "04-optimizations",
    "05-distributed",
    "06-interview",
    "07-hands-on",
    "08-production-deployment",
    "09-advanced-features",
]


def check_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"ERROR: '{name}' not found. Install with: {install_hint}")


def discover_files() -> list[Path]:
    files: list[Path] = []
    readme = SRC / "README.md"
    if readme.exists():
        files.append(readme)
    for section in SECTIONS:
        d = SRC / section
        if not d.is_dir():
            continue
        for md in sorted(d.glob("*.md")):
            files.append(md)
    return files


# Strip cross-file relative .md links. Keep external links as-is.
_inline_md_link = re.compile(r"\[([^\]]+)\]\(([^)\s]+\.md)(#[^)]*)?\)")


def preprocess(md_text: str) -> str:
    # Convert [text](xxx.md#anchor) -> text  (kill cross-file refs)
    md_text = _inline_md_link.sub(lambda m: m.group(1), md_text)
    return md_text


def combine_files(files: list[Path]) -> str:
    # files[0] is README.md; the remaining entries are the actual chapters.
    chapter_count = max(0, len(files) - 1)
    parts: list[str] = []
    parts.append("---\n")
    parts.append('title: "vLLM 学习手册"\n')
    parts.append(f'subtitle: "面向大模型推理岗 · {chapter_count} 章 · 10K+ 行"\n')
    parts.append('author: "整理自 vllm-learning"\n')
    parts.append('lang: zh-CN\n')
    parts.append('documentclass: report\n')
    parts.append('toc-title: "目录"\n')
    parts.append('---\n\n')

    for path in files:
        raw = path.read_text(encoding="utf-8")
        parts.append(preprocess(raw))
        parts.append("\n\n")  # ensure separation between chapters

    return "".join(parts)


def build_pdf(combined_md: Path, out_pdf: Path) -> None:
    print(f"\n[PDF] building {out_pdf} ...")
    cmd = [
        "pandoc",
        str(combined_md),
        "-o", str(out_pdf),
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        "--top-level-division=chapter",
        "-V", "geometry:a4paper,margin=2cm",
        "-V", "mainfont=PingFang SC",
        "-V", "monofont=Menlo",
        "-V", "CJKmainfont=PingFang SC",
        "-V", "linkcolor=blue",
        "-V", "colorlinks=true",
        "-V", "fontsize=10pt",
        "--highlight-style=tango",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:\n" + res.stdout[-2000:])
        print("STDERR:\n" + res.stderr[-2000:])
        sys.exit(f"pandoc PDF failed (exit {res.returncode})")
    print(f"[PDF] done: {out_pdf} ({out_pdf.stat().st_size // 1024} KB)")


def build_epub(combined_md: Path, out_epub: Path) -> None:
    print(f"\n[EPUB] building {out_epub} ...")
    cmd = [
        "pandoc",
        str(combined_md),
        "-o", str(out_epub),
        "--toc",
        "--toc-depth=2",
        "--top-level-division=chapter",
        "--highlight-style=tango",
        "--epub-cover-image", str(SRC / "build_pdf_epub.py"),   # placeholder; skipped if can't read
    ]
    # Drop cover-image flag if can't be used as image
    cmd = [c for c in cmd if c != "--epub-cover-image" and not c.endswith("build_pdf_epub.py")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:\n" + res.stdout[-2000:])
        print("STDERR:\n" + res.stderr[-2000:])
        sys.exit(f"pandoc EPUB failed (exit {res.returncode})")
    print(f"[EPUB] done: {out_epub} ({out_epub.stat().st_size // 1024} KB)")


def main() -> None:
    check_tool("pandoc", "brew install pandoc")
    check_tool("xelatex", "brew install --cask mactex-no-gui (~4 GB)")

    DST.mkdir(parents=True, exist_ok=True)

    files = discover_files()
    print(f"Concatenating {len(files)} markdown files ...")
    combined = combine_files(files)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "combined.md"
        tmp.write_text(combined, encoding="utf-8")
        print(f"Combined: {tmp} ({len(combined) // 1024} KB)")

        build_pdf(tmp, DST / "vllm-learning.pdf")
        build_epub(tmp, DST / "vllm-learning.epub")

    print("\nAll done.")
    print(f"  PDF:  {DST / 'vllm-learning.pdf'}")
    print(f"  EPUB: {DST / 'vllm-learning.epub'}")


if __name__ == "__main__":
    main()
