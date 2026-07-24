#!/usr/bin/env python3
"""Build a single combined PDF and EPUB from all vllm-learning markdown files.

PDF:  ElegantBook document class (green theme) with a TikZ vector cover,
      Unicode fallback fonts, framed code blocks (tcolorbox), and auto-
      wrapping code lines (fvextra). Diagrams (Mermaid) are pre-rendered
      to PNG and embedded as real images.

EPUB: Pandoc EPUB3 with a hand-tuned stylesheet (epub.css, wine-red theme),
      math via MathML, and a cover JPEG extracted from the PDF's first page
      via pdftoppm (so the EPUB cover always matches the PDF cover).

Requirements (auto-checked):
- pandoc          (brew install pandoc / apt install pandoc)
- xelatex         (brew install --cask basictex/mactex, or apt install texlive-xetex)
- mmdc            (npm i -g @mermaid-js/mermaid-cli)   — for Mermaid → PNG
- pdftoppm        (brew install poppler / apt install poppler-utils) — EPUB cover
- ElegantBook + deps (tlmgr install elegantbook fvextra tcolorbox newunicodechar)

Source:  $VLLM_LEARNING_SRC  (default: this directory)
Output:  $VLLM_LEARNING_DST/vllm-learning.{pdf,epub}  (default: ./_site/)

Override paths with VLLM_LEARNING_SRC / VLLM_LEARNING_DST env vars.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.source_sync.inventory import discover_chapter_files

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = Path(os.environ.get("VLLM_LEARNING_SRC", SCRIPT_DIR))
DST = Path(os.environ.get("VLLM_LEARNING_DST", SCRIPT_DIR / "_site"))

PREAMBLE_TEX = SRC / "preamble.tex"
COVER_TEX = SRC / "cover.tex"
EPUB_CSS = SRC / "epub.css"

# The book has 9 Parts (top-level chapters). Each Part directory maps to a
# Chinese chapter title used as \chapter{} in the PDF/EPUB.
SECTIONS: list[tuple[str, str]] = [
    ("01-overview", "总览：入门与架构"),
    ("02-core-concepts", "核心概念：PagedAttention · Batching · KV Cache"),
    ("03-code-walkthrough", "源码走读：引擎全链路"),
    ("04-optimizations", "性能优化：量化 · 投机解码 · 编译"),
    ("05-distributed", "分布式：多卡与多机推理"),
    ("06-interview", "工程问答：面试与自测"),
    ("07-hands-on", "实操实验：从环境到生产 Capstone"),
    ("08-production-deployment", "生产部署：上线与运维"),
    ("09-advanced-features", "应用特性：采样 · 多模态 · LoRA"),
]


# ============================================================
#                     FILE DISCOVERY + PREP
# ============================================================

def check_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"ERROR: '{name}' not found. Install with: {install_hint}")


def discover_files_by_part() -> list[tuple[str, list[Path]]]:
    """Return files grouped by Part directory, preserving section order.

    README.md is excluded (it's the repo front-page, not a chapter).
    Returns [(part_dir, [chapter_files...]), ...].
    """
    grouped: list[tuple[str, list[Path]]] = []
    for section, _title in SECTIONS:
        section_dir = SRC / section
        if not section_dir.is_dir():
            continue
        chapter_files = sorted(
            f for f in section_dir.glob("*.md") if f.name != "README.md"
        )
        if chapter_files:
            grouped.append((section, chapter_files))
    return grouped


def _demote_headings(md_text: str) -> str:
    """Demote all Markdown headings by one level (# → ##, ## → ###, etc.).

    This lets each Part directory become a \\chapter while each .md file's
    original H1 becomes a \\section within that chapter.
    """
    lines = md_text.splitlines()
    result = []
    for line in lines:
        if re.match(r'^#{1,5} ', line):
            result.append('#' + line)
        else:
            result.append(line)
    return '\n'.join(result)


# Strip cross-file relative .md links. Keep external links as-is.
_inline_md_link = re.compile(r"\[([^\]]+)\]\(([^)\s]+\.md)(#[^)]*)?\)")

# Neutralize badge images (shields.io / CI badges) — replace the inner
# ![alt](badge-url) with just the alt text, so the outer link survives as
# a clickable text link. Prevents pandoc from trying to fetch online SVGs.
_inline_badge_img = re.compile(
    r"!\[([^\]]*)\]\(https?://"
    r"(?:img\.shields\.io|github\.com/[^/]+/[^/]+/actions)"
    r"[^)]*\)"
)


def preprocess(md_text: str) -> str:
    md_text = _inline_badge_img.sub(lambda m: m.group(1) or "badge", md_text)
    md_text = _inline_md_link.sub(lambda m: m.group(1), md_text)
    return md_text


# ============================================================
#                     MERMAID PRE-RENDER
# ============================================================

# Same pattern as build_html.py — keep in sync.
_mermaid_pattern = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)


def render_mermaid_blocks(md_text: str, img_dir: Path) -> str:
    """Replace ```mermaid blocks with PNG image references via mmdc."""
    matches = list(_mermaid_pattern.finditer(md_text))
    if not matches:
        return md_text

    mmdc = shutil.which("mmdc")
    if mmdc is None:
        sys.exit(
            "ERROR: 'mmdc' (mermaid-cli) not found, but the markdown contains "
            f"{len(matches)} mermaid diagram(s). Install with: "
            "npm i -g @mermaid-js/mermaid-cli"
        )

    img_dir.mkdir(parents=True, exist_ok=True)
    mermaid_config = img_dir / "puppeteer-config.json"
    mermaid_config.write_text('{"args": ["--no-sandbox"]}\n', encoding="utf-8")

    rendered = md_text
    offset = 0
    count = 0
    failures = 0
    for m in matches:
        count += 1
        src = m.group(1).strip()
        png_name = f"mermaid-{count}.png"
        mmd_path = img_dir / f"mermaid-{count}.mmd"
        png_path = img_dir / png_name
        mmd_path.write_text(src, encoding="utf-8")

        cmd = [
            mmdc, "-i", str(mmd_path), "-o", str(png_path),
            "-b", "transparent", "-w", "1600", "-p", str(mermaid_config),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not png_path.exists():
            failures += 1
            err_line = (res.stderr.strip().splitlines()[-1]
                        if res.stderr.strip() else "unknown error")
            print(f"  [warn] mermaid-{count} failed to render: {err_line}")
            replacement = (
                f"> *图 {count}：Mermaid 图表无法在 PDF 中渲染"
                f"（在线版可正常显示）。源码如下：*\n\n"
                f"``````text\n{src}\n``````\n"
            )
        else:
            replacement = f"![图 {count}]({png_name})"

        start, end = m.span()
        rendered = rendered[:start + offset] + replacement + rendered[end + offset:]
        offset += len(replacement) - (end - start)

    ok = count - failures
    if failures:
        print(f"  mermaid: {ok}/{count} rendered, {failures} fell back to source code")
    else:
        print(f"  rendered {count} mermaid diagram(s) → {img_dir}")
    return rendered


# ============================================================
#                        COMBINE
# ============================================================

def combine_files(grouped: list[tuple[str, list[Path]]]) -> str:
    section_count = len(grouped)
    chapter_count = sum(len(files) for _, files in grouped)
    source_line_count = 0
    for _, files in grouped:
        for path in files:
            source_line_count += len(path.read_text(encoding="utf-8").splitlines())

    parts: list[str] = []
    parts.append("---\n")
    parts.append('title: "vLLM 学习手册"\n')
    parts.append(
        f'subtitle: "{section_count} 大章 · {chapter_count} 小节 · '
        f'{source_line_count} 行"\n'
    )
    parts.append('author: "整理自 vllm-learning"\n')
    parts.append('lang: zh-CN\n')
    parts.append('---\n\n')

    # Each Part directory = one \chapter. Each .md file inside = a \section
    # (original H1 demoted to H2 so pandoc --top-level-division=chapter
    # maps: Part → \chapter, file H1 → \section, file H2 → \subsection).
    section_meta = dict(SECTIONS)
    for section, files in grouped:
        title = section_meta.get(section, section)
        # # maps to \chapter via --top-level-division=chapter.
        # Numbered chapters give sections the 1.1, 1.2, 2.1 prefix.
        parts.append(f"# {title}\n\n")
        for path in files:
            raw = path.read_text(encoding="utf-8")
            raw = preprocess(raw)
            raw = _demote_headings(raw)
            parts.append(raw)
            parts.append("\n\n")

    return "".join(parts)


# ============================================================
#                          BUILDERS
# ============================================================

def build_pdf(combined_md: Path, out_pdf: Path, resource_dir: Path) -> None:
    """Build PDF with ElegantBook via pandoc + xelatex."""
    print(f"\n[PDF] building {out_pdf} ...")

    # Build pandoc args. preamble.tex and cover.tex are referenced by
    # absolute path so the build works from any CWD.
    extra_args = []
    if PREAMBLE_TEX.exists():
        extra_args += ["-H", str(PREAMBLE_TEX)]
    if COVER_TEX.exists():
        extra_args += ["--include-before-body", str(COVER_TEX)]

    cmd = [
        "pandoc",
        str(combined_md),
        "-o", str(out_pdf),
        "--from", "markdown+lists_without_preceding_blankline",
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--top-level-division=chapter",
        f"--resource-path={resource_dir}",
        "-V", "documentclass=elegantbook",
        "-V", "classoption=lang=cn",
        "-V", "classoption=nofont",
        "-V", "classoption=green",
        "-V", "classoption=device=normal",
        "--highlight-style=kate",
        "--columns=80",
    ] + extra_args

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:\n" + res.stdout[-2000:])
        print("STDERR:\n" + res.stderr[-2000:])
        sys.exit(f"pandoc PDF failed (exit {res.returncode})")
    print(f"[PDF] done: {out_pdf} ({out_pdf.stat().st_size // 1024} KB)")


def extract_pdf_cover(pdf_path: Path, out_jpg: Path) -> Path | None:
    """Extract the first page of the PDF as a JPEG for the EPUB cover."""
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        print("  [info] pdftoppm not found — skipping EPUB cover extraction")
        return None
    stem = str(out_jpg.with_suffix(""))
    res = subprocess.run(
        [pdftoppm, "-f", "1", "-singlefile", "-jpeg", "-r", "160",
         str(pdf_path), stem],
        capture_output=True, text=True,
    )
    if res.returncode == 0 and out_jpg.exists():
        print(f"  EPUB cover extracted from PDF: {out_jpg.name}")
        return out_jpg
    print(f"  [warn] pdftoppm failed: {res.stderr.strip()[:200]}")
    return None


def build_epub(combined_md: Path, out_epub: Path, resource_dir: Path,
               cover_jpg: Path | None) -> None:
    """Build EPUB3 with the hand-tuned stylesheet."""
    print(f"\n[EPUB] building {out_epub} ...")

    cmd = [
        "pandoc",
        str(combined_md),
        "-o", str(out_epub),
        "--from", "markdown+lists_without_preceding_blankline",
        "--to", "epub3",
        "--standalone",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--top-level-division=chapter",
        "--mathml",
        "--split-level=1",
        f"--resource-path={resource_dir}",
        "--highlight-style=kate",
        "--metadata", "lang=zh-CN",
        "--metadata", "title=vLLM 学习手册",
        "--metadata", "author=整理自 vllm-learning",
        "--metadata",
        f"identifier=https://github.com/jwzheng96/vllm-learning-book",
    ]
    if EPUB_CSS.exists():
        cmd += [f"--css={EPUB_CSS}"]
    if cover_jpg is not None and cover_jpg.exists():
        cmd += ["--epub-cover-image", str(cover_jpg)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:\n" + res.stdout[-2000:])
        print("STDERR:\n" + res.stderr[-2000:])
        sys.exit(f"pandoc EPUB failed (exit {res.returncode})")
    print(f"[EPUB] done: {out_epub} ({out_epub.stat().st_size // 1024} KB)")


# ============================================================
#                            MAIN
# ============================================================

def main() -> None:
    check_tool("pandoc", "brew install pandoc / apt install pandoc")
    check_tool("xelatex", "brew install --cask basictex "
               "or apt install texlive-xetex texlive-fonts-recommended")
    check_tool("mmdc", "npm i -g @mermaid-js/mermaid-cli")

    DST.mkdir(parents=True, exist_ok=True)

    grouped = discover_files_by_part()
    total_files = sum(len(f) for _, f in grouped)
    print(f"Concatenating {len(grouped)} Parts, {total_files} sections ...")
    combined = combine_files(grouped)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Render mermaid PNGs into the SAME directory as combined.md so that
        # xelatex can find them via bare filenames.
        combined = render_mermaid_blocks(combined, tmp)

        combined_md = tmp / "combined.md"
        combined_md.write_text(combined, encoding="utf-8")
        print(f"Combined: {combined_md} ({len(combined) // 1024} KB)")

        # Build PDF first (EPUB cover is extracted from it).
        pdf_path = DST / "vllm-learning.pdf"
        build_pdf(combined_md, pdf_path, tmp)

        # Extract EPUB cover from PDF page 1.
        cover_jpg = extract_pdf_cover(pdf_path, DST / "cover.jpg")

        epub_path = DST / "vllm-learning.epub"
        build_epub(combined_md, epub_path, tmp, cover_jpg)

    print("\nAll done.")
    print(f"  PDF:  {pdf_path}")
    print(f"  EPUB: {epub_path}")


if __name__ == "__main__":
    main()
