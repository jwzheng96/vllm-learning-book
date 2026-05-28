#!/usr/bin/env python3
"""Convert all vLLM learning Markdown files to professionally-styled HTML.

Source:  <script dir>/                   (this directory)
Output:  <script dir>/_site/             (build artifact, gitignored)

Override via env vars: VLLM_LEARNING_SRC / VLLM_LEARNING_DST.

Style:
- Editorial book look inspired by NousResearch/hermes tutorial:
  warm paper background, deep wine-red accent, system fonts, sidebar with Parts.
- Mermaid theme matched to the warm palette
- KaTeX for math (auto-render on $$ and $)
- Lunr.js client-side full-text search
- Reading-time estimate, lesson-meta callout, per-page TOC
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import shutil
from pathlib import Path

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension
from markdown.extensions.attr_list import AttrListExtension
from markdown.extensions.sane_lists import SaneListExtension

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = Path(os.environ.get("VLLM_LEARNING_SRC", SCRIPT_DIR))
# Default to ./_site/ inside the repo (gitignored), suitable for GitHub Pages.
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

# (part_label, part_name) for each top-level section.
SECTIONS_META: dict[str, tuple[str, str]] = {
    "01-overview":              ("Part I",    "入门与架构"),
    "02-core-concepts":         ("Part II",   "核心算法"),
    "03-code-walkthrough":      ("Part III",  "源码走读"),
    "04-optimizations":         ("Part IV",   "性能优化"),
    "05-distributed":           ("Part V",    "分布式"),
    "06-interview":             ("Part VI",   "面试准备"),
    "07-hands-on":              ("Part VII",  "实操实验"),
    "08-production-deployment": ("Part VIII", "生产部署"),
    "09-advanced-features":     ("Part IX",   "应用特性"),
}

# Backward-compat alias kept for search index "section" field.
SECTION_TITLES_ZH = {k: f"{p} · {n}" for k, (p, n) in SECTIONS_META.items()}


def discover_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    readme = SRC / "README.md"
    if readme.exists():
        files.append(("README", readme))
    for section in SECTIONS:
        d = SRC / section
        if not d.is_dir():
            continue
        for md in sorted(d.glob("*.md")):
            rel = f"{section}/{md.stem}"
            files.append((rel, md))
    return files


def page_title_from_file(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


# ============================================================
#                              CSS
# ============================================================
# Hermes-inspired editorial book style: warm paper background,
# deep wine-red accent, Apple system fonts, sidebar Parts.
# ============================================================

CSS = r"""
/* ============================================================
   vLLM 学习手册 — editorial book style
   Inspired by NousResearch/hermes tutorial palette.
   ============================================================ */

:root {
    /* 纸质米白背景 */
    --bg:           #fbfaf6;
    --bg-elev:      #f3f1e8;
    --bg-elev-2:    #e9e6d9;
    --bg-code:      #f6f4eb;
    --bg-hover:     #efece1;

    /* 暖深棕黑前景 */
    --fg:           #2a2723;
    --fg-soft:      #4d4a44;
    --fg-dim:       #807c75;
    --fg-bright:    #1a1814;

    /* 主强调：深酒红 */
    --accent:       #8b1538;
    --accent-dim:   #6b1028;
    --accent-soft:  #b04658;

    /* 链接：深靛蓝 */
    --link:         #1a4d80;
    --link-hover:   #0d3a66;

    /* 边框 */
    --border:        #d6d3c4;
    --border-soft:   #e6e3d4;
    --border-strong: #c2bea9;

    /* 语义色（callout） */
    --warn:      #b85c00;
    --warn-bg:   #faf0e3;
    --info:      #2c5282;
    --info-bg:   #e8eff5;
    --good:      #2f5d3a;
    --good-bg:   #ecf3eb;
    --research:  #6b4488;
    --research-bg:#f0eaf1;
    --tip-bg:    #f5ede0;

    /* 代码语法高亮 */
    --code-keyword: #8b1538;
    --code-string:  #2e6a3e;
    --code-comment: #807c70;
    --code-fn:      #1e5a99;
    --code-name:    #a82c30;
    --code-num:     #a86420;
    --code-default: #2a2723;

    /* layout */
    --sidebar-w:  292px;
    --toc-w:      220px;
    --content-w:  760px;
    --radius:     6px;
    --radius-sm:  4px;

    --font-sans: -apple-system, BlinkMacSystemFont, "Helvetica Neue",
                 "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
                 "Source Sans 3", Helvetica, Arial, sans-serif;
    --font-mono: ui-monospace, "SF Mono", "Cascadia Code",
                 "JetBrains Mono", Menlo, Consolas, monospace;
    --font-serif: Georgia, "Iowan Old Style", "Source Serif Pro", serif;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 1.5rem; }

html, body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: var(--font-sans);
    font-size: 16px;
    line-height: 1.75;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

a {
    color: var(--link);
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-color 0.12s, color 0.12s;
}
a:hover {
    color: var(--link-hover);
    border-bottom-color: var(--link-hover);
}

/* ============================================================
   LAYOUT
   ============================================================ */

.app-layout {
    display: grid;
    grid-template-columns: var(--sidebar-w) minmax(0, 1fr) var(--toc-w);
    min-height: 100vh;
}

nav.sidebar {
    background: var(--bg-elev);
    border-right: 1px solid var(--border);
    padding: 26px 22px;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    font-size: 13.5px;
}
nav.sidebar::-webkit-scrollbar { width: 8px; }
nav.sidebar::-webkit-scrollbar-track { background: transparent; }
nav.sidebar::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }

.book-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--accent);
    margin: 0 0 4px 0;
    letter-spacing: 0.2px;
}
.book-title a { color: var(--accent); border: none; }
.book-title a:hover { color: var(--accent-dim); border: none; }

.book-sub {
    color: var(--fg-dim);
    font-size: 11px;
    margin-bottom: 22px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.home-link {
    display: inline-block;
    color: var(--fg-soft);
    font-size: 12.5px;
    margin-bottom: 14px;
    border: none;
}
.home-link:hover { color: var(--accent); border: none; }

/* Search input in sidebar */
.search-wrap { position: relative; margin: 10px 0 8px 0; }
#search-input {
    width: 100%;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 6px 10px 6px 28px;
    font-size: 12.5px;
    outline: none;
    font-family: var(--font-sans);
    transition: border-color 0.15s;
}
#search-input:focus { border-color: var(--accent); }
.search-wrap::before {
    content: "⌕";
    position: absolute;
    top: 50%;
    left: 9px;
    transform: translateY(-50%);
    font-size: 14px;
    color: var(--fg-dim);
}
#search-results {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-top: 8px;
    max-height: 340px;
    overflow-y: auto;
    display: none;
    font-size: 12.5px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
#search-results.active { display: block; }
.result {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border-soft);
    cursor: pointer;
    line-height: 1.5;
}
.result:last-child { border-bottom: none; }
.result:hover { background: var(--bg-hover); }
.result-title { font-weight: 600; color: var(--fg-bright); }
.result-section { font-size: 10.5px; color: var(--fg-dim); margin-bottom: 2px; }
.result-snippet { color: var(--fg-soft); margin-top: 4px; }
.result-snippet mark { background: var(--tip-bg); padding: 0 2px; border-radius: 2px; color: var(--fg); }
.empty { padding: 14px; color: var(--fg-dim); text-align: center; }

/* Sidebar Part groups */
.sidebar .part {
    margin-top: 18px;
    padding-top: 12px;
    border-top: 1px solid var(--border-soft);
}
.sidebar .part:first-of-type {
    margin-top: 12px;
    padding-top: 12px;
}
.sidebar .part-label {
    text-transform: uppercase;
    font-size: 10.5px;
    letter-spacing: 1.4px;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 8px;
}
.sidebar .part-name {
    font-size: 13.5px;
    color: var(--fg);
    font-weight: 600;
    margin-bottom: 8px;
}

.sidebar ol.chapters {
    list-style: none;
    margin: 0;
    padding: 0;
    counter-reset: ch;
}
.sidebar ol.chapters li {
    counter-increment: ch;
    margin: 2px 0;
    position: relative;
}
.sidebar ol.chapters a {
    display: block;
    padding: 4px 8px 4px 36px;
    color: var(--fg-soft);
    border-radius: var(--radius-sm);
    border: none;
    line-height: 1.45;
    position: relative;
}
.sidebar ol.chapters a::before {
    content: attr(data-num);
    position: absolute;
    left: 8px;
    top: 4px;
    color: var(--fg-dim);
    font-size: 11px;
    font-family: var(--font-mono);
    width: 24px;
}
.sidebar ol.chapters a:hover {
    background: var(--bg-hover);
    color: var(--accent);
    border: none;
}
.sidebar ol.chapters li.current > a {
    background: var(--tip-bg);
    color: var(--accent);
    font-weight: 600;
}
.sidebar ol.chapters li.current > a::before {
    color: var(--accent);
}

/* ============================================================
   MAIN CONTENT
   ============================================================ */

main.content {
    padding: 56px 64px 96px 64px;
    max-width: var(--content-w);
    margin: 0 auto;
}

.breadcrumb {
    font-size: 12px;
    color: var(--fg-dim);
    margin-bottom: 12px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.breadcrumb .part-tag {
    color: var(--accent);
    font-weight: 700;
}
.breadcrumb a { color: var(--fg-dim); border: none; }
.breadcrumb a:hover { color: var(--accent); border: none; }

.reading-time {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px;
    color: var(--fg-dim);
    background: var(--bg-elev);
    border: 1px solid var(--border-soft);
    border-radius: 999px;
    padding: 3px 12px;
    margin: 0 0 24px 0;
}
.reading-time::before { content: "⏱"; font-size: 13px; }

/* TOC rail (right) */
aside.toc-rail {
    position: sticky;
    top: 0;
    height: 100vh;
    padding: 56px 24px 32px 0;
    overflow-y: auto;
    font-size: 13px;
}
.toc-rail h4 {
    margin: 0 0 10px 0;
    font-size: 10.5px;
    font-weight: 700;
    color: var(--fg-dim);
    text-transform: uppercase;
    letter-spacing: 1.4px;
}
.toc-rail ul { list-style: none; padding: 0; margin: 0; }
.toc-rail li { margin: 2px 0; }
.toc-rail a {
    color: var(--fg-soft);
    display: block;
    padding: 3px 8px;
    border-left: 2px solid transparent;
    line-height: 1.45;
    border-bottom: none;
}
.toc-rail a:hover {
    color: var(--accent);
    border-left-color: var(--accent);
    border-bottom: none;
}
.toc-rail li.toc-h3 a { padding-left: 20px; font-size: 12px; }

@media (max-width: 1180px) {
    .app-layout { grid-template-columns: var(--sidebar-w) minmax(0, 1fr); }
    aside.toc-rail { display: none; }
}
@media (max-width: 900px) {
    .app-layout { grid-template-columns: 1fr; }
    nav.sidebar {
        position: static;
        height: auto;
        border-right: none;
        border-bottom: 1px solid var(--border);
    }
    main.content { padding: 30px 22px 64px 22px; }
}

/* ============================================================
   PROSE
   ============================================================ */

.markdown-body { max-width: 100%; }

.markdown-body h1, .markdown-body h2,
.markdown-body h3, .markdown-body h4 {
    color: var(--fg-bright);
    font-weight: 700;
    line-height: 1.3;
}
.markdown-body h1 {
    font-size: 32px;
    margin: 0 0 24px 0;
    padding-bottom: 14px;
    border-bottom: 2px solid var(--accent);
    letter-spacing: -0.01em;
    color: var(--accent);
}
.markdown-body h2 {
    font-size: 23px;
    margin: 48px 0 16px 0;
    letter-spacing: -0.005em;
    border-bottom: none;
    padding-bottom: 0;
}
.markdown-body h2::before {
    content: "§ ";
    color: var(--accent);
    font-weight: 700;
    font-family: var(--font-serif);
}
.markdown-body h3 {
    font-size: 17.5px;
    margin: 32px 0 12px 0;
    color: var(--fg-bright);
}
.markdown-body h4 {
    font-size: 15px;
    margin: 24px 0 10px 0;
    color: var(--fg-soft);
}

.markdown-body p { margin: 14px 0; }
.markdown-body strong { font-weight: 600; color: var(--fg-bright); }
.markdown-body em { font-style: normal; color: var(--accent); font-weight: 500; }

.markdown-body ul, .markdown-body ol {
    padding-left: 28px;
    margin: 14px 0;
}
.markdown-body li { margin: 5px 0; }
.markdown-body li > p { margin: 0.3em 0; }

.markdown-body hr {
    display: none;  /* '---' in markdown is purely a visual separator; the H2/H3 spacing is enough */
}

/* ============================================================
   TABLES (three-line academic style)
   ============================================================ */

.markdown-body table,
.markdown-body table.three-line {
    border-collapse: collapse;
    margin: 22px 0;
    width: 100%;
    font-size: 14.5px;
    line-height: 1.6;
    border: none;
    table-layout: auto;
    word-break: normal;
}
.markdown-body table.three-line {
    border-top: 2px solid var(--border-strong);
    border-bottom: 2px solid var(--border-strong);
}
.markdown-body table.three-line thead tr {
    border-bottom: 1.2px solid var(--border-strong);
    background: var(--bg-elev);
}
.markdown-body table.three-line th,
.markdown-body table.three-line td {
    border: none;
    padding: 9px 14px;
    text-align: left;
    vertical-align: top;
    overflow-wrap: anywhere;
    word-break: break-word;
    white-space: normal;
}
.markdown-body table.three-line th {
    font-weight: 700;
    color: var(--fg-bright);
    font-size: 13.5px;
    letter-spacing: 0.005em;
}
.markdown-body table.three-line tbody tr {
    border-bottom: 1px solid var(--border-soft);
}
.markdown-body table.three-line tbody tr:last-child {
    border-bottom: none;
}
.markdown-body table.three-line tbody tr:hover {
    background: var(--bg-elev);
}
.markdown-body table.three-line code {
    overflow-wrap: anywhere;
    word-break: break-all;
    white-space: normal;
}

/* ============================================================
   CODE
   ============================================================ */

.markdown-body code {
    font-family: var(--font-mono);
    background: rgba(139, 21, 56, 0.06);
    color: var(--accent);
    padding: 1.5px 6px;
    border-radius: 3px;
    font-size: 0.86em;
    border: 1px solid rgba(139, 21, 56, 0.08);
}

.markdown-body pre {
    background: var(--bg-code);
    border: 1px solid var(--border-soft);
    border-radius: 5px;
    padding: 16px 22px;
    overflow-x: auto;
    line-height: 1.6;
    font-size: 13px;
    margin: 18px 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    white-space: pre;
}
.markdown-body pre code {
    padding: 0;
    background: transparent;
    color: var(--code-default);
    font-size: inherit;
    border-radius: 0;
    border: none;
    font-weight: 400;
}

/* Pygments syntax tokens — paper light */
.codehilite .k, .codehilite .kd, .codehilite .kn, .codehilite .kr,
.codehilite .o, .codehilite .ow { color: var(--code-keyword); font-weight: 500; }
.codehilite .s, .codehilite .s1, .codehilite .s2, .codehilite .sb { color: var(--code-string); }
.codehilite .c, .codehilite .c1, .codehilite .cm { color: var(--code-comment); font-style: italic; }
.codehilite .nb, .codehilite .nf, .codehilite .nc { color: var(--code-fn); }
.codehilite .mi, .codehilite .mf { color: var(--code-num); }
.codehilite .n  { color: var(--code-default); }
.codehilite .nd { color: var(--code-num); }
.codehilite .nt { color: var(--code-string); }

/* ============================================================
   MERMAID
   ============================================================ */

pre.mermaid, .mermaid {
    background: var(--bg-elev);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 22px 18px;
    margin: 22px 0;
    text-align: center;
    overflow-x: auto;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    /* Hide raw text until mermaid renders */
    color: transparent;
    font-size: 0;
}
pre.mermaid svg, .mermaid svg {
    max-width: 100%;
    height: auto;
    font-size: 14px;
}

/* ============================================================
   BLOCKQUOTES & CALLOUTS
   ============================================================ */

.markdown-body blockquote {
    margin: 22px 0;
    padding: 10px 18px;
    border-left: 3px solid var(--accent);
    background: var(--bg-elev);
    color: var(--fg-soft);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-style: normal;
}
.markdown-body blockquote > :first-child { margin-top: 0; }
.markdown-body blockquote > :last-child { margin-bottom: 0; }

/* Lesson-meta callout (章首导读: 谁该读 / 前置阅读 / 耗时 / 学完能) */
.markdown-body blockquote.lesson-meta {
    border-left: 4px solid #a86420;
    background: linear-gradient(0deg, rgba(168,100,32,0.05), rgba(168,100,32,0.05)), var(--bg-elev);
    border-radius: var(--radius);
    padding: 14px 22px;
    margin: 18px 0 26px 0;
    color: var(--fg);
    font-size: 14.5px;
    font-style: normal;
}
.markdown-body blockquote.lesson-meta strong {
    color: #a86420;
}

/* Optional custom callouts (use in markdown via raw HTML) */
.callout {
    border-left: 4px solid var(--info);
    background: var(--info-bg);
    padding: 14px 20px;
    margin: 22px 0;
    border-radius: 0 5px 5px 0;
    font-size: 15px;
    color: var(--fg);
}
.callout.warn { border-left-color: var(--warn); background: var(--warn-bg); }
.callout.good { border-left-color: var(--good); background: var(--good-bg); }
.callout.tip  { border-left-color: var(--accent); background: var(--tip-bg); }
.callout.research { border-left-color: var(--research); background: var(--research-bg); }
.callout .label {
    font-weight: 700;
    text-transform: uppercase;
    font-size: 10.5px;
    letter-spacing: 1.4px;
    color: var(--info);
    display: block;
    margin-bottom: 6px;
}
.callout.warn .label { color: var(--warn); }
.callout.good .label { color: var(--good); }
.callout.tip  .label { color: var(--accent); }
.callout.research .label { color: var(--research); }
.callout p:first-child { margin-top: 0; }
.callout p:last-child { margin-bottom: 0; }

/* ============================================================
   MATH (KaTeX)
   ============================================================ */

.markdown-body .math-display {
    display: block;
    margin: 22px 0;
    text-align: center;
    overflow-x: auto;
}
.markdown-body .math-display .katex-display { margin: 0; }
.markdown-body .math-inline { white-space: nowrap; }

/* ============================================================
   PAGE NAV (prev / next)
   ============================================================ */

.page-nav {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 72px;
    font-size: 13.5px;
}
.page-nav a {
    padding: 14px 18px;
    background: var(--bg-elev);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    color: var(--fg);
    line-height: 1.5;
    transition: border-color 0.15s, background 0.15s;
}
.page-nav a:hover {
    border-color: var(--accent);
    background: var(--tip-bg);
    color: var(--fg-bright);
}
.page-nav .label {
    display: block;
    font-size: 10.5px;
    color: var(--fg-dim);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 3px;
    font-weight: 700;
}
.page-nav .title { font-weight: 600; }
.page-nav a.prev .title::before { content: "← "; color: var(--accent); }
.page-nav a.next { text-align: right; }
.page-nav a.next .title::after { content: " →"; color: var(--accent); }
.page-nav .placeholder {
    background: transparent;
    border: 1px dashed transparent;
}

/* ============================================================
   MISC
   ============================================================ */

.markdown-body img { max-width: 100%; height: auto; }
.markdown-body kbd {
    padding: 2px 6px;
    font-size: 12px;
    font-family: var(--font-mono);
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 3px;
    box-shadow: 0 1px 0 var(--border);
}

::selection { background: var(--tip-bg); color: var(--fg-bright); }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: var(--fg-dim); }
"""

# ============================================================
#                          TEMPLATE
# ============================================================

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} · vLLM 学习手册</title>
  <link rel="stylesheet" href="{css_link}">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
</head>
<body>
  <div class="app-layout">
    <nav class="sidebar">
      <div class="book-title"><a href="{root}README.html">vLLM 学习手册</a></div>
      <div class="book-sub">A Code-Level Tutorial</div>
      <a class="home-link" href="{root}README.html">📖 返回总目录</a>
      <div class="search-wrap">
        <input id="search-input" type="search" placeholder="全文搜索..." autocomplete="off">
      </div>
      <div id="search-results"></div>
      <div id="sidebar-nav">{sidebar}</div>
    </nav>

    <main class="content">
      {breadcrumb}
      {reading_time}
      <article class="markdown-body" id="article">{body}</article>
      {page_nav}
    </main>

    <aside class="toc-rail" id="toc-rail">
      <h4>本页大纲</h4>
      <ul id="toc-list"></ul>
    </aside>
  </div>

  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{
      startOnLoad: false,
      securityLevel: 'loose',
      theme: 'neutral',
      fontFamily: '-apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif',
      themeVariables: {{
        fontSize: '14px',
        primaryColor: '#fbfaf6',
        primaryTextColor: '#2a2723',
        primaryBorderColor: '#8b1538',
        lineColor: '#4d4a44',
        secondaryColor: '#f3f1e8',
        tertiaryColor: '#e9e6d9',
        background: '#fbfaf6',
        mainBkg: '#fbfaf6',
        secondBkg: '#f3f1e8',
        tertiaryTextColor: '#2a2723',
        nodeBorder: '#c2bea9',
        clusterBkg: '#f6f4eb',
        clusterBorder: '#d6d3c4',
        titleColor: '#8b1538',
        edgeLabelBackground: '#fbfaf6',
        textColor: '#2a2723',
        noteBkgColor: '#f5ede0',
        noteBorderColor: '#a86420',
        noteTextColor: '#2a2723'
      }},
      flowchart: {{ curve: 'basis', useMaxWidth: true, htmlLabels: true }}
    }});
    document.querySelectorAll('pre.mermaid').forEach(el => {{
      var d = document.createElement('div'); d.className = 'mermaid';
      d.textContent = el.textContent; el.replaceWith(d);
    }});
    document.querySelectorAll('pre > code.language-mermaid').forEach(el => {{
      var d = document.createElement('div'); d.className = 'mermaid';
      d.textContent = el.textContent;
      el.parentElement.replaceWith(d);
    }});
    mermaid.run();
  </script>

  <script>
    /* Auto-generate right-rail TOC from h2 / h3 */
    (function() {{
      var article = document.getElementById('article');
      var list = document.getElementById('toc-list');
      var rail = document.getElementById('toc-rail');
      if (!article || !list) return;
      var hs = article.querySelectorAll('h2, h3');
      if (hs.length < 2) {{ if (rail) rail.style.display = 'none'; return; }}
      hs.forEach(function(h) {{
        if (!h.id) {{
          h.id = h.textContent.trim()
            .toLowerCase()
            .replace(/[^\\w\\u4e00-\\u9fa5]+/g, '-')
            .replace(/^-+|-+$/g, '');
        }}
        var li = document.createElement('li');
        li.className = 'toc-' + h.tagName.toLowerCase();
        var a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent.replace(/^\\d+(\\.\\d+)*\\.?\\s*/, '');
        li.appendChild(a);
        list.appendChild(li);
      }});
    }})();
  </script>

  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body, {{
              delimiters: [
                  {{left: '$$', right: '$$', display: true}},
                  {{left: '$', right: '$', display: false}}
              ],
              throwOnError: false
          }});"></script>
  <script src="https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js"></script>
  <script>
    /* Lunr full-text search */
    (function() {{
      var rootPrefix = "{root}";
      var input = document.getElementById('search-input');
      var resultsEl = document.getElementById('search-results');
      var navEl = document.getElementById('sidebar-nav');
      var docs = null, idx = null, loadPromise = null;

      function ensureIndex() {{
        if (loadPromise) return loadPromise;
        loadPromise = fetch(rootPrefix + 'search-index.json').then(r => r.json()).then(data => {{
          docs = {{}};
          data.docs.forEach(d => {{ docs[d.id] = d; }});
          idx = lunr(function() {{
            this.ref('id');
            this.field('title', {{ boost: 10 }});
            this.field('section', {{ boost: 3 }});
            this.field('body');
            data.docs.forEach(d => this.add(d));
          }});
        }});
        return loadPromise;
      }}
      function escapeRe(s) {{ return s.replace(/[.*+?^$()|\\[\\]\\\\]/g, '\\\\$&'); }}
      function highlight(text, q) {{
        var ts = q.split(/\\s+/).filter(t => t.length > 1).map(escapeRe);
        if (!ts.length) return text;
        return text.replace(new RegExp('(' + ts.join('|') + ')', 'gi'), '<mark>$1</mark>');
      }}
      function snippet(body, q, n) {{
        if (!body) return '';
        var ts = q.split(/\\s+/).filter(t => t.length > 1);
        if (!ts.length) return body.slice(0, n) + '...';
        var lower = body.toLowerCase(), best = -1;
        ts.forEach(function(t) {{
          var j = lower.indexOf(t.toLowerCase());
          if (j >= 0 && (best === -1 || j < best)) best = j;
        }});
        if (best === -1) return body.slice(0, n) + '...';
        var s = Math.max(0, best - 40);
        var e = Math.min(body.length, s + n);
        var out = body.slice(s, e);
        if (s > 0) out = '...' + out;
        if (e < body.length) out = out + '...';
        return out;
      }}
      function render(q) {{
        if (!q) {{ resultsEl.classList.remove('active'); resultsEl.innerHTML = ''; if (navEl) navEl.style.display = ''; return; }}
        if (!idx) {{ resultsEl.innerHTML = '<div class="empty">建立索引...</div>'; resultsEl.classList.add('active'); return; }}
        var qres;
        try {{ qres = idx.query(function(qb) {{
          q.split(/\\s+/).filter(t => t.length > 0).forEach(t => {{
            qb.term(t, {{ boost: 5, usePipeline: true }});
            qb.term(t, {{ boost: 1, wildcard: lunr.Query.wildcard.TRAILING }});
          }});
        }}); }} catch (e) {{ qres = []; }}
        if (!qres.length) {{ resultsEl.innerHTML = '<div class="empty">无结果</div>'; resultsEl.classList.add('active'); return; }}
        var html = '';
        qres.slice(0, 30).forEach(function(m) {{
          var d = docs[m.ref]; if (!d) return;
          html += '<div class="result" onclick="window.location.href=\\'' + rootPrefix + d.url + '\\'">';
          html += '<div class="result-section">' + (d.section || '') + '</div>';
          html += '<div class="result-title">' + highlight(d.title, q) + '</div>';
          html += '<div class="result-snippet">' + highlight(snippet(d.body, q, 130), q) + '</div>';
          html += '</div>';
        }});
        resultsEl.innerHTML = html;
        resultsEl.classList.add('active');
        if (navEl) navEl.style.display = 'none';
      }}
      if (input) {{
        var timer = null;
        input.addEventListener('focus', ensureIndex);
        input.addEventListener('input', function() {{
          clearTimeout(timer);
          var q = input.value.trim();
          timer = setTimeout(function() {{ ensureIndex().then(function() {{ render(q); }}); }}, 120);
        }});
        input.addEventListener('keydown', function(e) {{
          if (e.key === 'Escape') {{ input.value = ''; render(''); }}
        }});
      }}
    }})();
  </script>
</body>
</html>
"""


# ============================================================
#                          HELPERS
# ============================================================

def build_sidebar(files: list[tuple[str, Path]], current_rel: str, css_depth: int) -> str:
    """Render hermes-style sidebar with Part labels + chapter numbers."""
    prefix = "../" * css_depth
    parts: list[str] = []
    titles = {rel: page_title_from_file(p) for rel, p in files}

    # Group chapters by section
    by_section: dict[str, list[tuple[str, str]]] = {}
    for rel, _ in files:
        if rel == "README":
            continue
        section = rel.split("/")[0]
        by_section.setdefault(section, []).append((rel, titles[rel]))

    for section in SECTIONS:
        if section not in by_section:
            continue
        part_label, part_name = SECTIONS_META.get(section, ("", section))
        parts.append('<div class="part">')
        parts.append(f'  <div class="part-label">{part_label}</div>')
        parts.append(f'  <div class="part-name">{part_name}</div>')
        parts.append('  <ol class="chapters">')
        for rel, title in by_section[section]:
            href = f"{prefix}{rel}.html"
            # extract leading "NN" or "NNX" number from the chapter filename
            stem = rel.split("/")[-1]
            num_match = re.match(r"^(\d+[a-z]?)", stem)
            num = num_match.group(1) if num_match else ""
            active = ' class="current"' if rel == current_rel else ""
            parts.append(
                f'    <li{active}><a data-num="{num}" href="{href}">{title}</a></li>'
            )
        parts.append('  </ol>')
        parts.append('</div>')

    return "\n".join(parts)


# Detect chapter-head meta callouts (containing 谁该读 / 前置阅读 / 耗时 / 学完能)
_LESSON_META_KEYWORDS = ("谁该读", "前置阅读", "耗时", "学完能", "学到什么", "估时")


def tag_lesson_meta_blockquotes(html: str) -> str:
    def repl(match: re.Match) -> str:
        inner = match.group(1)
        text = re.sub(r"<[^>]+>", "", inner)
        if sum(1 for kw in _LESSON_META_KEYWORDS if kw in text) >= 2:
            return f'<blockquote class="lesson-meta">{inner}</blockquote>'
        return match.group(0)
    return re.sub(r'<blockquote>(.*?)</blockquote>', repl, html, flags=re.DOTALL)


def md_to_plain_text(raw: str) -> str:
    t = re.sub(r"```.*?```", " ", raw, flags=re.DOTALL)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"^[#>\-\*\+\s]+", "", t, flags=re.MULTILINE)
    t = t.replace("|", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def estimate_reading_minutes(raw_md: str) -> int:
    text = md_to_plain_text(raw_md)
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    minutes = (cjk + other / 5) / 300
    return max(1, round(minutes))


_md_link_pattern = re.compile(r'(href|src)="([^"#]+?)\.md(#[^"]*)?"')


def rewrite_md_links(html: str) -> str:
    return _md_link_pattern.sub(lambda m: f'{m.group(1)}="{m.group(2)}.html{m.group(3) or ""}"', html)


def add_three_line_class(html: str) -> str:
    return re.sub(r'<table(\s*)>', r'<table class="three-line"\1>', html)


_table_sep_re = re.compile(r'^\s*\|[\s\-:|]+\|?\s*$')
_ordered_list_re = re.compile(r"^\d+\.\s+\S")
_unordered_list_re = re.compile(r"^[-*+]\s+\S")
_heading_re = re.compile(r"^#{1,6}\s")


def _prev_is_block_friendly(prev: str) -> bool:
    """判断前一行是否已是 block 元素（不需要再插空行）。

    注意：用正则区分列表标记 vs 粗体——`- text` / `* text` / `+ text` 是列表，
    `**bold**` 不是。原先 startswith(('-','*','+',...)) 会把粗体误判。"""
    if prev.strip() == "":
        return True
    ps = prev.lstrip()
    if ps.startswith(("|", ">", "#", "```")):
        return True
    if _unordered_list_re.match(ps) or _ordered_list_re.match(ps):
        return True
    return False


def normalize_block_boundaries(text: str) -> str:
    """python-markdown 对"段落直接接 block 元素"的容忍度因上下文而异，
    会导致 list / blockquote / table / heading 被吞进段落。

    这里在 build 阶段统一在每个 block 元素前插入空行（如果原本没有），
    覆盖 5 类：
      - 表格（必须有空行才能被 TableExtension 识别）
      - 列表（被 H3 + 段落 + - 模式坑过）
      - blockquote
      - fenced code
      - heading

    跳过 fenced code 内部，避免误伤伪代码注释里的 `#`、`-` 等。
    """
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # 进出代码 fence
        if stripped.startswith("```"):
            # fence 开始时要检查前面，结束时不需要
            opening_fence = not in_fence
            if opening_fence and i > 0:
                prev = lines[i-1]
                if not _prev_is_block_friendly(prev):
                    out.append("")
            out.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            out.append(line)
            continue

        if i == 0:
            out.append(line)
            continue
        prev = lines[i-1]

        # 已是 block-friendly 上下文，无需插
        if _prev_is_block_friendly(prev):
            out.append(line)
            continue

        # 检测当前行是否启动了新 block
        needs_blank = False
        # 表格：当前 pipe 行 + 下一行分隔
        if (stripped.startswith("|") and i + 1 < len(lines)
                and _table_sep_re.match(lines[i+1])):
            needs_blank = True
        # 列表（无序）
        elif _unordered_list_re.match(stripped):
            needs_blank = True
        # 列表（有序）
        elif _ordered_list_re.match(stripped):
            needs_blank = True
        # blockquote
        elif stripped.startswith(">"):
            needs_blank = True
        # heading
        elif _heading_re.match(stripped):
            needs_blank = True

        if needs_blank:
            out.append("")
        out.append(line)
    return "\n".join(out)


# Backward-compat alias if any external caller imports the old name.
normalize_table_blank_lines = normalize_block_boundaries


# Mermaid extraction (before markdown processing, to preserve raw text)
_mermaid_pattern = re.compile(r'```mermaid\s*\n(.*?)\n```', re.DOTALL)


def extract_mermaid_blocks(text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []
    def _sub(m: re.Match) -> str:
        blocks.append(m.group(1))
        return f"\n\n%%MERMAID_PLACEHOLDER_{len(blocks)-1}%%\n\n"
    new_text = _mermaid_pattern.sub(_sub, text)
    return new_text, blocks


def reinsert_mermaid_blocks(html: str, blocks: list[str]) -> str:
    for i, block in enumerate(blocks):
        placeholder = f"%%MERMAID_PLACEHOLDER_{i}%%"
        # HTML-escape so <br/> survives as literal text for mermaid auto-render.
        escaped = _html.escape(block, quote=False)
        rendered = f'<pre class="mermaid">{escaped}</pre>'
        html = html.replace(f"<p>{placeholder}</p>", rendered)
        html = html.replace(placeholder, rendered)
    return html


# LaTeX math extraction (preserves underscores from being eaten by markdown emphasis)
_display_math_pattern = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
_inline_math_pattern = re.compile(
    r'(?<![\w$])\$(?!\$)([^\n$]+?)\$(?!\$)(?!\w)'
)


def extract_math_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
    blocks: list[tuple[str, str]] = []
    def _display(m: re.Match) -> str:
        blocks.append(("display", m.group(1).strip()))
        return f"%%MATH_PLACEHOLDER_{len(blocks)-1}%%"
    text = _display_math_pattern.sub(_display, text)
    def _inline(m: re.Match) -> str:
        blocks.append(("inline", m.group(1)))
        return f"%%MATH_PLACEHOLDER_{len(blocks)-1}%%"
    text = _inline_math_pattern.sub(_inline, text)
    return text, blocks


def reinsert_math_blocks(html: str, blocks: list[tuple[str, str]]) -> str:
    for i, (kind, content) in enumerate(blocks):
        placeholder = f"%%MATH_PLACEHOLDER_{i}%%"
        escaped = _html.escape(content, quote=False)
        if kind == "display":
            rendered = f'<div class="math-display">$${escaped}$$</div>'
            html = html.replace(f"<p>{placeholder}</p>", rendered)
        else:
            rendered = f'<span class="math-inline">${escaped}$</span>'
        html = html.replace(placeholder, rendered)
    return html


# ============================================================
#                       CONVERT ONE PAGE
# ============================================================

def convert_one(rel: str, path: Path, files: list[tuple[str, Path]]) -> str:
    css_depth = rel.count("/")
    css_link = ("../" * css_depth) + "style.css"
    root_prefix = "../" * css_depth

    raw = path.read_text(encoding="utf-8")
    raw_no_mermaid, mermaid_blocks = extract_mermaid_blocks(raw)
    raw_no_math, math_blocks = extract_math_blocks(raw_no_mermaid)
    raw_normalized = normalize_block_boundaries(raw_no_math)

    md = markdown.Markdown(
        extensions=[
            TableExtension(),
            FencedCodeExtension(),
            CodeHiliteExtension(guess_lang=False, css_class="codehilite", noclasses=False),
            TocExtension(permalink=False),
            AttrListExtension(),
            SaneListExtension(),
        ]
    )
    body = md.convert(raw_normalized)
    body = reinsert_math_blocks(body, math_blocks)
    body = reinsert_mermaid_blocks(body, mermaid_blocks)
    body = rewrite_md_links(body)
    body = add_three_line_class(body)
    body = tag_lesson_meta_blockquotes(body)

    title = page_title_from_file(path)

    if rel == "README":
        breadcrumb = ""
        reading_time = ""
    else:
        section = rel.split("/")[0]
        part_label, part_name = SECTIONS_META.get(section, ("", section))
        breadcrumb = (
            f'<div class="breadcrumb">'
            f'<a href="{root_prefix}README.html">vLLM 学习手册</a> · '
            f'<span class="part-tag">{part_label}</span> · {part_name}'
            f'</div>'
        )
        minutes = estimate_reading_minutes(raw)
        reading_time = f'<div class="reading-time">预计阅读 {minutes} 分钟</div>'

    # Build prev/next page navigation
    ordered_rels = [r for r, _ in files]
    idx_p = ordered_rels.index(rel)
    file_map = dict(files)
    nav_parts = ['<nav class="page-nav">']
    if idx_p > 0:
        prev_rel = ordered_rels[idx_p - 1]
        prev_title = page_title_from_file(file_map[prev_rel])
        prev_href = f"{root_prefix}{prev_rel}.html"
        nav_parts.append(
            f'<a class="prev" href="{prev_href}">'
            f'<span class="label">上一篇</span>'
            f'<span class="title">{prev_title}</span></a>'
        )
    else:
        nav_parts.append('<span class="placeholder"></span>')
    if idx_p + 1 < len(ordered_rels):
        next_rel = ordered_rels[idx_p + 1]
        next_title = page_title_from_file(file_map[next_rel])
        next_href = f"{root_prefix}{next_rel}.html"
        nav_parts.append(
            f'<a class="next" href="{next_href}">'
            f'<span class="label">下一篇</span>'
            f'<span class="title">{next_title}</span></a>'
        )
    else:
        nav_parts.append('<span class="placeholder"></span>')
    nav_parts.append('</nav>')
    page_nav = "\n".join(nav_parts)

    sidebar = build_sidebar(files, rel, css_depth)
    return TEMPLATE.format(
        title=title,
        css_link=css_link,
        root=root_prefix,
        sidebar=sidebar,
        body=body,
        breadcrumb=breadcrumb,
        reading_time=reading_time,
        page_nav=page_nav,
    )


# ============================================================
#                        SEARCH INDEX
# ============================================================

def build_search_index(files: list[tuple[str, Path]]) -> dict:
    docs = []
    for rel, path in files:
        raw = path.read_text(encoding="utf-8")
        section_key = "README" if rel == "README" else rel.split("/")[0]
        section_label = "目录首页" if section_key == "README" else SECTION_TITLES_ZH.get(section_key, section_key)
        docs.append({
            "id": rel,
            "url": f"{rel}.html",
            "title": page_title_from_file(path),
            "section": section_label,
            "body": md_to_plain_text(raw)[:4000],
        })
    return {"docs": docs}


# ============================================================
#                            MAIN
# ============================================================

def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    (DST / "style.css").write_text(CSS, encoding="utf-8")
    (DST / ".nojekyll").write_text("", encoding="utf-8")

    files = discover_files()
    print(f"Found {len(files)} markdown files")

    index_data = build_search_index(files)
    (DST / "search-index.json").write_text(
        json.dumps(index_data, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  search-index.json ({len(index_data['docs'])} docs)")

    for rel, path in files:
        html = convert_one(rel, path, files)
        out_path = DST / f"{rel}.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

    (DST / "index.html").write_text(
        (DST / "README.html").read_text(encoding="utf-8"), encoding="utf-8"
    )

    print(f"\nDone. Open: file://{DST}/index.html")


if __name__ == "__main__":
    main()
