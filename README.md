# vLLM 学习手册

[![下载 PDF](https://img.shields.io/badge/📥_PDF-下载-success.svg)](https://jwzheng96.github.io/vllm-learning-book/vllm-learning.pdf) [![下载 EPUB](https://img.shields.io/badge/📥_EPUB-下载-success.svg)](https://jwzheng96.github.io/vllm-learning-book/vllm-learning.epub) [![在线阅读](https://img.shields.io/badge/🌐_在线阅读-jwzheng96.github.io-8b1538)](https://jwzheng96.github.io/vllm-learning-book/) [![Pages CI](https://github.com/jwzheng96/vllm-learning-book/actions/workflows/pages.yml/badge.svg)](https://github.com/jwzheng96/vllm-learning-book/actions/workflows/pages.yml) [![Upstream sync](https://github.com/jwzheng96/vllm-learning-book/actions/workflows/sync-upstream.yml/badge.svg)](https://github.com/jwzheng96/vllm-learning-book/actions/workflows/sync-upstream.yml)

> 📥 **[下载 PDF / EPUB](https://jwzheng96.github.io/vllm-learning-book/vllm-learning.pdf)**（推荐）— ElegantBook 排版，离线阅读体验最佳；也可[在线阅读](https://jwzheng96.github.io/vllm-learning-book/)（侧栏导航 + 全文搜索 + Mermaid 渲染）。

**vLLM = PagedAttention + Continuous Batching + Prefix Caching**——本书围绕这个核心公式，用 60 章把大模型推理引擎从论文原理讲到 K8s 生产部署。每章用语义锚点对照锁定 commit 的 vLLM 源码，可以"读笔记 ↔ 跳源码"无缝切换。

| 📚 **60 章** | 📝 **24K+ 行** | 📊 **94 张图** |
| :---: | :---: | :---: |
| 论文 → 生产全链路 | 源码级走读 | Mermaid 架构图 |

适合 **系统补课 / 业务接入 / 性能优化 / 底层贡献**；不适合零基础（先读 [`00-prerequisites.md`](01-overview/00-prerequisites.md)）。

📖 **详细目录、学习路径、源码地标、自检清单** → 见 [`CONTENTS.md`](CONTENTS.md)

---

## 📑 内容速览

| Part | 主题 | 章数 | 一句话核心 |
| :--: | --- | :--: | --- |
| I | [总览](CONTENTS.md#part-1) | 6 | 前置概念、架构、V0→V1、项目结构、进程与 IPC |
| II | [核心概念](CONTENTS.md#part-2) | 5 | PagedAttention、Continuous Batching、KV 管理、Prefix Caching、Chunked Prefill |
| III | [源码走读](CONTENTS.md#part-3) | 10 | 入口→调度→KV→Runner→Attention→CUDA→输入输出全链路 |
| IV | [优化](CONTENTS.md#part-4) | 5 | 量化、投机解码、CUDA Graph、编译内核、存算比推导 |
| V | [分布式](CONTENTS.md#part-5) | 5 | TP/PP/EP、PD 分离、专家并行、Context Parallel、万卡集群 |
| VI | [工程问答](CONTENTS.md#part-6) | 4 | 30 道高频题、系统设计、容量计算、模拟面试 |
| VII | [实操](CONTENTS.md#part-7) | 8 | 环境、调试、实验、Profiling、API 服务、基准、调优、Capstone |
| VIII | [生产部署](CONTENTS.md#part-8) | 12 | 架构、路由、网关、弹性、SLO、可靠性、监控、安全、升级 |
| IX | [应用特性](CONTENTS.md#part-9) | 5 | 采样、结构化输出、多模态、LoRA、Embedding |

> 🧭 不知道从哪开始？4 条学习路径（30 分钟理解 / 源码主线 / 工业实战 / 面试冲刺）见 [`CONTENTS.md`](CONTENTS.md#-学习路径)。

---

## 🔧 构建与部署

所有产物默认输出到 `_site/`：

```bash
python3 build_html.py      # → _site/  (搜索 + Mermaid + 阅读时间)
python3 build_pdf_epub.py  # → _site/vllm-learning.{pdf,epub}  (ElegantBook)
./deploy_gh_pages.sh <repo-url>
```

<details>
<summary><b>📖 编译细节</b>（pandoc / xelatex / ElegantBook / mmdc）</summary>

PDF 使用 [ElegantBook](https://github.com/ElegantLaTeX/ElegantBook)（green 主题），封面由 `cover.tex`（TikZ 矢量）生成，排版定制见 `preamble.tex`。Mermaid 图表通过 [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) 预渲染为 PNG。EPUB 使用 `epub.css`（酒红主题），封面从 PDF 首页提取。

**macOS：**

```bash
brew install pandoc poppler librsvg
brew install --cask basictex
npm i -g @mermaid-js/mermaid-cli
pip install cairosvg
export PATH="/Library/TeX/texbin:$PATH"
sudo tlmgr update --self
sudo tlmgr install elegantbook fvextra tcolorbox newunicodechar framed \
  comment csquotes biblatex trimspaces environ mdframed zref needspace \
  tex-gyre newtx tikzfill pdfcol etoc fixtounicode mathrsfs rsfs
```

**Linux / CI**：`texlive-xetex + fonts-noto-cjk` + 同样的 `tlmgr` 包，字体自动回退 Noto Sans CJK SC。完整流程见 [`DEPLOY.md`](DEPLOY.md)。

</details>

<details>
<summary><b>📊 源码版本锁定状态</b>（自动化维护）</summary>

<!-- vllm-version:start -->
- Validated vLLM: `b23bd73f540175f9e117eaee5029cd7d8df63964`
- Upstream committed: `2026-07-20T15:32:54+00:00`
- Validated: `2026-07-20T17:53:34Z`
- Latest candidate: `b23bd73f540175f9e117eaee5029cd7d8df63964`
- Candidate lag: `2268` commits
- Impact report: [artifacts/source-sync/latest-impact.md](artifacts/source-sync/latest-impact.md)
<!-- vllm-version:end -->

每章用 fail-closed 语义门禁对照锁定 commit 的 vLLM 源码。完整流程见 [`docs/source-sync.md`](docs/source-sync.md)。

</details>

---

## 🤖 自动排障 Skill：`vllm-doctor`

仓库内置 Claude Code skill `vllm-doctor`，把第 06-07-08 章的 incident playbook 编成 agent 可自动跑的 7 阶段流程。缺少证据时 fail closed。

```bash
cp -r .claude/skills/vllm-doctor ~/.claude/skills/   # 安装
# 在 Claude Code 里输入 /vllm-doctor（需 export VLLM_NAMESPACE / PROM_URL / KUBECONFIG）
```

覆盖 8 类事故：KV 抢占级联、NCCL hang、GPU OOM、重试雪崩、prefix cache 塌方、冷启动、输出质量异常、LoRA 抖动。详见 [`.claude/skills/vllm-doctor/SKILL.md`](.claude/skills/vllm-doctor/SKILL.md)。

---

## 🤝 贡献

发现 `file_path:line_number` 失效了？vLLM 主分支变化快，欢迎 PR 修正。想新加一章？沿用"章首导读 + 正文 + 小结/自检/下一步"模板。

---

## ⭐ Star History

<a href="https://star-history.com/#jwzheng96/vllm-learning-book&Date">
  <img alt="Star History" src="https://img.shields.io/github/stars/jwzheng96/vllm-learning-book?style=social" width="auto" />
</a>

<!-- star-history.com 的动态 SVG 图表（api.star-history.com 当前 503，恢复后替换回：
<a href="https://star-history.com/#jwzheng96/vllm-learning-book&Date">
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jwzheng96/vllm-learning-book&type=Date" width="100%" />
</a>
-->

---

**开始读 [`01-overview/01-what-is-vllm.md`](01-overview/01-what-is-vllm.md)。** 或者从 [`00-prerequisites.md`](01-overview/00-prerequisites.md) 铺前置。
