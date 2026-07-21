# 构建与部署

本目录的 Markdown 学习笔记是**唯一的源**。HTML 是构建产物（gitignored），可一键生成 3 种产物：

| 产物 | 命令 | 输出 |
| --- | --- | --- |
| **HTML 站**（GitHub Pages 用） | `python3 build_html.py` | `_site/` |
| **PDF 单本** | `python3 build_pdf_epub.py` | `_site/vllm-learning.pdf` |
| **EPUB 单本** | `python3 build_pdf_epub.py` | `_site/vllm-learning.epub` |
| **GitHub Pages 自动部署** | `git push` | GitHub Actions 跑 `build_html.py`，推到 `gh-pages` |

> **重要**：`_site/` 已在 `.gitignore` 里。**永远不要 commit 它**——CI 在每次 push 时自动重建，commit 进去会与 CI 输出冲突。

---

## 一次性环境准备

```bash
# Python 依赖
python3 -m pip install --user -r requirements-docs.txt

# PDF / EPUB 工具链（可选；CI 不构建）
# macOS
brew install pandoc
brew install --cask mactex-no-gui     # 4 GB，含 xelatex（CJK 支持）
# Linux
# sudo apt install pandoc texlive-xetex texlive-fonts-recommended fonts-noto-cjk
```

---

## 1. 本地 HTML 站

```bash
python3 build_html.py
open _site/index.html
```

特性：

- **Editorial book style**：米白纸质背景 + 酒红强调 + Apple system fonts（对标 hermes 教程）
- Sidebar 按 **Part I–IX** 分组，当前章高亮
- 每页右上角显示**预计阅读时间**（300 字/分钟估算）
- 章首"导读" blockquote 自动识别为 `lesson-meta` 样式（暖橙色卡片）
- 「全文搜索」输入框：Lunr.js 客户端索引，支持中文 + 英文
- ```` ```mermaid ```` 代码块自动渲染为 SVG（neutral 主题，与正文配色一致）
- `$...$` / `$$...$$` 数学公式由 KaTeX 自动渲染
- 三线表 / 自定义 callout（`.callout.tip/warn/good/research`）/ code highlight
- 响应式：< 1180px 隐藏右侧 TOC；< 900px sidebar 转顶部
- 自带 `.nojekyll`，GitHub Pages 直接可用
- 路径用脚本目录解析；想换位置设环境变量 `VLLM_LEARNING_SRC` / `VLLM_LEARNING_DST`

---

## 2. PDF / EPUB（可选）

```bash
python3 build_pdf_epub.py
```

需要 `pandoc + xelatex`。脚本会：

1. 按 README → 01-overview → ... → 09-advanced-features 顺序拼接全部 .md
2. 剔除跨文件相对链接（保留文字）
3. 加 YAML 元数据（title / author / lang / documentclass）
4. xelatex 用 PingFang SC（macOS 自带）；Linux 把字体改为 Noto Sans CJK SC

输出落到 `_site/vllm-learning.pdf` / `_site/vllm-learning.epub`，与 HTML 同目录。

---

## 3. GitHub Pages 部署（推荐：GitHub Actions 自动）

### 方案 A · GitHub Actions（推荐，零维护）

仓库已带 `.github/workflows/pages.yml`。流程：

```bash
# 1. 在 GitHub 建空仓库 yourname/vllm-learning-book
# 2. 本地 init + push（不必先 build HTML，CI 会自动跑）
cd path/to/vllm-learning
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:yourname/vllm-learning-book.git
git push -u origin main

# 3. 仓库设置 → Pages → Source: "GitHub Actions"
# 4. 推完几分钟后访问 https://yourname.github.io/vllm-learning-book/
```

每次 `git push` 自动：① 初始化锁定的 vLLM 子模块 ② 跑 source-sync 单测和 committed full review ③ 构建 `_site/` ④ 上传并部署。`--profile full --require-committed` 会同时要求锁定 SHA / 子模块、语义源码锚点、60 章 inventory、零 unmanaged source line，以及 `content-review.toml` 每章 review 全部在当前 SHA 完成；任一项失败都不会发布。**你永远只 commit Markdown 和契约元数据，HTML 从不进 git**。

上游版本刷新、语义源码锚点、影响报告和人工 review 的操作见 [`docs/source-sync.md`](docs/source-sync.md)。每周 `sync-upstream.yml` 只维护一个候选 PR，不会自动合并或部署。

### 方案 B · 手动推 gh-pages 分支（备用）

```bash
python3 build_html.py
./deploy_gh_pages.sh git@github.com:yourname/vllm-learning-book.git
# 仓库设置：Settings → Pages → Branch: gh-pages / (root)
```

仅当方案 A 不能用（如私有 runner 受限）时考虑。

---

## 4. 仓库结构

```
vllm-learning-book/             ← GitHub 仓库根
├── README.md                   ← 书的首页（hermes 风格 hero）
├── 01-overview/ ... 09-advanced-features/   ← 60 章源 Markdown
├── build_html.py               ← HTML 构建脚本
├── build_pdf_epub.py           ← PDF + EPUB 构建脚本
├── deploy_gh_pages.sh          ← 手动 gh-pages 部署（备用）
├── DEPLOY.md                   ← 本文件
├── .gitignore                  ← 忽略 _site/ 等构建产物
├── .github/workflows/
│   ├── validate.yml            ← PR 单测、契约与 HTML 构建
│   ├── sync-upstream.yml       ← 每周/手动生成上游候选 PR
│   ├── pages.yml               ← full 门禁通过后构建 + 部署
│   └── gpu-validation.yml      ← 手动 self-hosted GPU 证据工作流
├── scripts/gpu-validation.sh   ← 启动服务、探测 API、脱敏并归档证据
├── _site/                      ← 🚫 构建产物，gitignored
│   ├── index.html
│   ├── style.css
│   ├── search-index.json
│   ├── .nojekyll
│   └── 01-overview/ ... 09-advanced-features/
└── vllm/                       ← submodule → vllm-project/vllm
                                 ←  仅供源码引用核对，不进 build
```

---

## 5. 手动 GPU 验证（可选，不属于普通 CI）

`.github/workflows/gpu-validation.yml` 只有 `workflow_dispatch`，运行在带 `[self-hosted, linux, x64, nvidia-gpu]` labels 的 runner。runner contract：

- 已安装与锁定子模块兼容的 vLLM、PyTorch、CUDA / driver 与 `curl`；workflow 不执行 dependency / driver install；
- 能访问用户选择的 model ID，并有足够 GPU / storage；
- GitHub Actions secret `VLLM_API_KEY` 已配置；
- runner 上的 8000 端口未被其他健康服务占用。

从 Actions 页面选择 model ID 和 TP `1|2|4|8`。脚本验证参数、记录 tutorial / vLLM SHA、实际 package / CUDA / PyTorch / GPU / topology，启动带 API key 的 `vllm serve`，在 300 秒内等待 `/health`，再保存 `/v1/models`、确定性 chat、streaming 和 `/metrics` 的 headers / 脱敏 body。成功或失败都会上传 `artifacts/gpu-validation/<UTC>-<source-sha>/`；API key 不进入 command evidence、response artifact 或 server log artifact。

这条 workflow 不是“硬件已经验证”的声明。只有 artifact ID 被登记进最终验证报告并与当前 source SHA 一致时，相关章节才能标注 GPU verified。

---

## 6. 常见问题

**Q: HTML 暗黑模式不切换？**
A: 看一下浏览器 localStorage 是否被禁用。脚本通过 `localStorage.setItem('vllm-learning-theme', 'dark|light')` 持久化。

**Q: 搜索没反应？**
A: 第一次聚焦搜索框时会异步加载 `search-index.json` + 构建 Lunr 索引（~200 ms）。看 DevTools Network 面板确认 fetch 成功。

**Q: PDF 中文显示乱码？**
A: 默认用 `PingFang SC`（macOS）。Linux 改成 `Noto Sans CJK SC`，编辑 `build_pdf_epub.py` 把 `mainfont`/`CJKmainfont` 改掉。

**Q: GitHub Pages 显示空白？**
A: 确认 `.nojekyll` 文件存在于站点根（已自动生成）。否则 Pages 会用 Jekyll 处理，部分文件会被忽略。

**Q: 我想加 PWA / 离线访问？**
A: 加一个 service worker 即可。可以基于 `style.css` + `search-index.json` 缓存所有页面。需要时再扩展。
