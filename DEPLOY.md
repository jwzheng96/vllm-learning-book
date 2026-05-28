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
python3 -m pip install --user markdown pygments

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

每次 `git push` 自动：① CI 跑 `build_html.py` 生成 `_site/` ② 上传到 Pages artifact ③ 部署到 Pages。**你永远只 commit markdown，HTML 从不进 git**。

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
├── 01-overview/ ... 09-advanced-features/   ← 46 章源 markdown
├── build_html.py               ← HTML 构建脚本
├── build_pdf_epub.py           ← PDF + EPUB 构建脚本
├── deploy_gh_pages.sh          ← 手动 gh-pages 部署（备用）
├── DEPLOY.md                   ← 本文件
├── .gitignore                  ← 忽略 _site/ 等构建产物
├── .github/workflows/
│   └── pages.yml               ← Actions 自动构建 + 部署
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

## 5. 常见问题

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
