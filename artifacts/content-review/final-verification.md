# vLLM Learning Final Verification

本报告记录 60 章课程刷新在最终提交前后的可复核证据。静态完成不等同于硬件实测；未执行的验证在下文明确列出。

## Locked Official Source

| 项目 | 值 |
| --- | --- |
| 官方仓库 | `https://github.com/vllm-project/vllm` |
| 官方分支 | `main` |
| 完整 SHA | `b23bd73f540175f9e117eaee5029cd7d8df63964` |
| 上游提交时间 | `2026-07-20T15:32:54Z` |
| full 验证记录时间 | `2026-07-21T05:49:50Z` |
| 本地锁文件 / submodule | 均为 `b23bd73f540175f9e117eaee5029cd7d8df63964` |

最终复查通过 GitHub 官方 commits API 完成；API 返回的 `main` SHA 与 `source.lock.json` 相同。`rlocal` 当时不能经 `git ls-remote` 连接 `github.com:443`，因此没有用失败的 Git 传输结果替代官方 API 结果，也没有移动已经最新的锁定版本。

## Inventory and Review Coverage

| 扫描项 | 结果 |
| --- | ---: |
| `curriculum.toml` / 顶层章节 Markdown | 60 / 60 |
| 章节正文行数（60 个章节） | 24,876 |
| 发布输入行数（README + 60 章） | 25,203 |
| 语义源码契约 | 243 |
| `content-review.toml` review rows | 60 |
| `status = "reviewed"` | 60 |
| `reviewed_commit` 等于锁定 SHA | 60 |
| source / command / metric / diagram 四项均完成 | 60 / 60 |
| 未解析源码契约 | 0 |
| 未管理的遗留数字行号引用 | 0 |
| 未登记或缺失的章节 / review row | 0 |

历史影响报告 [`../source-sync/latest-impact.md`](../source-sync/latest-impact.md) 比较旧基线 `27b85d2084c48f9b12f8cfd6638a56fe9b257635` 与锁定 SHA，标记当时库存中的 50 章受影响、0 个契约未解析，并保留 2,078 个课程 source area 之外的上游变更文件。最终官方 `main` 复查没有越过锁定 SHA，因此本次最终刷新新增的 affected chapter 数是 **0**；无需再次生成同一组差异。

## Automated Verification

所有命令均在 `rlocal` 的工作树中运行；退出码均为 0，除非明确标为预期 RED。

| 命令 | 结果 |
| --- | --- |
| `python3 -m unittest discover -s tests/source_sync -v` | 67 tests，`OK`，5.080 s |
| `bash -n scripts/gpu-validation.sh` | 通过 |
| GPU 脚本非法模型名、非法 TP、缺少 API key 三个负例 | 均以预期的参数错误码 2 退出 |
| `python3 -m tools.source_sync validate --profile full` | `OK: full source and content review is valid` |
| `VLLM_LEARNING_DST="$site_dir" python3 build_html.py` | 61 个 Markdown 文档进入搜索索引 |
| `find "$site_dir" -name "*.html"` | 62 个 HTML 文件，满足 README + 60 章页面要求 |
| `test -s "$site_dir/search-index.json"` | 通过；搜索索引 401,814 bytes |
| `git diff --check` | 通过 |

在输入审阅中发现模型 ID 可用 `--help` 形式穿过原正则。回归测试先得到预期 RED（脚本返回 1，而非参数错误 2），随后将首字符限制为字母或数字；目标测试、脚本语法和完整测试集均转为绿色。

GitHub Actions 的 `validate`、`pages` 与上游同步候选门禁都已切换到 `--profile full`；发布在构建和上传之前额外要求 `--require-committed`。手动 GPU workflow 只接受 `workflow_dispatch`，使用带 `nvidia-gpu` 标签的 self-hosted runner，不安装依赖，并在成功或失败时上传证据目录。

## Publication and Hardware Boundary

`command -v pandoc` 与 `command -v xelatex` 都没有返回路径。按计划未安装大型 TeX 工具链，也未宣称 PDF/EPUB 构建通过；HTML 是本环境完成的发布构建。

No current-SHA GPU hardware run was performed; performance examples are labeled expected or illustrative.

因此当前没有可索引的 GPU run ID。`scripts/gpu-validation.sh` 和 `.github/workflows/gpu-validation.yml` 已提供后续人工硬件验证入口；证据会记录教程/源码 SHA、GPU 拓扑、Python/vLLM/PyTorch/CUDA 环境、服务日志、models/chat/stream/metrics 响应和最终状态，并在归档前清除 API key。

## Preserved Bugfix Work

独立 vLLM checkout 保持在干净的 `bugfix/parallel-config-size-validation` 分支，并跟踪 `fork/bugfix/parallel-config-size-validation`。下列三个提交均由 `git merge-base --is-ancestor <sha> HEAD` 验证可从当前分支头到达：

- `004b8601c97e1c9f3d18085e6e9827b567f5462c`
- `d1cd0162d316837c144b00b8294230dfaa8029ce`
- `090fd61d806461ef83df9f5f59ea2b41ea6778c7`（当前分支头）

本课程刷新没有改写、合并或删除该分支。

## Known External Constraints

- `rlocal` 到 `github.com:443` 的 Git HTTPS 连接在最终检查时超时；官方 GitHub commits API 可访问并返回了相同的锁定 SHA。
- 当前没有满足 workflow runner contract 的 NVIDIA GPU 硬件执行记录，因而所有硬件结论保持未验证。
- `pandoc` 与 `xelatex` 缺失，PDF/EPUB 留待具备工具链的发布环境执行。

这些约束不改变源码契约、内容审查、测试和 HTML 发布门禁的完成状态，也不被表述成硬件或 PDF/EPUB 成功。
