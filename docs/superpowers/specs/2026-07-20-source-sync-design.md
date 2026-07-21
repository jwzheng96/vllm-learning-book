# vLLM 源码与教程持续同步设计

**状态：** 已批准  
**日期：** 2026-07-20  
**范围：** `vllm/` 主线更新、`vllm-learning/` 源码引用、上游变更检测与发布门禁

## 1. 背景与问题

`vllm-learning/` 是关于上游 `vllm-project/vllm` 的中文源码教程。当前教程通过 Git submodule 锁定 vLLM，但教程中大量使用人工维护的 `file_path:line_number`。上游代码移动后，行号可能指向错误语句，单纯检查“文件和行号存在”不能证明教程仍然正确。

2026-07-20 盘点时的事实：

- `vllm-learning/vllm` 锁定 `27b85d2084c48f9b12f8cfd6638a56fe9b257635`。
- 官方 `main` 在盘点时已前进到 `4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc`。
- Pages workflow 没有 checkout submodule，也没有源码引用校验。
- 独立 `vllm/` 仓库当前处于用户的 bugfix 分支，它比本地 `main` 多 3 个提交，更新主线时不得覆盖该分支。

## 2. 目标

1. 以 `vllm-project/vllm` 的官方 `main` 为唯一上游源码真相。
2. 本次将独立 `vllm/` 仓库的 `main` 和教程 submodule 更新到最终验证时观测到的官方最新 commit。
3. 用稳定的语义锚点替代手写行号，但读者仍看到精确行号和不可变 commit 链接。
4. 每周检查上游 `main`，对变更进行影响分析并创建候选升级 PR。
5. 候选版本只有在引用、内容复核、测试和构建都通过后才能成为发布基线。
6. 上游破坏时保留上一个正确的线上版本，同时给出可操作的失效报告。

## 3. 非目标

- 不自动改写中文技术结论，也不根据名称相似度猜测新符号。
- 不自动合并上游升级 PR。
- 不修改上游 vLLM 源码以适配教程。
- 不手工编辑 `vllm-learning-html/` 或 `_site/` 构建产物。
- 不声称“永远与每个 `main` commit 零延迟一致”。系统保证每周生成候选并显式暴露差距，发布基线始终是已验证 commit。

## 4. 版本策略与真相来源

### 4.1 上游与本地仓库

- 上游仓库：`https://github.com/vllm-project/vllm`。
- 跟踪分支：`main`。
- 独立 `vllm/` 仓库只 fast-forward 它的本地 `main`；用户当前的 `bugfix/parallel-config-size-validation` 及其提交完整保留。
- `vllm-learning/vllm` 始终以 detached commit 形式指向已验证基线。

### 4.2 版本锁

`source.lock.json` 是面向工具和读者的显式版本记录。下面使用盘点时的官方 commit 展示完整格式；实施时所有值由工具写入实际数据：

```json
{
  "schema_version": 1,
  "repository": "https://github.com/vllm-project/vllm",
  "branch": "main",
  "commit": "4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc",
  "committed_at": "2026-07-17T11:19:05Z",
  "validated_at": "2026-07-20T15:00:00Z"
}
```

submodule gitlink 是实际源码内容锁；`source.lock.json` 提供时间、分支和仓库语义。校验器要求两者 SHA 完全一致。README 的版本 badge 和版本说明由工具从锁文件刷新，不形成第三个可独立修改的真相来源。

## 5. 语义引用契约

### 5.1 Markdown 格式

作者在普通 Markdown 链接前放置单行 JSON 指令：

```markdown
<!-- vllm-source: {"path":"vllm/v1/core/sched/scheduler.py","symbol":"Scheduler.schedule","anchor":"token_budget = self.max_num_scheduled_tokens","span":1} -->
[`Scheduler.schedule` 的 token budget](https://github.com/vllm-project/vllm/blob/4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc/vllm/v1/core/sched/scheduler.py#L1-L1)
```

指令是机器可读契约，紧随其后的链接是 GitHub Markdown、HTML、PDF 和 EPUB 都能直接消费的读者界面。`refresh` 只重写 URL，保留作者编写的链接文本。

### 5.2 字段语义

- `path` 必填，相对 submodule 根目录，不得越过边界，不得是符号链接。
- `symbol` 可选，用于 Python 类、方法、函数或嵌套定义；通过 Python AST 解析定义范围。
- `anchor` 可选，是必须唯一命中的精确文本。同时有 `symbol` 时只在符号范围内查找，适用于精确定位函数内某条语句。
- `span` 可选，为从定位行开始的正整数行数，默认为 1。
- 如果只有 `path`，生成不带行号的文件或目录链接。
- 如果只有 `symbol`，链接指向定义首行。
- 对 C++、CUDA、CMake 和配置文件使用 `anchor`；不在首版引入 tree-sitter 依赖。
- 不支持“第 N 次出现”。多重匹配必须通过更稳定的 `symbol` 或更具体的 `anchor` 消除。

### 5.3 解析结果

每条引用解析为：

- 正规化后的相对路径；
- 开始行和结束行（如果有）；
- 锁定 commit 的 GitHub blob/tree URL；
- 章节路径与引用位置；
- 用于影响报告的符号和源码区域。

## 6. 章节与源码区域映射

`curriculum.toml` 是机器可读的章节清单。每章必须有且只有一条记录，至少包含：

```toml
[[chapter]]
path = "03-code-walkthrough/02-scheduler.md"
title = "Scheduler 源码走读"
level = "intermediate"
tracks = ["internals", "interview"]
environments = ["no-gpu", "nvidia-gpu"]
source_areas = ["vllm/v1/core/sched/**", "vllm/config/scheduler.py"]
```

`source_areas` 使系统能发现“上游语义变了，但某个旧锚点仍然存在”。只要候选 commit 与基线 commit 之间的文件命中某章 `source_areas`，该章就进入人工语义复核清单。

## 7. 工具边界

工具代码位于 `tools/source_sync/`，按职责拆分：

- `models.py`：版本锁、指令、解析结果和影响项的数据类型。
- `markdown.py`：扫描 Markdown、解析 JSON 指令、定位紧随链接并执行定点重写。
- `resolver.py`：Python AST 符号解析、文本锚点解析和路径边界校验。
- `versions.py`：submodule SHA、锁文件、上游 commit 和 README 版本块的一致性。
- `impact.py`：读取 git diff 文件集，结合引用和 `curriculum.toml` 生成章节影响报告。
- `cli.py`：提供 `validate`、`refresh`、`impact` 和 `check-upstream` 四个稳定子命令。

界面约束：

- `validate` 只读，任何错误返回非零状态。
- `refresh` 只改写锁文件、README 版本块、语义引用 URL 和机器生成的影响报告。
- `refresh` 必须幂等；对同一 commit 连续执行两次，第二次不产生 diff。
- 工具不执行 Markdown 中的任何内容，不访问 submodule 之外的锚点路径。

## 8. 每周同步数据流

1. GitHub Actions 每周一定时触发，并支持 `workflow_dispatch` 指定候选 SHA。
2. checkout 教程仓库和 submodule，查询官方 `main` 的完整 SHA。
3. 若 SHA 与 `source.lock.json` 相同，以“无变更”成功退出。
4. 将 submodule 移到候选 SHA，生成 `git diff BASELINE_SHA..CANDIDATE_SHA` 文件集，其中两个参数都是已验证的 40 位 SHA。
5. 执行 `refresh`，重新解析引用并更新版本信息。
6. 生成影响报告：上游 commit 范围、变更文件、失效锚点、命中章节、未覆盖源码区域和待复核项。
7. 运行静态校验、单元测试、链接检查和 HTML 构建。
8. 不论候选是否全部通过，都可以创建或刷新同一个升级 PR；PR 在失败时保持红色检查，不可发布。
9. 人工复核影响章节的解释、配置、指标和实验，修复后重跑全部门禁。
10. 合并后的 submodule SHA 成为新的已验证基线，Pages 才发布新版。

## 9. 失败处理

| 失败 | 行为 |
| --- | --- |
| 文件删除或路径越界 | 失败，列出章节、指令和旧路径 |
| Python 符号不存在 | 失败，不做模糊匹配 |
| 文本锚点零命中 | 失败，显示所在符号和锚点摘要 |
| 文本锚点多重命中 | 失败，显示所有候选行，要求收窄指令 |
| 锚点有效但 `source_areas` 变更 | 生成必须人工勾选的语义复核项 |
| 候选主线无法构建 | 保留旧发布基线，PR 持续报错 |
| 无法访问上游 | workflow 失败并保留上次状态，不写入空 SHA |

## 10. 验证和测试

### 10.1 单元测试

- 正确解析顶层函数、类、同步/异步方法和嵌套定义。
- `anchor` 在符号范围内解析，不被函数外的相同文本干扰。
- 覆盖零命中、多命中、语法错误、路径越界、非 UTF-8 和非法 `span`。
- 仅路径引用、单行引用和多行 span 生成正确 GitHub URL。
- Markdown 重写不改变指令之外的字节。
- 同一候选 commit 连续 refresh 两次结果完全一致。

### 10.2 仓库级门禁

- submodule gitlink、submodule HEAD、`source.lock.json` 和 README 版本块 SHA 相同。
- 每条 `vllm-source` 指令都唯一解析，紧随链接中的 commit 和行号与解析结果相同。
- 教程 Markdown 不再包含未登记的 vLLM 手写行号引用；代码块中用于教学的输出数字不受此规则影响。
- `curriculum.toml` 与实际章节文件一一对应，路径、track、level、environment 和 `source_areas` 合法。
- 内部 Markdown 链接有效，`build_html.py` 在干净输出目录成功构建。
- Pages workflow checkout submodule，并在上传 artifact 前运行同一套 `validate`。

### 10.3 GPU 验证边界

普通 GitHub-hosted CI 不伪装 GPU 结果。真实推理、性能和多卡实验使用手动或 self-hosted NVIDIA GPU workflow，保存 vLLM SHA、GPU 型号、驱动、CUDA、模型、参数和执行日期。没有 GPU runner 时，必须标记“未进行当前 SHA 硬件复验”，不得编造基准数据；静态正确性和教程交付不以拥有 GPU runner 为前提。

## 11. 安全与权限

- 定时 workflow 只需 `contents: write` 和 `pull-requests: write`，Pages 部署权限与同步 job 分离。
- 候选 SHA 必须是 40 位十六进制并可从官方上游获得。
- JSON 指令只用标准库解析，不使用 `eval`。
- 自动化不合并 PR，不触发上游 vLLM 仓库写操作。

## 12. 发布语义

站点同时显示：

- 已验证 vLLM commit 及时间；
- 教程验证时间；
- 官方 `main` 候选 commit（如有）；
- 已验证基线与候选之间的 commit 数；
- 当前是否存在失败的同步 PR。

“与最新主线同步”的严格含义是：在记录的最终验证时间，官方 `main` SHA 与已验证 SHA 相同，全部门禁通过。之后新上游提交会在下一次每周或手动 workflow 中进入候选。

## 13. 完成标准

1. 独立 `vllm/` 仓库的本地 `main` fast-forward 到最终验证时的官方 `main`，原 bugfix 分支与 3 个自有提交可达。
2. `vllm-learning/vllm` 和 `source.lock.json` 指向同一 commit。
3. 教程内所有 vLLM 行号引用都迁移到语义契约，没有未登记遗留。
4. 上游变更能自动映射到受影响章节，失效锚点能产生精确错误。
5. 定时和手动同步 workflow 可生成候选 PR，且失败候选不会被 Pages 发布。
6. 单元测试、仓库校验、内部链接检查和 HTML 构建全部通过。
