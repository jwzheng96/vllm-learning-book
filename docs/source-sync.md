# vLLM 上游源码同步手册

本手册把“教程仍然对应当前 vLLM 源码”拆成两道门：机器可判定的源码契约，以及必须由人完成的语义复核。前者通过不代表后者已经完成。

## 1. 为什么禁止手写行号

`file.py:123` 会在无关代码插入后静默漂移：链接仍能打开，却可能指向完全不同的逻辑。教程改用紧邻链接的 `vllm-source` 指令，保存稳定路径、Python 限定符号或唯一文本锚点；刷新工具在锁定的完整 commit 上重新计算行号，并在路径、符号或锚点失效时直接失败。

源码链接必须指向 `source.lock.json` 的不可变 40 位 SHA，不能指向会移动的 `main`。

## 2. 四种源码契约

路径级引用适合目录地图或只需要说明“实现位于这里”的场景：

```markdown
<!-- vllm-source: {"path":"vllm/v1/attention/backends"} -->
[Attention 后端目录](https://github.com/vllm-project/vllm/tree/FULL_SHA/vllm/v1/attention/backends)
```

Python 符号引用优先使用最窄、稳定的限定名：

```markdown
<!-- vllm-source: {"path":"vllm/v1/core/sched/scheduler.py","symbol":"Scheduler.schedule"} -->
[Scheduler.schedule](https://github.com/vllm-project/vllm/blob/FULL_SHA/vllm/v1/core/sched/scheduler.py#L1)
```

当结论依赖函数内某条具体语句时，在符号作用域内增加唯一锚点：

```markdown
<!-- vllm-source: {"path":"vllm/v1/core/sched/scheduler.py","symbol":"Scheduler.schedule","anchor":"scheduled_running_reqs.append(request)"} -->
[调度入选语句](https://github.com/vllm-project/vllm/blob/FULL_SHA/vllm/v1/core/sched/scheduler.py#L1)
```

CUDA/C++ 没有 Python AST 符号解析，使用文件内唯一文本锚点；连续几行确有共同语义时才设置 `span`：

```markdown
<!-- vllm-source: {"path":"csrc/attention/paged_attention_v1.cu","anchor":"namespace vllm {","span":1} -->
[PagedAttention CUDA 锚点](https://github.com/vllm-project/vllm/blob/FULL_SHA/csrc/attention/paged_attention_v1.cu#L1)
```

指令与链接必须各占一行且物理相邻。不要把多处不同概念压成一条逗号分隔的行号列表。

## 3. 常用命令

```bash
# 机器契约：路径/符号/URL/版本锁/inventory
python3 -m tools.source_sync validate --profile contracts --require-committed

# 人工内容复核也全部完成后使用
python3 -m tools.source_sync validate --profile full --require-committed

# 比较旧基线与候选并生成影响报告
python3 -m tools.source_sync impact \
  --baseline OLD_FULL_SHA --candidate NEW_FULL_SHA \
  --output artifacts/source-sync/latest-impact.md

# 子模块已切到候选后，重算锁文件、README 和所有托管链接
python3 -m tools.source_sync refresh \
  --candidate-sha NEW_FULL_SHA \
  --validated-at 2026-07-20T18:00:00Z \
  --report artifacts/source-sync/latest-impact.md

# 查询官方 main；只报告，不修改仓库
python3 -m tools.source_sync check-upstream
```

## 4. 常见错误

| 错误 | 含义 | 处理 |
| --- | --- | --- |
| `source path does not exist` | 文件被移动或删除 | 用候选源码确认新权威路径，不能随便锚到同名文件 |
| `Python symbol not found` | 类/函数改名或层级变化 | 追踪调用链后改为当前最窄限定名，并复核正文结论 |
| `anchor ... found 0` | 文本已删除或改写 | 找到当前等价语句；若语义消失，删除或改写正文 |
| `anchor ... found 2` | 锚点不再唯一 | 加 Python `symbol` 缩小作用域，或选择更唯一的完整语句 |
| `managed source URL is stale` | 链接仍指向旧 SHA/旧行 | 运行 `refresh`，不要手改 URL |
| `uncovered source file` | 上游变更没有章节 source area 接住 | 补精确 `source_areas` 或记录为何无需教程覆盖 |
| `review must be complete` | 章节还没针对当前锁定 SHA 做人工复核 | 更新正文并完成 `content-review.toml`，不能只改状态位 |

## 5. 手动同步与回滚

1. 记录当前子模块 SHA，确认教程与独立 vLLM 工作区没有未保存改动。
2. 从 `https://github.com/vllm-project/vllm.git` 获取官方 `main` 的完整候选 SHA。
3. 只在快进关系成立时原子更新独立 vLLM 的本地 `main`；不要 checkout、rebase 或 force 更新用户分支。
4. 将教程子模块 detached checkout 到同一 SHA。
5. 生成 impact，运行 refresh、contracts 单测与 HTML 构建。
6. 按影响报告逐章复核，最后运行 full profile。

若候选不适合发布，用一个新的提交把 gitlink、`source.lock.json`、README 版本块和托管链接恢复到上一个已验证 SHA。不要用 `reset --hard`，这样回滚原因和审计轨迹都能保留。

## 6. 人工复核清单

源码契约门只证明“能定位”：

- 路径、符号和文本锚点在当前 SHA 唯一存在；
- 链接、子模块、gitlink、锁文件和 README 一致；
- 没有手写数字行号或 inventory 漏项。

章节语义门还必须逐章证明“说得对”：

- 默认值、CLI 参数、指标名与含义仍一致；
- Mermaid 请求流、进程边界和失败路径仍一致；
- 命令在声明的环境中可执行，回滚步骤使用受支持接口；
- 初学者前置知识、生产取舍和面试追问都有闭环；
- `content-review.toml` 记录当前 SHA、UTC 时间、检查项和必要备注。

## 7. GPU 证据规则

`hardware_verified = []` 的意思是“本章没有在当前锁定 SHA 和明确硬件环境上复测”，不是“默认已验证”。没有当前硬件记录时，只能把数字标为示例、估算或上游公开结果，不能写成实测结论。

GPU 记录至少包含 GPU 型号与数量、驱动/CUDA、模型与精度、关键 vLLM 参数、输入输出长度分布、并发、采样窗口和原始结果位置。手动 GPU 工作流产生的证据必须和章节 review 使用同一个源码 SHA。
