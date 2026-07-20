# vLLM Learning Baseline Content Audit

本报告是扩充至 60 章之前的冻结基线。数字来自 `tools.source_sync` CLI、`curriculum.toml`、`content-review.toml`、源码契约解析结果和 Markdown 仓库扫描，不代表已经完成章节级语义复核。

## Locked Source Version

| 项目 | 值 |
| --- | --- |
| 教程先前基线 | `27b85d2084c48f9b12f8cfd6638a56fe9b257635` |
| 当前锁定 / 候选 commit | `b23bd73f540175f9e117eaee5029cd7d8df63964` |
| 上游 commit 时间 | `2026-07-20T15:32:54+00:00` |
| 本地契约验证时间 | `2026-07-20T17:53:34Z` |
| 两个 commit 的跨度 | `2268` commits |
| 上游差异文件 | `3743` |

锁定值来自 `source.lock.json`；完整差异见 [`artifacts/source-sync/latest-impact.md`](../source-sync/latest-impact.md)。

## Inventory Reconciliation

- `curriculum.toml` 中有 **50** 条章节记录。
- 九个章节目录中有 **50** 个顶层章节 Markdown 文件；不存在未登记章节，也不存在指向缺失文件的记录。
- `content-review.toml` 中有 **50** 条一一对应的复核记录。
- README 在本轮之前曾宣称 **49 章**，与文件系统和清单少算 1 章；随后又改为手写 50。构建器现在只从 `curriculum.toml` 取顺序和数量，避免手写总数再次漂移。
- README 不是章节，不计入 50；HTML/PDF/EPUB 仍将 README 作为出版物首页。

## Pending Reviews by Part

| 分区 | 清单章节 | Pending | Reviewed |
| --- | ---: | ---: | ---: |
| `01-overview` | 6 | 6 | 0 |
| `02-core-concepts` | 5 | 5 | 0 |
| `03-code-walkthrough` | 8 | 8 | 0 |
| `04-optimizations` | 5 | 5 | 0 |
| `05-distributed` | 5 | 5 | 0 |
| `06-interview` | 2 | 2 | 0 |
| `07-hands-on` | 4 | 4 | 0 |
| `08-production-deployment` | 10 | 10 | 0 |
| `09-advanced-features` | 5 | 5 | 0 |
| **合计** | **50** | **50** | **0** |

## Affected Chapters from Upstream Diff

上游影响分析将 **50/50** 章全部标记为 affected。按分区展开如下；章节的精确顺序以 `curriculum.toml` 为准。

- `01-overview`（6）：`00-prerequisites`、`01-what-is-vllm`、`02-architecture`、`03-v0-vs-v1`、`04-project-structure`、`05-process-and-ipc-internals`。
- `02-core-concepts`（5）：`01-paged-attention`、`02-continuous-batching`、`03-kv-cache-management`、`04-prefix-caching`、`05-chunked-prefill`。
- `03-code-walkthrough`（8）：`01-entry-points`、`02-scheduler`、`02b-scheduling-policies`、`03-kv-cache-manager`、`04-model-runner`、`05-attention-backends`、`06-cuda-kernels`、`07-model-architectures`。
- `04-optimizations`（5）：`01-quantization`、`02-speculative-decoding`、`03-cudagraph-and-compile`、`04-compilation-internals`、`05-roofline-and-arithmetic-intensity`。
- `05-distributed`（5）：`01-tp-pp-ep`、`02-disaggregated`、`03-expert-parallel-deep-dive`、`04-context-parallel`、`05-large-scale-cluster-inference`。
- `06-interview`（2）：`01-common-questions`、`02-system-design`。
- `07-hands-on`（4）：`01-setup`、`02-trace-a-request`、`03-mini-experiments`、`04-profiling-and-debugging`。
- `08-production-deployment`（10）：`01-deployment-architectures`、`02-smart-routing-and-load-balancing`、`03-gateway-and-service-mesh`、`04-autoscaling-and-capacity`、`05-slo-and-observability`、`06-reliability-and-failure-modes`、`07-incident-playbook`、`08-monitoring-cookbook`、`09-vllm-doctor-skill`、`10-gpu-utilization-and-tail-latency`。
- `09-advanced-features`（5）：`01-sampling-and-logits`、`02-structured-output`、`03-multimodal`、`04-lora-serving`、`05-embedding-and-pooling`。

全量 affected 的原因是基线跨越 2268 个 commit，并覆盖引擎、调度、KV cache、模型执行、分布式、量化和服务入口等主要 source area；它意味着“必须复核”，不等同于“每章都已发现错误”。

## Unmanaged or Unresolved Source Contracts

| 扫描项 | 数量 | 处置 |
| --- | ---: | --- |
| 语义源码契约 | 216 | 已纳入刷新器 |
| 带命名 symbol 的契约 | 214 | 在锁定源码树中解析 |
| 带文本 anchor 的契约 | 75 | 在锁定源码树中解析 |
| 未解析契约 | 0 | 契约门禁通过 |
| 遗留 `file.py:123` / `file.cu:123` 数字行号 | 0 | 已迁移为语义锚点 |
| 未被任何章节 source area 覆盖的变更文件 | 2078 | 保留在影响报告中；多数是 CI、测试、模型或教程当前范围外的上游文件 |

“未覆盖变更文件”不是未解析引用：前者提示课程覆盖边界，后者会直接让契约门禁失败。

## Named Symbols/Flags/Metrics Requiring Review

仓库扫描给出以下需要在章节复核时逐项确认的命名表面：

- **Named symbols：214 个契约**。重点族包括 `LLM` / `AsyncLLM` / `EngineCore` 入口，`Scheduler.schedule`，`KVCacheManager` / `BlockPool`，`GPUModelRunner`，attention backend 选择，sampling、spec decode、parallel state、LoRA 和 multimodal 管线。契约解析成功只证明符号存在，不能替代参数、返回值和行为语义复核。
- **CLI flags：85 个唯一值**。扫描集合覆盖服务与模型（如 `--model`、`--served-model-name`、`--dtype`、`--quantization`）、缓存与调度（如 `--gpu-memory-utilization`、`--block-size`、`--max-num-batched-tokens`、`--scheduling-policy`）、并行（如 `--tensor-parallel-size`、`--pipeline-parallel-size`、`--data-parallel-size`、`--enable-expert-parallel`）、编译与 attention（如 `--compilation-config`、`--cudagraph-capture-sizes`、`--attention-backend`）以及 benchmark 参数。每条命令都要在当前 SHA 的 CLI 定义或 `--help` 中核对。
- **指标原始 token：74 个唯一值**。扫描包含 `vllm:` 前缀的 counter、gauge、histogram 以及 `_bucket` / `_count` / `_sum` 派生名；复核时必须区分 exporter 暴露名、PromQL 派生表达式和文档示例，不把正则命中的派生后缀误当独立指标。
- **环境变量：14 个唯一值**：`VLLM_ATTENTION_BACKEND`、`VLLM_COMPILE`、`VLLM_DOCTOR_FIXTURE`、`VLLM_LOGGING_LEVEL`、`VLLM_LOG_BATCHSIZE_INTERVAL`、`VLLM_LOG_MODEL_INSPECTION`、`VLLM_LOG_STATS_INTERVAL`、`VLLM_NAMESPACE`、`VLLM_NVTX_LOGGING`、`VLLM_TORCH_COMPILE_CACHE_DIR`、`VLLM_TORCH_PROFILER_DIR`、`VLLM_TRACE_FUNCTION`、`VLLM_USE_PRECOMPILED`、`VLLM_USE_V1`。

这些计数是文本扫描的待复核队列，不是“当前版本全部有效”的声明；章节 review row 只有在命令、指标、图和源码契约均核对后才能转为 `reviewed`。

## GPU Verification Availability

- 当前 `content-review.toml` 的 50 条记录均为 `hardware_verified = []`。
- 本轮基线没有当前 SHA 的 NVIDIA GPU、多 GPU、RDMA 或 Kubernetes 实测记录可索引。
- 因此当前只能声明 **static source review**；不得显示“GPU verified”或类似硬件徽章。
- 后续硬件验证记录必须至少包含：锁定 commit、时间、GPU/驱动/CUDA 环境、完整命令、数据集或请求形状、原始结果位置和结论。记录未进入索引前，不改变章节硬件状态。

## Planned Ten Chapters

| 分区 | 新章节 | 补齐的学习链路 |
| --- | --- | --- |
| `03-code-walkthrough` | `08-input-processing-and-tokenization.md` | API 输入、tokenization、请求归一化到引擎输入 |
| `03-code-walkthrough` | `09-output-processing-and-streaming.md` | sampler 输出、detokenization、流式响应与取消 |
| `07-hands-on` | `05-serve-openai-api.md` | 从零启动并验证第一个 OpenAI-compatible API 服务 |
| `07-hands-on` | `06-benchmark-methodology.md` | 可复现 benchmark、负载模型、TTFT/TPOT/吞吐解释 |
| `07-hands-on` | `07-tuning-playbook.md` | 从症状到旋钮、实验矩阵和回归门禁 |
| `07-hands-on` | `08-production-capstone.md` | 从容量目标到上线、观测、故障演练的综合项目 |
| `08-production-deployment` | `11-security-and-multi-tenancy.md` | 鉴权、限流、租户隔离、模型与数据安全 |
| `08-production-deployment` | `12-upgrades-rollbacks-and-compatibility.md` | 版本兼容、灰度、回滚和状态迁移 |
| `06-interview` | `03-capacity-and-troubleshooting-drills.md` | 容量计算、指标诊断和追问题 |
| `06-interview` | `04-mock-interview-and-rubric.md` | 分层模拟题、评分表和改进闭环 |

新增完成后，目标库存为 **60 章**。新增文件必须同时进入 `curriculum.toml` 与 `content-review.toml`，否则共享发现函数会让构建失败。
