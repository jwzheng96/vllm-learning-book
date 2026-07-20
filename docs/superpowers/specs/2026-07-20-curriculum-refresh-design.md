# vLLM 最新主线学习手册升级设计

**状态：** 已批准  
**日期：** 2026-07-20  
**依赖：** `2026-07-20-source-sync-design.md`  
**范围：** 全部现有教程的最新源码复核，以及新手、工业实战和面试闭环的内容补齐

## 1. 背景与库存

`vllm-learning/` 当前已有从基础概念到生产部署的大量中文内容。2026-07-20 库存盘点得到 50 个章节 Markdown，共约 21,523 行；README 却声称“49 章”，说明当前不仅源码行号会漂移，连章节库存也没有机器可验证的单一清单。

现有手册已覆盖 PagedAttention、Scheduler、KV cache、Attention backend、量化、投机解码、分布式、观测和故障手册。本次升级不重写整本书，而是：

1. 对所有现有章节按同一个最新 vLLM commit 做证据化复核；
2. 修正过时路径、符号、配置、指标、命令和行为描述；
3. 用完整学习路径串联已有内容；
4. 补齐“首次服务—基准—调优—上线—故障—复盘—面试表达”的缺口。

## 2. 目标读者与学习成果

### 2.1 目标读者

- 了解 Python，但没有 LLM 推理引擎经验的初学者。
- 需要把模型接入线上服务的应用或平台工程师。
- 需要定位 TTFT、TPOT、吞吐、显存或长尾问题的性能工程师。
- 准备大模型推理、AI Infra 或服务端面试的学习者。

### 2.2 完成后能力

学习者应能：

1. 区分 prefill/decode、TTFT/TPOT/ITL、计算密集/带宽密集，完成 KV cache 和吞吐的基础估算。
2. 在支持的环境中启动 OpenAI-compatible vLLM 服务，发送普通和流式请求，读取核心指标。
3. 从 HTTP 入口追踪到 input processing、EngineCore、Scheduler、KV cache、ModelRunner、Attention、Sampler 和输出流。
4. 对同一 workload 设置基线，只改一个变量，解释结果并给出回滚条件。
5. 根据模型、GPU、SLO 和流量选择量化、并行、cache、batching、调度和投机解码策略。
6. 设计包含路由、扩缩容、可观测、隔离、升级回滚和故障处理的生产服务。
7. 在面试中用 30 秒摘要、3 分钟源码证据和工程取舍三个层次回答问题。

## 3. 课程信息架构

保留现有 `01` 至 `09` 主题目录和相对链接，在 README 增加四条可并行使用的学习路径。

### 3.1 入门线：从零到首个可观测服务

1. token、attention、KV cache、prefill/decode 和延迟指标。
2. 选择无 GPU 或 NVIDIA GPU 实验路径。
3. 安装或连接服务，启动 OpenAI-compatible endpoint。
4. 发送普通、streaming 和并发请求。
5. 在 `/metrics` 观察请求数、队列、KV 使用率、TTFT 和 token 速率。
6. 用一页实验报告记录环境、命令、结果和故障。

### 3.2 源码线：从一个请求到一个 token

主链路固定为：

```text
HTTP / AsyncLLM
  -> input validation / tokenization / multimodal preprocessing
  -> EngineCore client and IPC
  -> Scheduler and request state
  -> KV cache manager and block pool
  -> GPU model runner and input batch
  -> attention backend / model forward
  -> sampler / structured output
  -> detokenization / streaming response
```

每个节点都必须提供数据契约、关键类/函数、一步时序和可观测证据。新工作以 V1 为准，V0 只用于理解兼容和迁移，不再作为主实现。

### 3.3 工业实战线：从能跑到可上线

1. 定义 workload 和 SLO，区分稳态吞吐、开环到达和闭环并发。
2. 建立基线，保存可复现配置和原始结果。
3. 调整 batching、KV cache、prefix cache、chunked prefill、compile/CUDA Graph、量化和 speculative decoding。
4. 选择 TP、PP、DP、EP、CP 和 prefill/decode 分离方案。
5. 建立路由、扩缩容、SLO、指标、trace、日志和告警。
6. 执行 OOM、KV 抢占、NCCL hang、重试风暴、cache 命中下降和冷启动演练。
7. 执行 canary、drain、升级、回滚和容量再校准。

### 3.4 面试线：从理解到可辩护的工程表达

每个核心主题建立三层答案：

- 30 秒：问题、结论和一个关键原因。
- 3 分钟：数据流、算法、源码入口和主要取舍。
- 工程追问：如何验证、何时失效、如何监控、如何回滚。

面试线包含显存/容量/吞吐计算、源码追踪、系统设计、故障定位和模拟面试评分。

## 4. 章节标准

每章在 `curriculum.toml` 中登记，并在正文中保持读者可见的统一结构：

1. 谁该读、前置阅读、预计时间、难度和实验环境。
2. 学完能力，使用可观察动词，不用“了解”作为唯一结果。
3. 心智模型和必要的数学/系统背景。
4. 当前主线数据契约与源码路径，使用语义引用。
5. 最小实验、trace 练习或可复现算例。
6. 工业选型：何时使用、何时不用、成本和回滚条件。
7. 常见失败、观测证据和排查顺序。
8. 小结、自检、面试表达和下一步。

每个命令块必须说明执行目录、前提、成功证据和清理/回滚方式。对环境敏感的值使用变量，不把某张 GPU 上的偶然性能数字写成普遍保证。

## 5. 现有内容复核

### 5.1 复核范围

所有现有章节都必须有一条 review ledger 记录，不得只处理含行号的文件。每章复核：

- 路径、类、函数、配置项和 CLI 参数是否存在；
- 请求状态、调度顺序、cache 行为、默认值和回退路径是否一致；
- metrics 的注册名、Prometheus 暴露名和语义是否一致；
- Mermaid 数据流和文字是否与当前 V1 一致；
- 安装、启动、请求、基准和部署命令是否使用当前接口；
- 工程建议是否清楚区分源码事实、默认策略、经验法则和特定环境结果。

### 5.2 Review ledger

`content-review.toml` 为每章记录：

- `path`；
- `reviewed_commit`；
- `reviewed_at`；
- `source_contracts`；
- `commands_checked`；
- `metrics_checked`；
- `diagrams_checked`；
- `hardware_verified`；
- `notes`。

`reviewed_commit` 必须与 `source.lock.json` 一致。账本使用以下确定类型：

```toml
[[review]]
path = "03-code-walkthrough/02-scheduler.md"
reviewed_commit = "4c6e2e4b308c15fc2bcdf10e278f2591c9cec0dc"
reviewed_at = "2026-07-20T15:00:00Z"
source_contracts = true
commands_checked = true
metrics_checked = true
diagrams_checked = true
hardware_verified = []
notes = []
```

`hardware_verified` 是硬件运行记录 ID 的字符串数组；每个 ID 在硬件报告索引中对应 commit、GPU、驱动、CUDA、模型、命令、原始产物和日期。空数组明确表示该章没有当前 commit 的硬件验证，不影响静态复核状态，也不允许页面显示 GPU 实测 badge。

## 6. 新增与重点扩展章节

新增 10 个章节，填补当前从请求入口到生产交付之间的缺口。完成后库存为 60 个章节 Markdown，README 和构建导航必须由库存校验得出同样数字。

### 6.1 源码请求闭环

1. `03-code-walkthrough/08-input-processing-and-tokenization.md`
   - OpenAI schema 到 `EngineCoreRequest` 之前的校验、tokenization、prompt adapter、LoRA 和多模态预处理。
   - 区分前端 CPU 成本、异步阻塞和模型前向成本。
2. `03-code-walkthrough/09-output-processing-and-streaming.md`
   - sampler 结果、detokenization、stop condition、usage accounting、streaming chunk 和客户端取消。
   - 说明首 token 、后续 token 和网络 backpressure 如何影响用户延迟。

### 6.2 可运行实操闭环

3. `07-hands-on/05-serve-openai-api.md`
   - 受支持的无 GPU/CPU 环境、远程 endpoint 路径和 NVIDIA GPU 路径。
   - 启动服务，完成 health、models、chat/completions、streaming、并发请求和 metrics 检查。
4. `07-hands-on/06-benchmark-methodology.md`
   - 开环/闭环、warmup、输入/输出长度分布、并发度、request rate、百分位数和 goodput。
   - 记录命令、环境、配置、原始数据和结论的实验模板。
5. `07-hands-on/07-tuning-playbook.md`
   - 基于“现象→黄金指标→资源瓶颈→单变量实验→回滚”调整常用配置。
   - 对照小 prompt/大 decode、长 prompt/短 decode、高 prefix 复用、多 LoRA 和严格 TTFT SLO。
6. `07-hands-on/08-production-capstone.md`
   - 端到端交付：需求、workload、架构、部署、基准、调优、SLO、监控、故障演练、回滚、容量计划和复盘。
   - 输出可转化为项目经历的证据包。

### 6.3 生产安全与生命周期

7. `08-production-deployment/11-security-and-multi-tenancy.md`
   - 认证、TLS 终止、请求/输出限制、租户配额、资源隔离、日志脱敏、模型与 LoRA 信任边界。
   - 区分 vLLM 自身能力和网关/Kubernetes/密钥系统责任。
8. `08-production-deployment/12-upgrades-rollbacks-and-compatibility.md`
   - vLLM、模型、量化格式、GPU/CUDA/驱动、API 客户端和配置的兼容矩阵。
   - canary、影子流量、质量回归、性能回归、drain、回滚和变更记录。

### 6.4 面试计算与模拟

9. `06-interview/03-capacity-and-troubleshooting-drills.md`
   - KV 显存、权重显存、batch token、并行度、吞吐和容量估算题。
   - TTFT、TPOT、OOM、preemption、NCCL、cache miss、CPU bottleneck 和重试风暴的证据化排查题。
10. `06-interview/04-mock-interview-and-rubric.md`
    - 概念、源码、计算、系统设计和事故处理五轮模拟。
    - 准确性、证据、取舍、验证和表达五维评分表。

## 7. 双轨实验设计

### 7.1 无 GPU 轨

“无 GPU”不默认等于“vLLM 可在任意笔记本本地运行”。它包含三种明确场景：

1. 纯本地：源码导航、数学计算、配置比较、影响报告和离线 metrics/trace 分析。
2. 连接远程 vLLM endpoint：完成 API、streaming、并发、基准客户端和可观测练习。
3. 官方明确支持的 CPU 平台：只按当前源码与官方安装文档列出要求，不把 macOS 或任意 x86 环境泛化为可支持。

### 7.2 NVIDIA GPU 轨

- 单 GPU 必做：服务、基准、batching/KV 调优、prefix cache、量化或 compile 至少一项对照。
- 多 GPU 选做：TP/PP/DP 中至少一种，并观察通信与利用率。
- 高阶选做：EP、CP、disaggregated prefill/decode、speculative decoding 或多 LoRA。

每个 GPU 实验必须保存：

- vLLM commit 和安装方式；
- GPU、驱动、CUDA、框架和模型版本；
- 启动命令和完整有效配置；
- workload 和 warmup；
- 原始结果而非只保留结论；
- 改动的唯一变量、结果解释和回滚条件。

## 8. 工业终章项目

### 8.1 输入

- 一个明确的模型和硬件边界；
- 输入/输出 token 分布、峰值/平均请求率和并发度；
- TTFT、TPOT、错误率和 goodput SLO；
- 多租户、质量、成本和变更窗口约束。

### 8.2 必交付物

1. 一页需求与假设。
2. 带请求、路由、推理、metrics 和依赖边界的架构图。
3. 可复现的部署配置和健康检查。
4. 基线报告和至少两次单变量调优实验。
5. 容量计算，包含峰值余量、故障余量和冷启动。
6. SLO dashboard 和至少三条可操作告警。
7. 至少两个故障演练记录，包含检测、缓解、恢复和预防。
8. 升级、drain、canary 和回滚方案。
9. 一份事故复盘或性能复盘。
10. 5 分钟项目陈述、三个技术难点和三个面试追问答案。

### 8.3 验收

终章项目不按某个绝对 tokens/s 评分，而按可复现性、指标完整性、瓶颈判断、取舍、故障处理和回滚能力评分。

## 9. 面试内容规范

每道问题包含：

1. 题目与考察目标。
2. 30 秒回答。
3. 3 分钟展开，包含当前源码证据。
4. 至少两个工程追问。
5. 常见错误回答，说明错在概念、数据、默认假设还是忽略取舍。
6. 评分键：准确性、源码/指标证据、工程取舍、验证与回滚、表达结构。

计算题必须显示公式、单位、中间量、假设和 sanity check。系统设计题必须先问 workload 与 SLO，不允许直接背一张固定架构图。

## 10. 证据与时效性规则

- vLLM 实现事实优先引用锁定 commit 中的语义源码链接。
- 用户接口、安装、平台支持和部署事实使用同一 commit 对应的官方文档，并记录访问日期。
- 论文结论引用原论文，不把早期 vLLM 论文的上限性提升数字直接当成当前版本或用户硬件的保证。
- 工程经验必须显式标为经验法则，并给出用指标验证或否定它的方法。
- GPU 数据必须附带环境和原始输出；无证据时使用“预期现象”而不是“实测结果”。

## 11. 构建与验证

### 11.1 静态门禁

- 章节库存、README 索引、`curriculum.toml` 和 `content-review.toml` 一致。
- 所有章节都有环境、难度、前置、学习成果、源码区域和最新复核记录。
- 所有 vLLM 行号引用符合源码同步契约。
- 内部链接、图片、Mermaid 代码块、章节导航和 README 统计可验证。
- 命令块有执行目录、前提、成功证据和清理步骤，不包含未定义占位符。

### 11.2 构建门禁

- HTML 在干净输出目录生成，章节数、标题、搜索索引和导航与库存一致。
- PDF/EPUB 构建脚本的章节顺序与 `curriculum.toml` 一致；完整 PDF/EPUB 产物只在具备 pandoc/xelatex 的环境作为发布门禁。
- 构建输出不作为源文件手工编辑。

### 11.3 内容复核门禁

每章只有在下列条件都有证据时才完成：

1. review ledger 指向当前锁定 SHA。
2. 本章所有语义锚点解析成功。
3. 本章 `source_areas` 在基线到候选的 diff 已复核。
4. 受影响的行为、默认值、指标和命令已修正。
5. 章末自检和面试回答与修正后正文一致。

## 12. 实施顺序

1. 先完成源码同步工具、版本锁、章节库存和复核账本。
2. 将 vLLM 更新到当时官方 `main`，生成从旧基线到候选的影响报告。
3. 按请求主链复核 `01`、`02`、`03`、`09`，建立新手和源码认知基础。
4. 复核 `04`、`05`，更新性能与分布式实现。
5. 复核 `07`、`08`，完成实验、部署、可观测和故障闭环。
6. 复核并扩展 `06`，使所有面试答案可回指当前源码和实战证据。
7. 新增 10 章，更新 README 学习路径、索引、章节数和预计时长。
8. 在内容修订期间定期查询上游。最终验证前再对齐一次官方 `main`；如果 SHA 前进，只重新复核影响报告命中的章节。
9. 运行全部静态、单元、链接和构建门禁，记录最终验证时间。

## 13. 完成标准

1. 现有 50 章全部有当前锁定 commit 的 review ledger，没有未复核章节。
2. 所有过时路径、符号、配置、默认行为、指标和命令已按锁定源码更新。
3. 所有行号引用迁移为语义引用，并能重新生成当前行号。
4. 新增 10 章全部符合章节标准，不是纲要或未完成占位页。
5. 入门路径能从前置概念到首个可观测服务；源码路径覆盖完整请求闭环。
6. 工业实战路径包含基准、调优、容量、安全、升级、观测、故障和终章项目。
7. 面试路径包含三层回答、计算题、源码题、系统设计、排障题和可重复使用的评分表。
8. README 章节数、学习路径、预计时长和章节索引与库存一致。
9. 源码同步门禁、内容门禁、内部链接和 HTML 构建全部通过。
10. 最终报告明确记录已验证 SHA、时间、未执行的 GPU 实测和任何外部环境限制。
