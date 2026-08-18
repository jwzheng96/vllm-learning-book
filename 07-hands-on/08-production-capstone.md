# 08. 生产 Capstone：交付一个可运维、可扩容、可回滚的 vLLM 服务

> **谁该读这一篇？** 想把零散知识变成作品集项目、生产演练或系统设计面试案例的读者。
>
> **前置阅读：** 本目录 01–07；生产部署目录可与本章并行查阅。
>
> **耗时：** 无 GPU 的设计/证据演练约半天；受控硬件实测通常 2–5 天。
>
> **学完能：** 从需求、架构和基线走到容量、SLO、事故、升级回滚，并用五分钟讲清决策证据。

> **当前源码复核：** 项目基线锁定 `b23bd73f540175f9e117eaee5029cd7d8df63964`。静态完成不等于硬件验证；所有实测声明必须附 immutable run ID 与 artifact hash。

---

## 1. 场景

你负责一个内部 RAG/chat 服务。业务方给出峰值与延迟目标，但输入长度有长尾，模型可能需要量化，平台还要求多租户隔离、升级可回滚和事故审计。

你要交付的不是“启动成功”，而是一个可审查的服务包：

```text
requirements → architecture → deployment → benchmark/tuning
      → capacity → SLO/alerts → incident drills → upgrade/rollback
      → retrospective → interview narrative
```

## 2. 完成路线

| 路线 | 环境 | 可完成内容 | 不得声称 |
| --- | --- | --- | --- |
| A. NVIDIA 实测 | 受支持 Linux GPU 测试环境 | 全部功能、性能、容量、故障演练 | 未覆盖的生产规模/硬件 |
| B. 受支持 CPU | 当前支持 CPU 平台 | 功能、可观测、流程与小规模基线 | GPU kernel/显存/吞吐结论 |
| C. 远端 endpoint | 授权测试服务 | 客户端、SLO、benchmark、错误与部分事故 | server 内部配置/硬件归因 |
| D. 无运行环境 | 文档 + 示例数据 | 设计、计算、查询、runbook tabletop | 任何当前 SHA 的运行验证 |

报告首页写路线和限制。路线 D 是合格的设计作品，但 rubric 中 hardware evidence 项得 0，不用伪造截图。

## 3. 项目目录

```text
capstone/
├── 01-requirements.md
├── 02-architecture.md
├── 03-deployment.md
├── 04-experiments/
│   ├── baseline.md
│   ├── experiment-a.md
│   └── experiment-b.md
├── 05-capacity-plan.md
├── 06-slo-dashboard-alerts.md
├── 07-incidents/
│   ├── kv-pressure.md
│   └── dependency-failure.md
├── 08-upgrade-rollback.md
├── 09-retrospective.md
├── 10-interview-narrative.md
└── artifacts/
    └── manifest.json
```

每份结论通过相对链接指向 artifact；artifact manifest 保存 size、SHA-256、生成命令、UTC 与敏感信息处理说明。

## 4. 交付物 1：需求与 workload contract

必须量化：

- API/task、模型和 immutable revision、质量 gate；
- steady/peak/burst request rate 与并发；
- input/output token p10/p50/p90/p99/max；
- streaming、logprobs、structured output、LoRA、多模态比例；
- TTFT/TPOT/E2E/availability/error SLO 与统计窗口；
- tenant、数据分级、保留/删除、地域与审计要求；
- 成本/功耗/容量边界；
- RTO/RPO、升级窗口与 rollback deadline。

每项注明来源：业务数据、合成假设、平台限制或实验。假设要有验证期限和 owner。

### 验收

- 单位完整，percentile 与窗口明确；
- offered load 与 admitted load 分开；
- model quality 和 safety 不是“以后再测”；
- 没有用平均长度代替分布。

## 5. 交付物 2：架构与责任边界

至少画出：client → gateway/auth/rate limit → router → vLLM replicas → model/cache/storage → metrics/log/trace。

在图旁做责任表：

| 控制 | vLLM | gateway/router | orchestrator/platform | secret/observability |
| --- | --- | --- | --- | --- |
| OpenAI-compatible model serving | owner | route | schedule | observe |
| external auth/TLS/quota | API key 可做基础保护 | owner | network policy | secret rotation |
| replica lifecycle/drain | health + process | stop routing | owner | alert |
| model artifact provenance | load/verify compatibility | — | image/model policy | digest/audit |
| tenant logs/redaction | avoid sensitive payload | header/filter | storage policy | owner |

说明单点、failure domain、backpressure、retry budget、幂等性和跨 tenant cache 策略。

## 6. 交付物 3：可复现部署

记录 immutable 组合：source SHA、image digest、model/tokenizer revision、driver/CUDA/PyTorch、backend、完整 resolved config。

部署步骤必须包含：

1. preflight：资源、端口、模型权限、磁盘、secret；
2. 启动：最小命令/manifest，secret 不入仓库；
3. readiness：health、models、golden request；
4. observability：metrics/log/trace；
5. drain/TERM：只操作目标 workload/PID；
6. rollback：上一 image/config/model revision；
7. cleanup：精确 namespace/资源名和保留 artifact。

路线 C/D 用 deployment manifest + dry review，不声称执行。

## 7. 交付物 4：基线 + 两个单变量实验

从 [`templates/experiment-report.md`](templates/experiment-report.md) 复制三份。

基线回答“当前方案在 workload/SLO 下的曲线”。两个实验从下列选择，且各只改一个变量：

- batch token budget；
- max concurrency / admission；
- prefix caching；
- KV dtype；
- weight quantization；
- TP/DP 布局；
- compile/eager；
- speculative decoding。

每份必须同时报告 TTFT/TPOT/ITL/E2E、throughput/goodput/error、实际 token 分布、KV/preemption、成本代理和质量 gate。结论允许 rejected 或 inconclusive；“没有收益但证据完整”优于挑最好看的轮次。

## 8. 交付物 5：容量计划

从 [`templates/capacity-plan.md`](templates/capacity-plan.md) 复制。

核心计算：

```text
peak admitted rps = peak offered rps × (1 - rejected_by_policy_fraction)
required replicas = ceil(peak admitted rps / per-replica goodput_at_SLO)
planned replicas = ceil(required replicas × (1 + headroom_fraction))
token demand/s = input token rate + output token rate
```

同时验证内存/KV：

```text
KV bytes/token/layer ≈ 2 × num_kv_heads × head_size × dtype_bytes
request KV ≈ tokens_in_active_context × layers × KV bytes/token/layer
```

这是估算，模型的 hybrid/sliding-window/MLA/cache spec 会改变结果。最终用当前启动日志/配置和压力实验校准。

容量计划至少有 base/peak/degraded 三种场景、N+1 或故障域余量、冷启动/HPA lead time、模型下载带宽和成本。

## 9. 交付物 6：SLO dashboard 与 alerts

Dashboard 至少包含同一时间轴的：

- request rate、success/error/abort；
- TTFT/TPOT/ITL/E2E p50/p90/p99；
- running/waiting；
- KV usage、preemption rate、prefix hit/query rate；
- prompt/generation token throughput；
- replica health/restart/ready time；
- GPU/CPU/memory/network/NCCL；
- deploy/config/model revision annotation。

Alert 必须有：condition、for window、severity、impact、first checks、safe mitigation、escalation、rollback link。避免只用 `GPU util > 90%`；高利用率可能是健康饱和。

路线 D 至少写可语法审查的 PromQL 与预期标签，注明未连接真实 Prometheus。

## 10. 交付物 7：两个事故演练

从 [`templates/incident-review.md`](templates/incident-review.md) 复制两份。

### Incident A：KV pressure / preemption

注入方式必须有界：在隔离测试环境逐档提高长请求并发或降低受控实例 KV 预算。观察 KV、preemption、queue、TTFT/E2E、recompute 证据。停止条件是 OOM 前的预设阈值、错误开始增长或影响逃逸出 namespace。

### Incident B：依赖/worker/网络失败

选择一个可恢复且获授权的故障：测试实例 TERM、模型存储不可达模拟、connector/下游超时或网络策略阻断。不要在共享节点做 `kill -9`/宽泛 firewall 变更。

两次都要覆盖 detect → triage → mitigate → recover → verify → follow-up，并记录实际/桌面演练。

## 11. 交付物 8：升级与回滚演练

建立兼容矩阵：

| 维度 | old | candidate | gate |
| --- | --- | --- | --- |
| vLLM source/image | immutable SHA/digest | immutable SHA/digest | source impact review |
| model/tokenizer/template | revision/hash | revision/hash | golden quality |
| driver/CUDA/PyTorch/backend | versions | versions | startup + kernel |
| API/config | schema snapshot | schema snapshot | contract/client tests |
| performance | baseline artifact | candidate artifact | goodput/SLO/cost |

流程：shadow（可选）→ canary → 小流量 → 扩流 → 观察完整周期。每步有自动 gate。rollback 恢复 old image/config/model revision、摘候选、drain、golden request、指标回基线。

同时说明 cache 不兼容/冷 cache 的影响，避免回滚后因 warmup 差异误报二次事故。

## 12. 交付物 9：复盘

回答：

1. 哪个初始假设被证伪？
2. 哪项证据最改变设计？
3. 哪个数字仍是估算，如何验证？
4. 哪个安全/可靠性控制不属于 vLLM？
5. 当前最大单点和下一项投资是什么？
6. 若流量/上下文/模型翻倍，先失守哪里？
7. 文档如何随上游 main 自动发现影响？

把未完成项写成 owner + due date + acceptance evidence，不写模糊“后续优化”。

## 13. 交付物 10：五分钟面试叙事

建议时间：

| 时间 | 内容 |
| --- | --- |
| 0:00–0:40 | 业务、workload、SLO、安全边界 |
| 0:40–1:30 | 架构与为何这样切责任 |
| 1:30–2:40 | 基线、两个实验、被证伪的假设 |
| 2:40–3:30 | 容量公式、headroom、成本 |
| 3:30–4:20 | 事故证据与恢复 |
| 4:20–5:00 | upgrade/rollback、限制与下一步 |

每个结论用“现象 → 证据 → 机制 → 取舍 → 决策”表达。面试官追问数字时，能指出 artifact 与适用范围；没有跑过的部分直接说是设计/估算。

## 14. 总体验收 Rubric（100 分）

| 维度 | 分值 | 满分证据 |
| --- | ---: | --- |
| 需求/workload | 10 | 分布、SLO、质量、安全、成本都有来源 |
| 架构/责任边界 | 10 | 图、故障域、backpressure、control owner |
| 可复现部署 | 10 | immutable manifest、readiness、drain、cleanup |
| 实验方法 | 15 | 基线 + 2 单变量、重复、原始 artifact、有效比较 |
| 容量 | 10 | goodput@SLO、headroom、故障与成本场景 |
| 可观测/SLO | 10 | 当前指标、PromQL、alert/runbook/revision annotation |
| 事故演练 | 10 | 两类事故、stop condition、恢复验证 |
| 安全/隐私 | 10 | secret、auth/TLS、tenant、redaction、audit |
| 升级/回滚 | 10 | compatibility/golden/canary/drain/rollback |
| 表达/诚实边界 | 5 | 五分钟清楚，实测/估算/未验证分开 |

建议通过线 80；没有当前 SHA 硬件证据时仍可完成路线 C/D，但“实验方法/容量”相应项只能按静态设计证据得分。

## 自检

1. 为什么启动成功只覆盖交付物的一小部分？
2. 哪些控制应由 gateway/platform 而非 vLLM 承担？
3. 容量为什么用 goodput@SLO 而不是最大 throughput？
4. 事故演练如何设置 blast radius 与 stop condition？
5. 面试中如何表述未做 GPU 实测的项目？

### 参考答案

1. 启动成功只证明进程能加载并监听端口，还没有证明正确性、streaming、容量、SLO、指标、故障恢复、安全、升级回滚和证据包完整。交付应覆盖“能跑、可测、可观测、可恢复、可解释”五层。
2. Gateway/platform 更适合承担鉴权、租户 quota、路由、限流、重试、熔断、TLS、审计、模型制品和扩缩容；vLLM 负责模型执行、调度、采样和引擎级 metrics。边界清晰才能避免把控制面逻辑塞进 worker。
3. 最大 throughput 可能在请求违反 SLO、错误上升或 retry storm 时取得。goodput@SLO 只统计满足时延、错误和质量门的有效请求，更接近真实业务容量和成本。
4. 先定义隔离的 canary、最大影响范围、观察窗口、自动/人工 stop condition 和回滚命令；故障注入只针对一个 failure domain，并保留正常池。恢复后要跑 golden、指标回基线和资源清理验证。
5. 明确区分静态源码复核、命令审查、模拟/无 GPU 验证和真实硬件实测；给出可复现实验命令、预期指标、硬件缺口和上线前 gate。不能用估算或文档阅读伪装成 GPU benchmark。

## 下一步

进入 [`../08-production-deployment/01-deployment-architectures.md`](../08-production-deployment/01-deployment-architectures.md)，把 capstone 的架构、SLO、可靠性和生命周期设计进一步生产化。
