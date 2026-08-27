# vLLM 学习手册

<!-- vllm-version:start -->
- Validated vLLM: `b23bd73f540175f9e117eaee5029cd7d8df63964`
- Upstream committed: `2026-07-20T15:32:54+00:00`
- Validated: `2026-08-25T07:44:40Z`
- Latest candidate: `5e379a361e3ea8bb82b7efd768c36f39a0cf32fd`
- Candidate lag: `1463` commits
<!-- vllm-version:end -->

[![Pages](https://github.com/jwzheng96/vllm-learning-book/actions/workflows/pages.yml/badge.svg)](https://github.com/jwzheng96/vllm-learning-book/actions/workflows/pages.yml)
[![Upstream sync](https://github.com/jwzheng96/vllm-learning-book/actions/workflows/sync-upstream.yml/badge.svg)](https://github.com/jwzheng96/vllm-learning-book/actions/workflows/sync-upstream.yml)
[![Site](https://img.shields.io/badge/site-jwzheng96.github.io%2Fvllm--learning--book-8b1538)](https://jwzheng96.github.io/vllm-learning-book/)
[![vLLM](https://img.shields.io/badge/vllm-b23bd73_(2026--07--20)-1a4d80)](https://github.com/vllm-project/vllm/tree/b23bd73f540175f9e117eaee5029cd7d8df63964)

> 一份写给大模型推理工程入门者的源码教程。
> **64 章 · 30K+ 行**，从 PagedAttention 论文到 384 卡 H100 / 昇腾 910B 生产部署、端到端 profiling 与 Mooncake 分布式 KV 存储，覆盖整条链路。
> 每章都用可刷新语义锚点对照锁定 commit 的 vLLM 源码，可以“读笔记 ↔ 跳源码”无缝切换。
>
> 📖 在线阅读：**[jwzheng96.github.io/vllm-learning-book](https://jwzheng96.github.io/vllm-learning-book/)**

---

## 这份手册解决什么问题

如果你正在做下面这些事，这是为你写的：

- **系统补课**：想把 vLLM 的核心机制啃透，不再只记"PagedAttention 解决了什么"这种结论。
- **业务接入**：要上线 LLM 推理服务，要选 v0/v1、调度策略、量化方案、部署架构。
- **性能优化**：TTFT/TPOT 不达标，需要从架构层定位到内核层逐级排查。
- **底层贡献**：想给 vLLM 提 PR，先得知道 scheduler / kv manager / attention backend 怎么咬合。

它**不**适合：完全没接触过 LLM 推理 → 先看 [`01-overview/00-prerequisites.md`](01-overview/00-prerequisites.md) 把前置概念铺平；纯 prompt engineer 不碰服务侧 → 这本太工程。

---

## 怎么用这份资料

**两种打开方式：**

| 方式 | 入口 | 适合 |
| --- | --- | --- |
| **Markdown 直接读** | 本 README.md → 按章节文件名跳转 | IDE 内阅读、对照源码、想在 GitHub 上读 |
| **HTML 在线版** | `python3 build_html.py` 后开 `_site/index.html` | 想要侧栏 + 全文搜索 + Mermaid 渲染 + 暗色主题 + 阅读时间提示 |

所有跨章链接、内嵌 Mermaid、代码块都在两种模式下都能用。HTML 版额外有 lunr.js 全文搜索和阅读时间估算。

源码版本、语义锚点、影响报告和人工复核的完整流程见 [`docs/source-sync.md`](docs/source-sync.md)。这里的“已验证”是 fail-closed 语义门禁：锁定 SHA 与子模块一致、64 章 inventory 完整、源码锚点可解析、没有 unmanaged source line，而且 `content-review.toml` 的每一章都在该 SHA 上完成 source / command / metric / diagram review。PR 与 Pages 使用 `validate --profile full --require-committed`；每周工作流只创建候选 PR，不会自动合并或发布。

文中任何硬件验证徽章都必须对应可索引、可复现的运行记录（锁定 commit、硬件、命令和结果）；只有静态源码复核时，不得标注为 GPU 已验证。手动 GPU workflow 见 `.github/workflows/gpu-validation.yml`：它只在预置 vLLM / driver 的 self-hosted NVIDIA runner 上运行 `scripts/gpu-validation.sh`，并归档脱敏证据；普通 CI 不安装或假装拥有 GPU toolchain。

**每章统一的结构：**

> **谁该读这一篇？** ...
> **前置阅读：** ...
> **耗时：** N 分钟
> **学完能：** ...

正文（含 mermaid / 表格 / 代码引用）

```
## 小结
## 自检（3-5 题，自答）
## 下一步（跳转推荐）
```

按这个节奏走，刷完整本约 25-35 小时（不含动手实验）。

---

## 学习路径

```mermaid
flowchart TB
    Start[选择你的目标] --> Q[30 分钟理解]
    Start --> S[源码主线]
    Start --> P[工业实战]
    Start --> I[面试冲刺]

    Q --> Q1[前置概念 → vLLM 是什么 → 架构 → 首个 API 服务]
    S --> S1[入口 → 输入 → 调度 / KV → Runner / Attention → Sampling → 输出]
    P --> P1[环境 → 基准 → 调优 → 部署 / SLO → 安全 / 升级 → Capstone]
    I --> I1[高频题 → 计算题 → 系统设计 → 排障 → 模拟面试]

    classDef phase fill:#eff5ff,stroke:#2563eb,color:#1a1f29;
    classDef topic fill:#f7f8fa,stroke:#5b6573,color:#1a1f29;
    class Q,S,P,I phase;
    class Start,Q1,S1,P1,I1 topic;
```

四条路径可以独立走，也可以从“30 分钟理解”起步后再分流。

### 30 分钟理解

适合第一次接触 vLLM、需要快速建立全局心智模型的人。

[`前置知识`](01-overview/00-prerequisites.md)（按需跳读） → [`vLLM 是什么`](01-overview/01-what-is-vllm.md) → [`整体架构`](01-overview/02-architecture.md) → [`启动 OpenAI-Compatible API`](07-hands-on/05-serve-openai-api.md)

### 源码主线

适合准备读代码、改代码或定位引擎问题的人。顺序刻意沿一次请求的数据流展开。

[`入口与主循环`](03-code-walkthrough/01-entry-points.md) → [`输入与 Tokenization`](03-code-walkthrough/08-input-processing-and-tokenization.md) → [`Scheduler`](03-code-walkthrough/02-scheduler.md) → [`KV Cache`](03-code-walkthrough/03-kv-cache-manager.md) → [`Model Runner`](03-code-walkthrough/04-model-runner.md) → [`Attention`](03-code-walkthrough/05-attention-backends.md) → [`Sampling`](09-advanced-features/01-sampling-and-logits.md) → [`输出与 Streaming`](03-code-walkthrough/09-output-processing-and-streaming.md)

### 工业实战

适合要把服务从“能跑”推进到“可量化、可调优、可上线、可回滚”的工程团队。

[`环境搭建`](07-hands-on/01-setup.md) → [`Benchmark 方法论`](07-hands-on/06-benchmark-methodology.md) → [`调优 Playbook`](07-hands-on/07-tuning-playbook.md) → [`部署架构`](08-production-deployment/01-deployment-architectures.md) → [`容量规划`](08-production-deployment/04-autoscaling-and-capacity.md) → [`SLO 与可观测性`](08-production-deployment/05-slo-and-observability.md) → [`H100/910B 大规模部署`](08-production-deployment/13-384-h100-glm-deepseek-deployment.md) → [`端到端 Profiling`](08-production-deployment/15-end-to-end-latency-profiling-and-optimization.md) → [`Mooncake 分布式 KV 存储`](08-production-deployment/16-mooncake-distributed-inference-storage.md) → [`生产 Capstone`](07-hands-on/08-production-capstone.md)

### 面试冲刺

适合用可计算、可追问、可评分的方式准备推理工程面试。

[`30 道三层回答`](06-interview/01-common-questions.md) → [`容量与故障练习`](06-interview/03-capacity-and-troubleshooting-drills.md) → [`系统设计`](06-interview/02-system-design.md) → [`五轮模拟面试`](06-interview/04-mock-interview-and-rubric.md)

如果目标是完整掌握，仍建议按 01 → 09 顺序阅读，并在每章用锁定 commit 的语义源码锚点回到真实实现。

---

## 章节索引（带钩子）

每章后面一句话告诉你为什么要读。

### 1. 总览 · `01-overview/` — 6 章

- [`00-prerequisites.md`](01-overview/00-prerequisites.md) — 🆕 token / KV cache / TTFT·TPOT / TP·PP·DP 一次铺平，零基础起点。
- [`01-what-is-vllm.md`](01-overview/01-what-is-vllm.md) — vLLM 是什么，为什么快，靠哪三大武器。
- [`02-architecture.md`](01-overview/02-architecture.md) — 三层进程、四个核心数据结构、一步推理的完整数据流。
- [`03-v0-vs-v1.md`](01-overview/03-v0-vs-v1.md) — V1 重构改了什么，为什么改。
- [`04-project-structure.md`](01-overview/04-project-structure.md) — 1700+ 文件按模块分类导航（这章很长，按需查阅）。
- [`05-process-and-ipc-internals.md`](01-overview/05-process-and-ipc-internals.md) — 从进程/地址空间讲到 ZMQ socket pattern、共享内存状态机与 384 卡分布式通信边界。

### 2. 核心概念 · `02-core-concepts/` — 5 章

- [`01-paged-attention.md`](02-core-concepts/01-paged-attention.md) — 用 OS 虚拟内存类比讲 paged KV cache + block table。
- [`02-continuous-batching.md`](02-core-concepts/02-continuous-batching.md) — 为什么 vLLM 不用 static batching，迭代级调度的两个前提。
- [`03-kv-cache-management.md`](02-core-concepts/03-kv-cache-management.md) — Block Pool / 引用计数 / preempt vs swap 策略。
- [`04-prefix-caching.md`](02-core-concepts/04-prefix-caching.md) — Merkle 链式 hash + extra_keys，多卡 / LoRA / 多模态怎么算。
- [`05-chunked-prefill.md`](02-core-concepts/05-chunked-prefill.md) — 长 prompt 怎么切才不卡 decode，`max-num-batched-tokens` 怎么调。

### 3. 源码走读 · `03-code-walkthrough/` — 10 章

- [`01-entry-points.md`](03-code-walkthrough/01-entry-points.md) — `LLM` / `AsyncLLM` / `EngineCore` 的调用链。
- [`02-scheduler.md`](03-code-walkthrough/02-scheduler.md) — `Scheduler.schedule()` 完整走读，token budget + preemption。
- [`02b-scheduling-policies.md`](03-code-walkthrough/02b-scheduling-policies.md) — FCFS vs PRIORITY、优先级反演、抢占代价量化。
- [`03-kv-cache-manager.md`](03-code-walkthrough/03-kv-cache-manager.md) — `allocate` / `free` / `hash` 的代码级细节。
- [`04-model-runner.md`](03-code-walkthrough/04-model-runner.md) — `execute_model`：输入拼装 / forward / sampler。
- [`05-attention-backends.md`](03-code-walkthrough/05-attention-backends.md) — FlashAttn / FlashInfer / Triton / MLA 怎么选。
- [`06-cuda-kernels.md`](03-code-walkthrough/06-cuda-kernels.md) — PagedAttention v1/v2、RoPE、RMSNorm CUDA 实现。
- [`07-model-architectures.md`](03-code-walkthrough/07-model-architectures.md) — MLA / Mamba / MoE / GQA 在源码层的差异。
- [`08-input-processing-and-tokenization.md`](03-code-walkthrough/08-input-processing-and-tokenization.md) — 从 OpenAI 请求、chat template、tokenizer 到 EngineCoreRequest。
- [`09-output-processing-and-streaming.md`](03-code-walkthrough/09-output-processing-and-streaming.md) — 从 sampler / detokenizer 到 SSE、finish reason 与取消传播。

### 4. 优化 · `04-optimizations/` — 5 章

- [`01-quantization.md`](04-optimizations/01-quantization.md) — FP8 / INT4 / AWQ / GPTQ / Marlin 的选型矩阵。
- [`02-speculative-decoding.md`](04-optimizations/02-speculative-decoding.md) — EAGLE3 / MTP / PARD / DFlash / DSpark、动态 K 与自适应验证的源码导读。
- [`03-cudagraph-and-compile.md`](04-optimizations/03-cudagraph-and-compile.md) — CUDA Graph + torch.compile 何时开 / 何时关 / 失败降级路径。
- [`04-compilation-internals.md`](04-optimizations/04-compilation-internals.md) — CompilerManager / VllmBackend / 自定义 pass 深度。
- [`05-roofline-and-arithmetic-intensity.md`](04-optimizations/05-roofline-and-arithmetic-intensity.md) — 🆕 存算比定量推导：prefill 为何 compute-bound、decode 为何 memory-bound，Llama-70B 算一遍，batching 摊薄权重却摊不薄 KV。

### 5. 分布式 · `05-distributed/` — 5 章

- [`01-tp-pp-ep.md`](05-distributed/01-tp-pp-ep.md) — TP 切法 / PP 流水气泡 / EP expert 负载均衡。
- [`02-disaggregated.md`](05-distributed/02-disaggregated.md) — Prefill/Decode 分离的数据流 + NIXL RDMA + 决策表。
- [`03-expert-parallel-deep-dive.md`](05-distributed/03-expert-parallel-deep-dive.md) — MoE AllToAll 6 个后端、EPLB、宽 EP 部署模式。
- [`04-context-parallel.md`](05-distributed/04-context-parallel.md) — PCP / DCP 双维度长上下文切分；MLA 模型 `a2a` backend 省 NCCL。
- [`05-large-scale-cluster-inference.md`](05-distributed/05-large-scale-cluster-inference.md) — 🆕 千卡/万卡实战：通信墙/故障墙/长尾墙、EP 同步长尾、AllToAll 撞网络、blast radius、6 类规模化故障 runbook。

### 6. 工程问答 · `06-interview/` — 4 章

- [`01-common-questions.md`](06-interview/01-common-questions.md) — 30 个问题按结论、机制、取舍、验证 / 回滚和追问展开。
- [`02-system-design.md`](06-interview/02-system-design.md) — 需求优先的容量、架构、故障域、发布与成本推演。
- [`03-capacity-and-troubleshooting-drills.md`](06-interview/03-capacity-and-troubleshooting-drills.md) — 8 道带单位计算题 + 8 类证据优先故障题。
- [`04-mock-interview-and-rubric.md`](06-interview/04-mock-interview-and-rubric.md) — 概念 / 源码 / 计算 / 设计 / 事故五轮评分与项目叙事模板。

### 7. 实操 · `07-hands-on/` — 8 章

- [`01-setup.md`](07-hands-on/01-setup.md) — uv 环境 / 预编译 vs 源码 / GPU 检查。
- [`02-trace-a-request.md`](07-hands-on/02-trace-a-request.md) — debugger 跟一个请求从 HTTP 到 token。
- [`03-mini-experiments.md`](07-hands-on/03-mini-experiments.md) — 5 个动手实验（block_size / prefix hit / batching / 量化等）。
- [`04-profiling-and-debugging.md`](07-hands-on/04-profiling-and-debugging.md) — torch.profiler / NVTX / py-spy / 显存泄漏。
- [`05-serve-openai-api.md`](07-hands-on/05-serve-openai-api.md) — 从零启动、健康检查、鉴权、streaming 与失败证据。
- [`06-benchmark-methodology.md`](07-hands-on/06-benchmark-methodology.md) — 固定变量、quality gate、goodput 与可复现 benchmark。
- [`07-tuning-playbook.md`](07-hands-on/07-tuning-playbook.md) — 从症状到单变量、可回滚调优实验。
- [`08-production-capstone.md`](07-hands-on/08-production-capstone.md) — 交付可运维、可扩容、可回滚的证据包。

### 8. 生产部署 · `08-production-deployment/` — 16 章

- [`01-deployment-architectures.md`](08-production-deployment/01-deployment-architectures.md) — vLLM Production Stack / llm-d / AIBrix 三套参考栈对比。
- [`02-smart-routing-and-load-balancing.md`](08-production-deployment/02-smart-routing-and-load-balancing.md) — prefix-cache aware / session sticky / 负载打分。
- [`03-gateway-and-service-mesh.md`](08-production-deployment/03-gateway-and-service-mesh.md) — Istio + Gateway API Inference Extension + ExtProc。
- [`04-autoscaling-and-capacity.md`](08-production-deployment/04-autoscaling-and-capacity.md) — KEDA / 容量公式 / 冷启动 / 优雅 drain。
- [`05-slo-and-observability.md`](08-production-deployment/05-slo-and-observability.md) — TTFT/TPOT/p99 SLO + Prometheus + OTel。
- [`06-reliability-and-failure-modes.md`](08-production-deployment/06-reliability-and-failure-modes.md) — 8 个失效模式与防护。
- [`07-incident-playbook.md`](08-production-deployment/07-incident-playbook.md) — 8 个真实故障 runbook。
- [`08-monitoring-cookbook.md`](08-production-deployment/08-monitoring-cookbook.md) — 可直接抄走的 PromQL / 告警规则 YAML / Grafana dashboard 骨架。
- [`09-vllm-doctor-skill.md`](08-production-deployment/09-vllm-doctor-skill.md) — 把 06-07-08 章人工流程编成 agent 自动跑：7 阶段工作流 + 决策树 + 三级整改 + 离线 dry-run。
- [`10-gpu-utilization-and-tail-latency.md`](08-production-deployment/10-gpu-utilization-and-tail-latency.md) — 🆕 全链路性能诊断：GPU-Util 为何是谎言、MBU/MFU、带宽/利用率为何打不满、长尾 8 类根因与处置。
- [`11-security-and-multi-tenancy.md`](08-production-deployment/11-security-and-multi-tenancy.md) — 威胁模型、auth、quota、artifact / LoRA trust、脱敏与 tenant isolation。
- [`12-upgrades-rollbacks-and-compatibility.md`](08-production-deployment/12-upgrades-rollbacks-and-compatibility.md) — 兼容矩阵、golden、shadow / canary、drain 与 rollback。
- [`13-384-h100-glm-deepseek-deployment.md`](08-production-deployment/13-384-h100-glm-deepseek-deployment.md) — 48 节点/384×H100：GLM-5.2、DeepSeek-V4-Flash/Pro 的副本内并行、H100 验证边界、上线门禁与故障处置。
- [`14-384-ascend-910b-glm-deepseek-deployment.md`](08-production-deployment/14-384-ascend-910b-glm-deepseek-deployment.md) — 48 节点/384×910B：vLLM Ascend 软件栈、GLM-5.2 与 DeepSeek-V4 Flash/Pro 的 8/16/32 卡服务单元、HCCL、rank、Kubernetes 与发布 runbook。
- [`15-end-to-end-latency-profiling-and-optimization.md`](08-production-deployment/15-end-to-end-latency-profiling-and-optimization.md) — 客户端到 kernel 的延迟账本：metrics/OTel/Torch profiler、H100 Nsight/NCCL、910B Ascend PT/MS Service Profiler 与模型专项优化闭环。
- [`16-mooncake-distributed-inference-storage.md`](08-production-deployment/16-mooncake-distributed-inference-storage.md) — Mooncake P/D 直传、共享 DRAM/SSD KV 池、MultiConnector、RDMA、Prometheus、Kubernetes 与生产故障 Runbook。

### 9. 应用特性 · `09-advanced-features/` — 5 章

- [`01-sampling-and-logits.md`](09-advanced-features/01-sampling-and-logits.md) — 当前 sampling 顺序、backend、seed 边界、logprobs 与投机兼容性。
- [`02-structured-output.md`](09-advanced-features/02-structured-output.md) — `auto` / 显式后端、grammar compile / bitmask、fallback 与错误边界。
- [`03-multimodal.md`](09-advanced-features/03-multimodal.md) — 图像/视频/音频编码器、encoder cache、Qwen2-VL。
- [`04-lora-serving.md`](09-advanced-features/04-lora-serving.md) — LoRAModelManager / Punica / 多 LoRA batching。
- [`05-embedding-and-pooling.md`](09-advanced-features/05-embedding-and-pooling.md) — BGE / E5 / BGE-M3 / 复用 vLLM 引擎做 embedding。

---

## vLLM 仓库地标速查

<!-- vllm-source: {"path":"vllm/v1/core/kv_cache_utils.py","symbol":"hash_block_tokens"} -->
[源码锚点：vllm/v1/core/kv_cache_utils.py · hash_block_tokens](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/core/kv_cache_utils.py#L596)

| 想知道什么 | 去哪里看 |
| --- | --- |
| 用户怎么调用 vLLM | `vllm/entrypoints/llm.py`、`vllm/entrypoints/openai/api_server.py` |
| 引擎主循环 | `vllm/v1/engine/core.py`、`vllm/v1/engine/llm_engine.py` |
| 调度器（决定本步跑哪些请求） | `vllm/v1/core/sched/scheduler.py` |
| KV cache 块管理 | `vllm/v1/core/kv_cache_manager.py`、`block_pool.py` |
| Prefix caching hash | `vllm/v1/core/kv_cache_utils.py` (`hash_block_tokens`) |
| Model runner（前向） | `vllm/v1/worker/gpu_model_runner.py` |
| Attention 后端选择 | `vllm/v1/attention/backends/`（FlashAttn / FlashInfer / Triton / MLA） |
| PagedAttention CUDA | `csrc/attention/` |
| 模型实现 | `vllm/model_executor/models/`（按当前 registry 核准架构支持） |
| 张量并行 / 集合通信 | `vllm/distributed/parallel_state.py` |
| 量化 | `vllm/model_executor/layers/quantization/` |
| 投机解码 | `vllm/v1/spec_decode/` |
| 采样 | `vllm/v1/sample/`、`csrc/sampler.cu` |
| KV transfer（disaggregated）| `vllm/distributed/kv_transfer/`、`vllm/v1/kv_offload/` |

完整地图见 [`01-overview/04-project-structure.md`](01-overview/04-project-structure.md)。

---

## 学习方法（5 条铁律）

1. **概念落到代码。** 不看博客的二手解读，源码才是唯一真相。
2. **先看数据契约，再看内部。** 读 Scheduler 之前先看 `SchedulerOutput` 的字段；读 KV manager 之前先看 `KVCacheBlocks` 的形状。
3. **打 print 比读注释快。** 用 `LLM("facebook/opt-125m").generate(...)` 给 Scheduler、KVCacheManager 各加一句 print，跑一次就懂。
4. **三张图刻进脑子。** 请求生命周期、KV 物理↔逻辑映射、Scheduler 一步内的决策流。每张都自己画一遍。
5. **永远做对比。** 讲 vLLM 的优势，必须能讲清"HF Transformers 是怎么做的、为什么慢"。

---

## 工程理解自检清单

读完整套笔记后，下面每个问题应该能在 1-2 分钟内讲清，并指出对应源码位置：

- Paged KV cache、continuous batching 与 prefix caching 分别解决什么，代价是什么？
- KV block size 太大太小各有什么问题？如何在目标 backend / workload 上验证？
- Continuous batching 和 static batching 的本质区别？为什么 GPU 利用率提升？
- Prefix caching 的 hash 怎么算？怎么避免冲突？多模态怎么处理？
- Chunked prefill 解决了什么？如何核准目标版本默认值与模型限制？
- Tensor parallel 在 MLP 用 column → row 的原因？AllReduce 落在哪？
- Speculative decoding 的接受率怎么算？拒绝采样的数学推导写一遍。
- FP8 / INT8 / INT4 各自的精度损失主要发生在哪？
- V0 → V1 重构的三个最大改变是什么？为什么这么改？
- KV 不够时怎么处理？V1 默认 recompute 还是 swap，为什么？

每题都有专门展开，见 [`06-interview/01-common-questions.md`](06-interview/01-common-questions.md)。

---

## 必读资料

按阅读顺序：

1. **PagedAttention 论文** — Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023。先读这篇，再读代码。
2. **Continuous Batching 博客** — Anyscale, *How Continuous Batching Enables 23× Throughput*。
3. **vLLM 官方文档** — https://docs.vllm.ai/ ，重点看 design / kernel 章节。
4. **FlashAttention v1/v2/v3** — 理解 SRAM tiling 的关键。
5. **Speculative Decoding 论文** — Leviathan et al., 2023, *Fast Inference from Transformers via Speculative Decoding*。
6. **EAGLE3 / MTP / PARD / Suffix Decoding** — 分别理解 hidden-state drafter、原生多 token 头、并行 draft 与 workload-aware proposal。

---

## 构建与部署

把这份手册转成 HTML 网站、PDF、EPUB，或者部署到 GitHub Pages，看 [`DEPLOY.md`](DEPLOY.md)。一键命令（脚本会自动用项目相对路径，不再依赖固定位置）：

```bash
python3 build_html.py      # → ../vllm-learning-html/  (含搜索 + 暗黑切换 + Mermaid + 阅读时间)
python3 build_pdf_epub.py  # → ../vllm-learning-html/vllm-learning.pdf + .epub
./deploy_gh_pages.sh <repo-url>
```

如果你想把材料放别处，设环境变量 `VLLM_LEARNING_SRC` / `VLLM_LEARNING_DST` 即可。

---

## 自动排障 Skill：`vllm-doctor`

仓库内置了一个 Claude Code skill `vllm-doctor`，把第 06-07-08 章里散落的 incident playbook 编成 agent 可以自动跑的 7 阶段流程：环境探测 → 拉 Golden 3 指标 → 决策树路由 → 深度诊断 → 生成整改计划 → 显式审批后执行 mutation → 恢复验证与报告。缺少当前 metric、threshold 或证据时 fail closed，不把“没有 active route”误报为恢复。

**安装到本地 Claude Code**：

```bash
cp -r .claude/skills/vllm-doctor ~/.claude/skills/
```

**触发**：在 Claude Code 里输入 `/vllm-doctor`（前置必须 export `VLLM_NAMESPACE` / `PROM_URL` / `KUBECONFIG`）。

**不连集群也想验证逻辑**：

```bash
export VLLM_DOCTOR_FIXTURE=/path/to/golden3.json   # 跳过 Prometheus，直接喂 mock 数据
```

覆盖的 8 类事故：KV 抢占级联、NCCL hang、GPU OOM、客户端重试雪崩、prefix cache 命中率塌方、冷启动、输出质量异常、LoRA 适配器抖动。完整说明见 [`.claude/skills/vllm-doctor/SKILL.md`](.claude/skills/vllm-doctor/SKILL.md)。

---

## 贡献与扩展

发现 `file_path:line_number` 失效了？vLLM 主分支变化快，欢迎 PR 修正。

想新加一章？沿用每章统一的"章首导读 + 正文 + 小结/自检/下一步"模板（任意章可作范例）。

---

**开始读 [`01-overview/01-what-is-vllm.md`](01-overview/01-what-is-vllm.md)。** 或者从 [`00-prerequisites.md`](01-overview/00-prerequisites.md) 铺前置。
