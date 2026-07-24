# vLLM 学习手册

> **vLLM = PagedAttention + Continuous Batching + Prefix Caching**

一本写给大模型推理工程入门者的源码教程。**60 章 · 24K+ 行**，从 PagedAttention 论文到 K8s 生产部署，覆盖整条链路。每章用语义锚点对照锁定 commit 的 vLLM 源码，"读笔记 ↔ 跳源码"无缝切换。

---

## 章节导航

<div class="grid cards" markdown>

-   **[第一章 · 总览](01-overview/00-prerequisites.md)**

    ---

    前置概念、架构、V0→V1、项目结构、进程与 IPC

-   **[第二章 · 核心概念](02-core-concepts/01-paged-attention.md)**

    ---

    PagedAttention、Continuous Batching、KV Cache、Prefix Caching

-   **[第三章 · 源码走读](03-code-walkthrough/01-entry-points.md)**

    ---

    入口→调度→KV→Runner→Attention→CUDA 全链路

-   **[第四章 · 性能优化](04-optimizations/01-quantization.md)**

    ---

    量化、投机解码、CUDA Graph、存算比推导

-   **[第五章 · 分布式](05-distributed/01-tp-pp-ep.md)**

    ---

    TP/PP/EP、PD 分离、专家并行、万卡集群

-   **[第六章 · 工程问答](06-interview/01-common-questions.md)**

    ---

    30 道高频题、系统设计、模拟面试

-   **[第七章 · 实操实验](07-hands-on/01-setup.md)**

    ---

    环境、调试、Profiling、API 服务、基准

-   **[第八章 · 生产部署](08-production-deployment/01-deployment-architectures.md)**

    ---

    架构、路由、网关、SLO、可靠性、监控、升级

-   **[第九章 · 应用特性](09-advanced-features/01-sampling-and-logits.md)**

    ---

    采样、结构化输出、多模态、LoRA、Embedding

</div>

---

## 这份手册适合谁

- **系统补课**：想把 vLLM 核心机制啃透，不再只记"PagedAttention 解决了什么"
- **业务接入**：要上线 LLM 推理服务，选 v0/v1、调度策略、量化方案、部署架构
- **性能优化**：TTFT/TPOT 不达标，需要从架构层定位到内核层逐级排查
- **底层贡献**：想给 vLLM 提 PR，先得知道 scheduler / kv manager / attention 怎么咬合

---

## 学习路径

| 路径 | 适合谁 | 路线 |
| :--- | :--- | :--- |
| **30 分钟理解** | 第一次接触 vLLM | 前置 → [是什么](01-overview/01-what-is-vllm.md) → [架构](01-overview/02-architecture.md) → [API 服务](07-hands-on/05-serve-openai-api.md) |
| **源码主线** | 读代码 / 改代码 | [入口](03-code-walkthrough/01-entry-points.md) → [调度](03-code-walkthrough/02-scheduler.md) → [KV](03-code-walkthrough/03-kv-cache-manager.md) → [Runner](03-code-walkthrough/04-model-runner.md) |
| **工业实战** | 上线推理服务 | [环境](07-hands-on/01-setup.md) → [基准](07-hands-on/06-benchmark-methodology.md) → [调优](07-hands-on/07-tuning-playbook.md) → [部署](08-production-deployment/01-deployment-architectures.md) |
| **面试冲刺** | 准备面试 | [高频题](06-interview/01-common-questions.md) → [系统设计](06-interview/02-system-design.md) → [模拟面试](06-interview/04-mock-interview-and-rubric.md) |

刷完整本约 **25–35 小时**。
