# Part III · 源码走读 · 引擎全链路

沿一次请求的数据流逐层走读源码：入口→调度→KV 管理→Model Runner→Attention→CUDA 内核→模型架构→输入输出。

← [返回主目录](../README.md) · 📖 [完整目录](../CONTENTS.md)

## 本章目录

| 文件 | 为什么读 |
| --- | --- |
| [`01-entry-points`](01-entry-points.md) | `LLM` / `AsyncLLM` / `EngineCore` 的调用链 |
| [`02-scheduler`](02-scheduler.md) | `Scheduler.schedule()` 完整走读，token budget + preemption |
| [`02b-scheduling-policies`](02b-scheduling-policies.md) | FCFS vs PRIORITY、优先级反演、抢占代价量化 |
| [`03-kv-cache-manager`](03-kv-cache-manager.md) | `allocate` / `free` / `hash` 的代码级细节 |
| [`04-model-runner`](04-model-runner.md) | `execute_model`：输入拼装 / forward / sampler |
| [`05-attention-backends`](05-attention-backends.md) | FlashAttn / FlashInfer / Triton / MLA 怎么选 |
| [`06-cuda-kernels`](06-cuda-kernels.md) | PagedAttention v1/v2、RoPE、RMSNorm CUDA 实现 |
| [`07-model-architectures`](07-model-architectures.md) | MLA / Mamba / MoE / GQA 在源码层的差异 |
| [`08-input-processing`](08-input-processing-and-tokenization.md) | 从 OpenAI 请求、chat template、tokenizer 到 EngineCoreRequest |
| [`09-output-processing`](09-output-processing-and-streaming.md) | 从 sampler/detokenizer 到 SSE、finish reason 与取消传播 |

