# Part II · 核心概念 · 核心算法

vLLM 三大武器的原理：PagedAttention、Continuous Batching、Prefix Caching，以及 Chunked Prefill。

← [返回主目录](../README.md) · 📖 [完整目录](../CONTENTS.md)

## 本章目录

| 文件 | 为什么读 |
| --- | --- |
| [`01-paged-attention`](01-paged-attention.md) | 用 OS 虚拟内存类比讲 paged KV cache + block table |
| [`02-continuous-batching`](02-continuous-batching.md) | 为什么不用 static batching，迭代级调度的两个前提 |
| [`03-kv-cache-management`](03-kv-cache-management.md) | Block Pool / 引用计数 / preempt vs swap 策略 |
| [`04-prefix-caching`](04-prefix-caching.md) | Merkle 链式 hash + extra_keys，多卡/LoRA/多模态怎么算 |
| [`05-chunked-prefill`](05-chunked-prefill.md) | 长 prompt 怎么切才不卡 decode，`max-num-batched-tokens` 怎么调 |

