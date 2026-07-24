# Part I · 总览 · 入门与架构

vLLM 的全貌：从 token/KV cache 等前置概念，到三层架构、V0→V1 演进、进程模型与 IPC 内部机制。

← [返回主目录](../README.md) · 📖 [完整目录](../CONTENTS.md)

## 本章目录

| 文件 | 为什么读 |
| --- | --- |
| [`00-prerequisites`](00-prerequisites.md) | 🆕 token / KV cache / TTFT·TPOT / TP·PP·DP 一次铺平，零基础起点 |
| [`01-what-is-vllm`](01-what-is-vllm.md) | vLLM 是什么，为什么快，靠哪三大武器 |
| [`02-architecture`](02-architecture.md) | 三层进程、四个核心数据结构、一步推理的完整数据流 |
| [`03-v0-vs-v1`](03-v0-vs-v1.md) | V1 重构改了什么，为什么改 |
| [`04-project-structure`](04-project-structure.md) | 1700+ 文件按模块分类导航（很长，按需查阅） |
| [`05-process-and-ipc-internals`](05-process-and-ipc-internals.md) | fork vs spawn / ZMQ ROUTER-DEALER / 共享内存零拷贝 |

