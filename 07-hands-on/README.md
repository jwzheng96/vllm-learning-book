# Part VII · 实操 · 实操实验

动手实践：环境搭建、请求追踪、mini 实验、Profiling、API 服务、基准、调优、生产 Capstone。

← [返回主目录](../README.md) · 📖 [完整目录](../CONTENTS.md)

## 本章目录

| 文件 | 为什么读 |
| --- | --- |
| [`01-setup`](01-setup.md) | uv 环境 / 预编译 vs 源码 / GPU 检查 |
| [`02-trace-a-request`](02-trace-a-request.md) | debugger 跟一个请求从 HTTP 到 token |
| [`03-mini-experiments`](03-mini-experiments.md) | 5 个动手实验（block_size / prefix hit / batching） |
| [`04-profiling`](04-profiling-and-debugging.md) | torch.profiler / NVTX / py-spy / 显存泄漏 |
| [`05-serve-openai-api`](05-serve-openai-api.md) | 从零启动、健康检查、鉴权、streaming |
| [`06-benchmark-methodology`](06-benchmark-methodology.md) | 固定变量、quality gate、goodput |
| [`07-tuning-playbook`](07-tuning-playbook.md) | 从症状到单变量、可回滚调优 |
| [`08-production-capstone`](08-production-capstone.md) | 交付可运维、可扩容、可回滚的证据包 |

