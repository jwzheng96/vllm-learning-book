# Part V · 分布式 · 多卡与多机

多卡/多机推理：TP/PP/EP、Prefill-Decode 分离、专家并行、Context Parallel、万卡集群实战。

← [返回主目录](../README.md) · 📖 [完整目录](../CONTENTS.md)

## 本章目录

| 文件 | 为什么读 |
| --- | --- |
| [`01-tp-pp-ep`](01-tp-pp-ep.md) | TP 切法 / PP 流水气泡 / EP expert 负载均衡 |
| [`02-disaggregated`](02-disaggregated.md) | Prefill/Decode 分离 + NIXL RDMA + 决策表 |
| [`03-expert-parallel`](03-expert-parallel-deep-dive.md) | MoE AllToAll 6 后端、EPLB、宽 EP 部署 |
| [`04-context-parallel`](04-context-parallel.md) | PCP/DCP 双维度长上下文切分 |
| [`05-large-scale`](05-large-scale-cluster-inference.md) | 🆕 千卡/万卡实战：通信墙/故障墙/长尾墙 |

