# Part VIII · 生产部署 · 上线与运维

上线全景：部署架构、智能路由、网关、弹性伸缩、SLO 与可观测性、可靠性、监控、安全、升级。

← [返回主目录](../README.md) · 📖 [完整目录](../CONTENTS.md)

## 本章目录

| 文件 | 为什么读 |
| --- | --- |
| [`01-deployment-architectures`](01-deployment-architectures.md) | vLLM Production Stack / llm-d / AIBrix 三套参考栈 |
| [`02-smart-routing`](02-smart-routing-and-load-balancing.md) | prefix-cache aware / session sticky / 负载打分 |
| [`03-gateway-and-service-mesh`](03-gateway-and-service-mesh.md) | Istio + Gateway API + ExtProc |
| [`04-autoscaling`](04-autoscaling-and-capacity.md) | KEDA / 容量公式 / 冷启动 / drain |
| [`05-slo-and-observability`](05-slo-and-observability.md) | TTFT/TPOT/p99 + Prometheus + OTel |
| [`06-reliability`](06-reliability-and-failure-modes.md) | 8 个失效模式与防护 |
| [`07-incident-playbook`](07-incident-playbook.md) | 8 个真实故障 runbook |
| [`08-monitoring-cookbook`](08-monitoring-cookbook.md) | 可直接抄走的 PromQL / 告警 / Grafana 骨架 |
| [`09-vllm-doctor-skill`](09-vllm-doctor-skill.md) | 把人工流程编成 agent 自动跑 |
| [`10-gpu-utilization`](10-gpu-utilization-and-tail-latency.md) | 🆕 GPU-Util 是谎言、MBU/MFU、长尾 8 类根因 |
| [`11-security`](11-security-and-multi-tenancy.md) | 威胁模型、auth、quota、tenant isolation |
| [`12-upgrades`](12-upgrades-rollbacks-and-compatibility.md) | 兼容矩阵、canary、drain、rollback |

