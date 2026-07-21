# vLLM 上线前 15 条 Checklist

每条都对应一个被本 skill 收录的失败模式。它是核对框架，不提供跨模型、硬件和平台通用的默认值。

## 容量 & 弹性

- [ ] **1. scale-to-zero 决策有证据**：记录目标模型 cold-start 分布、首请求 SLO、成本与 warm-pool 策略。
- [ ] **2. cooldown/稳定窗口经过 burst 演练**：能抑制 flapping，也不会错过容量缺口。
- [ ] **3. 节点容量预留可计算**：由供应时间、扩容 SLO、故障域和成本预算推导。

## KV / 显存

- [ ] **4. 显存/KV headroom 经过实测**：覆盖业务长度、并发、backend 与失败点，记录安全水位适用范围。
- [ ] **5. seq/token/chunked-prefill 联合调参**：用 SLO、吞吐、KV 和 preemption 的 Pareto 结果选值。
- [ ] **6. 长上下文隔离规则来自长度分布**：用容量和公平性实验决定分池边界。

## NCCL / 通信

- [ ] **7. collective timeout/monitoring 做过版本验证**：变量、单位与语义来自目标 NCCL/PyTorch 文档，并完成 staging hang 演练。
- [ ] **8. 硬件错误信号有平台 runbook**：定义 counter 语义、关联证据、节点处置审批和恢复条件。
- [ ] **9. 网络策略匹配实际 transport**：按拓扑、接口和动态端口验证，不抄固定端口段。

## 生命周期 & 健康检查

- [ ] **10. drain 状态机演练通过**：先从新请求路由移除，再观察 inflight/running/waiting，最后按目标平台信号终止；不依赖当前不存在的公共 `/shutdown`。

## 流量管理

- [ ] **11. Gateway admission contract 已定义**：租户预算、queue/KV 安全区间、状态码、重试和回滚都经过负载演练。
- [ ] **12. 路由变更走 canary**：按覆盖请求数、prefix token 命中基线、SLO 和错误预算毕业。

## 质量 & 模型

- [ ] **13. 版本化 golden/质量集通过**：覆盖业务切片且不含用户敏感数据；质量 gate 失败停止放量，回滚仍需明确批准。
- [ ] **14. quantization 变更必须重新校准**：FP8 / AWQ / GPTQ 切换前后用同一组 calibration set 验证。

## 可观测性

- [ ] **15. 指标契约已抓取和演练**：TTFT、queue、KV、preemption 及网关错误预算查询存在且告警能触发；缺失信号必须 fail closed。

## 来源

`vllm-learning/08-production-deployment/06-reliability-and-failure-modes.md` +
`vllm-learning/08-production-deployment/04-autoscaling-and-capacity.md`
