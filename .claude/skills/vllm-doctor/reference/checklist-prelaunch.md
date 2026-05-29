# vLLM 上线前 15 条 Checklist

每条都对应一个被本 skill 收录的失败模式。上线前过一遍，能把 80% 的常见事故挡在前面。

## 容量 & 弹性

- [ ] **1. minReplicas ≥ 2**：KEDA / HPA 不要 scale-to-zero，避免 5-10 分钟冷启动击穿首请求 SLO。
- [ ] **2. cooldownPeriod ≥ 300s**：避免 KV 抖动触发 replica flapping。
- [ ] **3. 节点池 over-provision 10-20%**：保证扩容时有 GPU 节点可调度。

## KV / 显存

- [ ] **4. `gpu_memory_utilization ≤ 0.85`**：留 15% 给激活峰值和 allocator 碎片。
- [ ] **5. `max_num_seqs` 经过实测**：开 `--enable-chunked-prefill --max-num-batched-tokens 2048-4096`，再调 `max_num_seqs`，目标 `preempt_rate < 0.1/s`。
- [ ] **6. 长上下文请求隔离**：`prompt_len > 16k` 的请求走单独 pod 池，不要混在主池里。

## NCCL / 通信

- [ ] **7. `NCCL_TIMEOUT=60 NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ENABLE_MONITORING=1`** 写进基线 env。让卡死自动 crash + restart。
- [ ] **8. DCGM 监控 NVLink CRC / Replay 错误**：CRC > 0 立即告警，节点上 taint + 报修。
- [ ] **9. 服务网格 sidecar 排除 NCCL 端口**：6000-6100 + IB 端口不走 Envoy。

## 生命周期 & 健康检查

- [ ] **10. `terminationGracePeriodSeconds ≥ 600` + `preStop /shutdown`**：缩容时给在跑请求 10 分钟排空。

## 流量管理

- [ ] **11. Gateway 必须做 admission control**：`kv_cache_usage > 0.85` 返回 429 / 503，挡在 vLLM 前面，而不是让它过载后再抢占。
- [ ] **12. 路由变更走 canary**：smart router / prefix-aware 路由策略改动必须 5% 流量观察 `prefix_cache_hit_rate` 不塌方。

## 质量 & 模型

- [ ] **13. 上线前必做 50 条 golden prompt 的 PPL + 格式合规 baseline**。canary 阶段加自动质量门：`format_compliance < 0.95` → rollback。
- [ ] **14. quantization 变更必须重新校准**：FP8 / AWQ / GPTQ 切换前后用同一组 calibration set 验证。

## 可观测性

- [ ] **15. Golden 3 + 错误预算燃烧率必须接 Grafana + 告警**：TTFT p99、queue depth、KV usage、`error_rate` 燃烧率。On-call 必须能在 30 秒内看到这 3 个图。

## 来源

`vllm-learning/08-production-deployment/06-reliability-and-failure-modes.md` L293-311 +
`vllm-learning/08-production-deployment/04-autoscaling-and-capacity.md` L253-267
