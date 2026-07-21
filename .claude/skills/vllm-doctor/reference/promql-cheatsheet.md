# PromQL Cheatsheet — vllm-doctor 用到的全部指标

按 RED + USE 框架分类。vLLM 指标名已对齐锁定源码 `b23bd73f540175f9e117eaee5029cd7d8df63964`；使用前仍必须抓目标实例 `/metrics`，确认启用条件、label 和类型。网关/业务自定义指标不伪装成 vLLM 内置指标。

## 延迟（Latency）

| 用途 | PromQL |
| --- | --- |
| TTFT p99 (ms) | `histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)) * 1000` |
| TTFT p50 (ms) | `histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)) * 1000` |
| TPOT p99 (ms) | `histogram_quantile(0.99, sum(rate(vllm:request_time_per_output_token_seconds_bucket[5m])) by (le)) * 1000` |
| 端到端 p99 | `histogram_quantile(0.99, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))` |
| 队列等待时间 p99 | `histogram_quantile(0.99, sum(rate(vllm:request_queue_time_seconds_bucket[5m])) by (le))` |

## 流量（Traffic）

| 用途 | PromQL |
| --- | --- |
| 请求成功率（QPS） | `sum(rate(vllm:request_success_total[1m]))` |
| Prompt token 速率 | `sum(rate(vllm:prompt_tokens_total[1m]))` |
| 生成 token 速率 | `sum(rate(vllm:generation_tokens_total[1m]))` |
| 在跑请求数 | `sum(vllm:num_requests_running)` |
| 等待中请求数 | `sum(vllm:num_requests_waiting)` |

## 错误（Errors）

vLLM 当前没有通用 HTTP `request_failed` counter。HTTP 状态码、超时和尝试次数必须用网关/客户端的真实指标名配置；缺失时报告 `insufficient evidence`。

| 用途 | PromQL |
| --- | --- |
| 完成请求速率 | `sum(rate(vllm:request_success_total[5m]))`（按 `finished_reason` 切片） |
| HTTP 失败/429/timeout | `$GATEWAY_STATUS_QUERY`（部署方提供并审查） |
| 抢占速率 | `sum(rate(vllm:num_preemptions_total[5m]))` |

## 饱和度（Saturation）

| 用途 | PromQL |
| --- | --- |
| KV cache 使用率 | `max(vllm:kv_cache_usage_perc)` |
| Prefix token 命中率 | `sum(rate(vllm:prefix_cache_hits_total[5m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])), 1)` |
| GPU 利用率 | `avg(DCGM_FI_DEV_GPU_UTIL)` |
| HBM 拷贝利用率（带宽瓶颈） | `avg(DCGM_FI_DEV_MEM_COPY_UTIL)` |
| 显存使用 | `avg(DCGM_FI_DEV_FB_USED) / avg(DCGM_FI_DEV_FB_TOTAL)` |
| 节点负载 | `avg(node_load1)` |

## 调度健康度

| 用途 | PromQL |
| --- | --- |
| 每次迭代 token 数 p50 | `histogram_quantile(0.50, sum(rate(vllm:iteration_tokens_total_bucket[5m])) by (le))` |
| 请求 inference time p99 | `histogram_quantile(0.99, sum(rate(vllm:request_inference_time_seconds_bucket[5m])) by (le))` |
| LoRA request info | `vllm:lora_requests_info`（数据并行部署注意源码中的准确性警告） |

## 推理质量（如有自定义指标）

| 用途 | PromQL |
| --- | --- |
| 格式合规率 | `$FORMAT_COMPLIANCE_QUERY`（业务自定义） |
| EOS/截断质量 | 从 `request_success_total{finished_reason=...}` 与离线质量集组合判断，先核对实际 label 值 |

## 错误预算燃烧率（SLO）

燃烧率需要网关提供 total/failed 或等价可用性 SLI。窗口和阈值从团队 SLO/错误预算政策生成；没有 total request 信号时不得用 vLLM completion counter 拼出“无错误”。

## 联合查询：Golden 3 同时看

Grafana 面板里：

```promql
# Panel 1
histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)) * 1000

# Panel 2
sum(vllm:num_requests_waiting)

# Panel 3
max(vllm:kv_cache_usage_perc)
```

## 来源

`vllm-learning/08-production-deployment/05-slo-and-observability.md` 与锁定源码 `vllm/v1/metrics/loggers.py`
