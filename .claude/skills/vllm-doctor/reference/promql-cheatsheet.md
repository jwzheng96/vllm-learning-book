# PromQL Cheatsheet — vllm-doctor 用到的全部指标

按 RED + USE 框架分类。指标名前缀以 vLLM v0.10+ 为准。

## 延迟（Latency）

| 用途 | PromQL |
| --- | --- |
| TTFT p99 (ms) | `histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)) * 1000` |
| TTFT p50 (ms) | `histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)) * 1000` |
| TPOT p99 (ms) | `histogram_quantile(0.99, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le)) * 1000` |
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

| 用途 | PromQL |
| --- | --- |
| 失败速率 | `sum(rate(vllm:request_failed_total[5m]))` |
| 按 reason 分布 | `sum(rate(vllm:request_failed_total[5m])) by (reason)` |
| Abort 速率 | `sum(rate(vllm:num_aborted_requests_total[5m]))` |
| 抢占速率 | `sum(rate(vllm:num_preemptions_total[5m]))` |

## 饱和度（Saturation）

| 用途 | PromQL |
| --- | --- |
| KV cache 使用率 | `max(vllm:gpu_cache_usage_perc)` |
| Prefix cache 命中率 | `avg(vllm:gpu_prefix_cache_hit_rate)` |
| GPU 利用率 | `avg(DCGM_FI_DEV_GPU_UTIL)` |
| HBM 拷贝利用率（带宽瓶颈） | `avg(DCGM_FI_DEV_MEM_COPY_UTIL)` |
| 显存使用 | `avg(DCGM_FI_DEV_FB_USED) / avg(DCGM_FI_DEV_FB_TOTAL)` |
| 节点负载 | `avg(node_load1)` |

## 调度健康度

| 用途 | PromQL |
| --- | --- |
| 每次迭代 token 数 p50 | `histogram_quantile(0.50, sum(rate(vllm:iteration_tokens_total_bucket[5m])) by (le))` |
| 最长单请求推理时间 | `max(vllm:request_inference_time_seconds)` |
| LoRA 加载耗时 p99 | `histogram_quantile(0.99, sum(rate(vllm:lora_loading_seconds_bucket[5m])) by (le))` |

## 推理质量（如有自定义指标）

| 用途 | PromQL |
| --- | --- |
| 格式合规率 | `avg(vllm:format_compliance_rate)` |
| EOS 命中率 | `sum(rate(vllm:eos_token_total[5m])) / sum(rate(vllm:request_success_total[5m]))` |

## 错误预算燃烧率（SLO）

```
( 1 - (sum(rate(vllm:request_success_total[5m])) / sum(rate(vllm:request_total[5m]))) )
  /
( 1 - <SLO_TARGET, 如 0.999> )
```

> 燃烧率 > 14.4 → 1 小时内会用完一个月的预算（fast burn alert）
> 燃烧率 > 6 → 6 小时内（slow burn alert）

## 联合查询：Golden 3 同时看

Grafana 面板里：

```promql
# Panel 1
histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)) * 1000

# Panel 2
sum(vllm:num_requests_waiting)

# Panel 3
max(vllm:gpu_cache_usage_perc)
```

## 来源

`vllm-learning/08-production-deployment/05-slo-and-observability.md` L78-139
