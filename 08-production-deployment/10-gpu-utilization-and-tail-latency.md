# 10. 全链路性能诊断：GPU 为什么打不满、利用率上不去、长尾怎么治

> **谁该读这一篇？** 已经把服务跑起来、SLO 也定了，但发现"卡很贵、吞吐却不及理论值一半""`nvidia-smi` 显示 100% 但吞吐很低""p50 很好 p99 爆炸"的人。这章把 [`04-optimizations/05-roofline-and-arithmetic-intensity.md`](../04-optimizations/05-roofline-and-arithmetic-intensity.md) 的理论上限，对到生产实测差距上，做一次从客户端到 SM 的全链路归因。
>
> **前置阅读：** [`05-slo-and-observability.md`](05-slo-and-observability.md)（TTFT/TPOT/SLI 体系）、[`08-monitoring-cookbook.md`](08-monitoring-cookbook.md)（PromQL 和告警）、[`04-optimizations/05-roofline-and-arithmetic-intensity.md`](../04-optimizations/05-roofline-and-arithmetic-intensity.md)（存算比/roofline）
>
> **耗时：** 约 35 分钟
>
> **学完能：**
> 1. 解释为什么 `nvidia-smi` 的 GPU-Util 接近 100% 也不代表算力/带宽被用满，并改用 MBU / MFU 衡量。
> 2. 说清 vLLM 里哪些指标来自 Scheduler，哪些来自请求完成统计，哪些来自 perf 估算计数器。
> 3. 拿到一组 vLLM metric，用一张决策表定位瓶颈是排队、batch 太小、CPU bubble、KV 压力、通信还是理论上限。
> 4. 按压测矩阵复现实验，从低 QPS、小 batch、长上下文、KV 贴顶、prefix miss 五类场景里定位问题。
> 5. 系统列出长尾延迟的根因，并对每个根因给出 vLLM 里的具体处置手段。

理论上限（roofline）告诉你这张卡**最多**能跑多快；这章讲的是为什么实测**总是**比理论低，以及差距藏在链路的哪一段。生产里大部分"GPU 很贵但不够用"的问题，根因不在卡，而在调度、KV、队列、通信和入口流量形态。

---

## 1. 先破一个误区：`nvidia-smi` 的 GPU-Util 是忙碌率，不是效率

最常见的误判：`nvidia-smi` 显示 `GPU-Util 100%`，于是认定"GPU 已经满了，只能加卡"。

`GPU-Util` 的真实含义接近 **"过去采样窗口内，至少有一个 kernel 在 GPU 上跑的时间占比"**。它只回答"GPU 闲没闲着"，**不回答"算力/带宽用了几成"**。一个 batch=1 的 decode kernel 可以让 Util 接近 100%，但实际有效算力可能只有 1% 以下，因为 decode 的算术强度太低，SM 大部分时间在等 HBM。

生产上真正要看两个"打满率"：

| 指标 | 定义 | 怎么算 | 主要看谁 |
| --- | --- | --- | --- |
| **MFU**（Model FLOPs Utilization） | 实际有效 FLOPs / 峰值 FLOPs | `rate(estimated_flops) / peak_flops` | prefill 和大 batch |
| **MBU**（Model Bandwidth Utilization） | 实际内存读写带宽 / 峰值 HBM 带宽 | `rate(estimated_read_bytes + estimated_write_bytes) / peak_bandwidth` | decode 和长上下文 |

<!-- vllm-source: {"path":"vllm/v1/metrics/perf.py","symbol":"PerfMetricsProm","anchor":"rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12"} -->
[源码锚点：vllm/v1/metrics/perf.py · PerfMetricsProm](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/perf.py#L1554)
<!-- vllm-source: {"path":"vllm/v1/metrics/perf.py","symbol":"PerfMetricsProm.__init__","anchor":"counter_flops = self._counter_cls("} -->
[源码锚点：vllm/v1/metrics/perf.py · PerfMetricsProm.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/perf.py#L1570)
<!-- vllm-source: {"path":"vllm/v1/metrics/perf.py","symbol":"PerfMetricsProm"} -->
[源码锚点：vllm/v1/metrics/perf.py · PerfMetricsProm](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/perf.py#L1548)

vLLM 当前已经有近似 perf 估算指标，不必完全手算。源码在 `vllm/v1/metrics/perf.py` 的 `PerfMetricsProm`，Prometheus 计数器分别是 `vllm:estimated_flops_per_gpu_total`、`vllm:estimated_read_bytes_per_gpu_total`、`vllm:estimated_write_bytes_per_gpu_total`（注册点见 `vllm/v1/metrics/perf.py`、`:1299`、`:1311`）。代码注释给出的 PromQL 原型在 `vllm/v1/metrics/perf.py` 和 `:1275`。

```promql
# 每 GPU 估算 TFLOPS
rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

# 每 GPU 估算内存带宽 GB/s
(
  rate(vllm:estimated_read_bytes_per_gpu_total[1m])
  + rate(vllm:estimated_write_bytes_per_gpu_total[1m])
) / 1e9
```

把上面的值再除以硬件峰值，就得到近似 MFU / MBU。比如 H100 SXM BF16 峰值约 990 TFLOPS、HBM 带宽约 3350 GB/s：

```promql
# H100 BF16 近似 MFU
rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12 / 990

# H100 近似 MBU
(
  rate(vllm:estimated_read_bytes_per_gpu_total[1m])
  + rate(vllm:estimated_write_bytes_per_gpu_total[1m])
) / 1e9 / 3350
```

**判据**来自上一章的存算比：decode 是 memory-bound，健康标志是 MBU 高，而不是 MFU 高；prefill 是 compute-bound，健康标志是 MFU 高。如果 decode 的 MBU 只有 20%-30%，说明带宽没喂满，要查 batch、CPU bubble、CUDA Graph、KV 压力或路由，而不是立刻加卡。

> 注意：这些是基于模型结构和 scheduled token 的估算指标，不是硬件 PMU 直接采样。它们适合做趋势和归因，最终确认仍要用 `nsys` / `torch.profiler` 看 kernel 级 DRAM throughput、launch gap、achieved occupancy。

---

## 2. 指标先对齐：vLLM 哪些 metric 能回答什么问题

读源码时先记住三层来源：

<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"gauge_scheduler_running = self._gauge_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L493)

1. **Scheduler 状态**：running / waiting / KV 使用率 / prefix cache / preemption。注册在 `vllm/v1/metrics/loggers.py` 到 `:625`。
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.record","anchor":"for ttft in iteration_stats.time_to_first_tokens_iter:"} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.record](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L1216)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"histogram_time_to_first_token = self._histogram_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L796)

2. **请求完成统计**：TTFT、ITL、TPOT、队列时间、prefill/decode 时间。注册在 `vllm/v1/metrics/loggers.py` 到 `:881`，观测逻辑在 `vllm/v1/metrics/loggers.py` 到 `:1211`。
<!-- vllm-source: {"path":"vllm/v1/metrics/perf.py","symbol":"ModelMetrics.get_step_perf_stats_per_gpu"} -->
[源码锚点：vllm/v1/metrics/perf.py · ModelMetrics.get_step_perf_stats_per_gpu](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/perf.py#L1339)

3. **Perf 估算**：每步按 `SchedulerOutput` 区分 prefill/decode，估算 FLOPs 和 bytes。入口是 `vllm/v1/metrics/perf.py`，新请求按 prefill 记（`:1068`），cached request 通常按 decode 记（`:1080` 到 `:1094`）。

<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"gauge_scheduler_running = self._gauge_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L493)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"gauge_scheduler_waiting = self._gauge_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L503)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"gauge_waiting_by_reason = self._gauge_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L513)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"gauge_kv_cache_usage = self._gauge_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L561)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"counter_num_preempted_reqs = self._counter_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L661)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"counter_prefix_cache_queries = self._counter_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L584)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"histogram_time_to_first_token = self._histogram_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L796)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"histogram_inter_token_latency = self._histogram_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L829)
<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__","anchor":"histogram_queue_time_request = self._histogram_cls("} -->
[源码锚点：vllm/v1/metrics/loggers.py · PrometheusStatLogger.__init__](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L922)
<!-- vllm-source: {"path":"vllm/v1/metrics/perf.py","symbol":"PerfMetricsProm"} -->
[源码锚点：vllm/v1/metrics/perf.py · PerfMetricsProm](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/perf.py#L1548)

| 你想判断什么 | 先看 metric | 源码锚点 | 解释 |
| --- | --- | --- | --- |
| batch 是否够大 | `vllm:num_requests_running` | `vllm/v1/metrics/loggers.py` | 当前进入 model execution batch 的请求数 |
| 是否在排队 | `vllm:num_requests_waiting` | `vllm/v1/metrics/loggers.py` | waiting + skipped waiting 会被写入这个 gauge，更新见 `:1070` |
| 为什么排队 | `vllm:num_requests_waiting_by_reason` | `vllm/v1/metrics/loggers.py` | `capacity` 是容量不够，`deferred` 是 LoRA/KV transfer/blocked 等约束 |
| KV 是否贴顶 | `vllm:kv_cache_usage_perc` | `vllm/v1/metrics/loggers.py` | 1.0 表示 KV cache 100% 占用，更新见 `:1081` |
| 是否发生抢占 | `rate(vllm:num_preemptions_total[5m])` | `vllm/v1/metrics/loggers.py` | 源码 Counter 名是 `vllm:num_preemptions`，Prometheus 暴露时有 `_total` 后缀 |
| prefix 是否命中 | `prefix_cache_hits / prefix_cache_queries` | `vllm/v1/metrics/loggers.py` | 本地 prefix cache，外部 KV connector 指标见 `:566` |
| TTFT 是否坏 | `vllm:time_to_first_token_seconds_bucket` | `vllm/v1/metrics/loggers.py` | 从请求进入到第一个 token |
| TPOT/ITL 是否坏 | `request_time_per_output_token_seconds_bucket`、`inter_token_latency_seconds_bucket` | `vllm/v1/metrics/loggers.py`、`:817` | ITL 是 token 间隔，TPOT 是请求级输出 token 平均 |
| 队列贡献多少 | `vllm:request_queue_time_seconds_bucket` | `vllm/v1/metrics/loggers.py` | WAITING 阶段时间 |
| 近似 MFU/MBU | `estimated_flops/read_bytes/write_bytes` | `vllm/v1/metrics/perf.py` | 基于每步 scheduled token 和模型配置估算 |

一个容易踩的坑：老资料里常见 `vllm:gpu_cache_usage_perc`，当前源码对应的是 **`vllm:kv_cache_usage_perc`**。看当前代码时以后者为准。

---

## 3. 从源码看调度瓶颈：token budget、seq budget、KV budget

性能问题最后都会落到三个 budget：

<!-- vllm-source: {"path":"vllm/config/scheduler.py","symbol":"SchedulerConfig","anchor":"max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)"} -->
[源码锚点：vllm/config/scheduler.py · SchedulerConfig](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/config/scheduler.py#L49)
<!-- vllm-source: {"path":"vllm/config/scheduler.py","symbol":"SchedulerConfig","anchor":"max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)"} -->
[源码锚点：vllm/config/scheduler.py · SchedulerConfig](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/config/scheduler.py#L63)
<!-- vllm-source: {"path":"vllm/v1/core/sched/scheduler.py","symbol":"Scheduler.schedule","anchor":"# Schedule newly needed KV blocks for the request."} -->
[源码锚点：vllm/v1/core/sched/scheduler.py · Scheduler.schedule](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/core/sched/scheduler.py#L557)

| budget | 配置/状态 | 源码 | 典型症状 |
| --- | --- | --- | --- |
| token budget | `max_num_batched_tokens` / `max_num_scheduled_tokens` | `vllm/config/scheduler.py`、`:56` | 每步 token 太少，prefill 被切太碎或 decode batch 太小 |
| seq budget | `max_num_seqs` | `vllm/config/scheduler.py` | running 达上限，waiting 增长，但 KV 未必满 |
| KV budget | block pool 可用块 | `vllm/v1/core/sched/scheduler.py` | allocate slots 失败，触发 preemption |

<!-- vllm-source: {"path":"vllm/v1/core/sched/scheduler.py","symbol":"Scheduler.schedule"} -->
[源码锚点：vllm/v1/core/sched/scheduler.py · Scheduler.schedule](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/core/sched/scheduler.py#L421)

`Scheduler.schedule()` 的注释很关键：它不硬分"prefill 阶段"和"decode 阶段"，而是让每个 request 的 `num_computed_tokens` 去追 `num_tokens_with_spec`，同一套逻辑覆盖 chunked prefill、prefix caching、speculative decoding（`vllm/v1/core/sched/scheduler.py`）。每步先拿到 `token_budget`（`:348`），然后：

1. 先调度 running 请求（`:364`），保证已经在跑的 decode 不被新 prefill 随便打断。
2. 为请求分配新 KV block（`:441` 到 `:448`）。
3. 如果 KV 不够，按策略 preempt 一个 running 请求（`:454` 到 `:483`）。
4. 再从 waiting 队列拉新请求（`:548`），检查 local/external prefix cache（`:590` 到 `:623`）。
5. 如果 chunked prefill 允许，就把新请求的 token 数裁到剩余 budget（`:659` 到 `:669`）。
6. 分配成功后进入 running（`:762` 到 `:803`）。

这解释了很多线上现象：

- `waiting` 高但 `kv_cache_usage_perc` 不高，通常是 `max_num_seqs`、LoRA budget、KV transfer 或其他约束卡住，不是显存不够。
- `kv_cache_usage_perc` 贴顶且 `num_preemptions_total` 上涨，是 KV block 不够，scheduler 已经开始牺牲部分请求。
- `max_num_batched_tokens` 调太小，TTFT 可能变稳但 prefill 被切碎；调太大，长 prefill 又可能让 decode TPOT 抖。
- `max_num_seqs` 不是越大越好，它会提高并发上限，也会提高 KV 压力和 preemption 风险。

<!-- vllm-source: {"path":"vllm/config/scheduler.py","symbol":"SchedulerConfig","anchor":"max_num_partial_prefills: int = Field(default=1, ge=1)"} -->
[源码锚点：vllm/config/scheduler.py · SchedulerConfig](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/config/scheduler.py#L70)

`SchedulerConfig` 里几个线上常调旋钮值得背下来：`max_num_partial_prefills` 控制可并发 chunked prefill 数（`vllm/config/scheduler.py`），`max_long_partial_prefills` 允许短 prompt 在某些场景跳过长 prompt（`:74` 到 `:78`），`enable_chunked_prefill` 默认开启（`:84`），`policy` 支持 `fcfs` 和 `priority`（`:109`），`async_scheduling` 用来减少 GPU 利用率空洞（`:146`）。

---

## 4. 全链路：吞吐在哪几段漏掉

一个请求从客户端到 SM，要穿过下面每一段。任何一段慢，GPU 都会饿着，它不是被算力限制，而是没活干在等上游：

```mermaid
flowchart LR
    C[客户端] --> LB[网关/LB<br/>路由+排队]
    LB --> TOK[tokenize<br/>HTTP 解析]
    TOK --> WQ[waiting 队列]
    WQ --> SCH[Scheduler<br/>token/seq/KV budget]
    SCH --> PREP[输入拼装<br/>block table / sampling meta]
    PREP --> LAUNCH[kernel launch<br/>CPU 到 GPU]
    LAUNCH --> SM[SM 计算<br/>+ HBM 访存]
    SM --> SAMP[sampler<br/>detokenize]
    SAMP --> NET[流式回传]
    classDef cpu fill:#fef3c7,stroke:#b45309,color:#1a1f29;
    classDef gpu fill:#dcfce7,stroke:#15803d,color:#1a1f29;
    class C,LB,TOK,WQ,SCH,PREP,LAUNCH,SAMP,NET cpu;
    class SM gpu;
```

黄色段全是 CPU / 网络 / 控制面，只有绿色段是 GPU 真正干活。如果黄色段加起来的时间能和绿色段相比，GPU 就会出现 bubble：算完一步，等 CPU 准备下一步。这在小模型、小 batch、低并发、复杂 logits processor、结构化输出、LoRA 多租户时尤其明显。

vLLM 针对这条链路的关键武器：

<!-- vllm-source: {"path":"vllm/config/scheduler.py","symbol":"SchedulerConfig","anchor":"async_scheduling: bool | None = None"} -->
[源码锚点：vllm/config/scheduler.py · SchedulerConfig](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/config/scheduler.py#L158)

- **Async scheduling**：减少调度带来的 GPU gap。配置注释直接写着它能避免 GPU utilization gaps（`vllm/config/scheduler.py`）。
- **CUDA Graph**：把稳定 shape 的 decode forward 捕获成图，降低 Python 到 CUDA launch 开销，见 [`04-optimizations/03-cudagraph-and-compile.md`](../04-optimizations/03-cudagraph-and-compile.md)。
<!-- vllm-source: {"path":"vllm/v1/worker/gpu_input_batch.py","symbol":"InputBatch"} -->
[源码锚点：vllm/v1/worker/gpu_input_batch.py · InputBatch](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/worker/gpu_input_batch.py#L92)

- **持久化 InputBatch**：`InputBatch` 预分配 token、block table、sampling 参数等张量（`vllm/v1/worker/gpu_input_batch.py`、`:133`、`:170`），步间增量更新，不每步重建。
- **chunked prefill**：把大 prefill 切进多个 step，让 decode 不被一个长 prompt 长时间阻塞。

---

## 5. 为什么 MBU 低：带宽没有被喂满

MBU 低 = decode 步里 HBM 没在满速读。常见原因按出现频率排：

### 5.1 batch 太小

低 QPS 或并发上不去时，`vllm:num_requests_running` 长期个位数。decode 是 memory-bound，batch 小会让同样一次权重读取只服务少量请求。解决方式不是调 kernel，而是：

- 合并流量，避免每个租户一个小实例。
- 提高 `max_num_seqs`，前提是 KV 装得下。
- 用 router 把相同模型/LoRA/session 聚合到少数副本。
- 低负载时缩容，不要追求低 QPS 下单卡满载。

### 5.2 CPU bubble 没消掉

症状：`num_requests_running` 不低，`kv_cache_usage_perc` 不满，但估算 MBU 低，profiler 里 kernel 之间有明显空白。常见原因：

- CUDA Graph 没命中，可能被动态 shape、某些采样参数、强制 eager 路径破坏。
- async scheduling 关闭或不适用。
- 自定义 logits processor、结构化输出、detokenize 在 CPU 上太重。
- Python 侧频繁构造新 batch，没有利用持久 InputBatch 的增量路径。

处理：先用 `nsys` 看 launch gap，再检查 CUDA Graph 日志和 `--enforce-eager`，最后减少 per-token CPU 工作。

### 5.3 KV 容量不足，running batch 上不去

<!-- vllm-source: {"path":"vllm/v1/core/sched/scheduler.py","symbol":"Scheduler.schedule","anchor":"# Schedule newly needed KV blocks for the request."} -->
[源码锚点：vllm/v1/core/sched/scheduler.py · Scheduler.schedule](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/core/sched/scheduler.py#L557)

KV block 不够时，scheduler 在 `allocate_slots` 失败后会 preempt running 请求（`vllm/v1/core/sched/scheduler.py` 到 `:483`）。指标组合：

```text
vllm:kv_cache_usage_perc > 0.90
rate(vllm:num_preemptions_total[5m]) > 0
vllm:num_requests_waiting 上涨
```

处理顺序：

1. 先降 admission 或 `max_num_seqs`，止住抢占级联。
2. 缩 `max_model_len` 或限制超长请求。
3. 开 KV cache FP8、GQA/MLA 模型，减少每 token KV 字节。
4. 横向扩容或做 prefix-aware routing。

### 5.4 TP/EP/DP 通信挡住 step

TP 每层有 collective，EP 有 AllToAll，DP/PP 有同步点。通信慢时 GPU 不是不忙，而是在等通信完成，HBM 读写和 SM 计算都被切碎。症状：

- 单机 NVLink 好，多机 PCIe/IB 下 TPOT p99 变差。
- `num_requests_running` 健康，但不同 rank step time 方差大。
- profiler 里 NCCL / AllToAll 占比高。

处理见 [`05-distributed/01-tp-pp-ep.md`](../05-distributed/01-tp-pp-ep.md) 和 [`05-distributed/03-expert-parallel-deep-dive.md`](../05-distributed/03-expert-parallel-deep-dive.md)：确认拓扑、NCCL、EP backend、EPLB、DP padding 和微批重叠。

### 5.5 长上下文 KV 墙

长上下文下读 KV 主导，batch 摊不薄 KV。此时 MBU 可能已经高，但吞吐仍低。这不是"没打满"，而是理论上限被 KV 字节锁死。区分方法：

- attention kernel 占比远高于 Linear/MoE。
- context length 越长，TPOT 近似线性变差。
- 增大 batch 不再明显提高 token/s。

处理方向是砍 KV 字节：GQA/MQA、MLA、KV FP8、sliding window、prefix caching、上下文裁剪。

一句话判据：**MBU 低且 batch 小，拉并发或消 CPU bubble；MBU 高但吞吐低，查 KV 墙、通信或理论上限。**

---

## 6. 为什么 SM 利用率不高

这里的"利用率"指真正的算力占用，近似用 MFU 或 profiler 里的 achieved occupancy 看，不是 `nvidia-smi`。

| 类型 | 是否问题 | 表现 | 处理 |
| --- | --- | --- | --- |
| memory-bound 的物理闲置 | 不一定 | decode MFU 低但 MBU 高 | 接受现实，优化 KV bytes 或提高 batch |
| CPU bubble | 是 | kernel 间空白，MBU/MFU 都低 | CUDA Graph、async scheduling、减少 CPU per-token 工作 |
| 通信同步 | 是 | NCCL/AllToAll 占比高，rank 方差大 | 拓扑、backend、EPLB、micro-batching |
| prefill/decode 互踩 | 是 | TTFT 或 TPOT p99 随长 prompt 抖动 | chunked prefill、P/D 分离 |
| kernel 选型不佳 | 是 | shape 不适配，Tensor Core/attention backend 未吃满 | 调 attention backend、量化 backend、对齐 batch/seq |

不要把 memory-bound 的低 SM 占用当成内核 bug。decode 的目标不是把 MFU 打到 90%，而是在给定 KV/权重字节下把 MBU、token/s、TPOT 放到可接受的 Pareto 点。

---

## 7. 诊断决策表：从 metric 反推瓶颈

先拉一组核心指标：

```promql
vllm:num_requests_running
vllm:num_requests_waiting
vllm:num_requests_waiting_by_reason
vllm:kv_cache_usage_perc
rate(vllm:num_preemptions_total[5m])
histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(vllm:request_time_per_output_token_seconds_bucket[5m])))
rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12
(rate(vllm:estimated_read_bytes_per_gpu_total[1m]) + rate(vllm:estimated_write_bytes_per_gpu_total[1m])) / 1e9
```

然后按表定位：

| 现象 | 最可能瓶颈 | 下一步动作 |
| --- | --- | --- |
| `running` 长期低，`waiting` 也低 | 流量太少或 router 分散 | 合并流量、缩容、提高单实例 batch |
| `waiting` 高，`running` 达到 `max_num_seqs`，KV 不满 | seq budget 卡住 | 提高 `max_num_seqs`，检查业务限并发 |
| `waiting_by_reason{reason="deferred"}` 高 | LoRA/KV transfer/blocked constraint | 查 LoRA 数、远端 KV、结构化输出或多模态 encoder |
| `kv_cache_usage_perc` > 0.9，preemption rate 上涨 | KV 不够 | 降入场并发、KV 量化、缩上下文、扩容 |
| TTFT p99 高，TPOT 正常，queue time 高 | 排队或 prefill 拥塞 | 队列驱动扩容、chunked prefill、P/D 分离 |
| TTFT 正常，TPOT p99 高 | decode 侧问题 | 查 KV、batch 抖动、通信、长 prefill 插队 |
| `prefix_cache_hits / queries` 掉，TTFT 同步涨 | prefix 亲和失效 | session/prefix-aware routing |
| running 健康，KV 不满，MBU/MFU 都低 | CPU bubble | CUDA Graph、async scheduling、减少 CPU 工作 |
| MBU 高，TPOT 仍高 | KV 墙或通信墙 | KV bytes 优化、MLA/GQA/KV FP8、通信优化 |
| MFU 高，TTFT 高，decode 正常 | prefill compute-bound 到顶 | 量化、加 prefill 池、P/D 分离 |

核心思路：**TTFT 异常往 prefill/排队 找，TPOT 异常往 decode/batch/KV 找，两者都正常但吞吐低，往 CPU bubble、路由分散和理论上限找。**

---

## 8. 压测矩阵：不要只跑一条 benchmark

很多团队只跑 `request_rate=inf`，然后拿最大吞吐当容量结论。这会漏掉长尾和低负载效率问题。至少跑下面 6 组：

| 实验 | 变量 | 看什么 | 结论 |
| --- | --- | --- | --- |
| 最大吞吐 | `request_rate=inf`，逐步加 `max_concurrency` | token/s、TPOT、MBU | 找吞吐天花板和 MBU 上限 |
| 低 QPS | 固定小 QPS，如 1/5/10 RPS | running、MBU、单用户 TPOT | 判断是否需要缩容/合流 |
| burst | Poisson 或 Gamma 到达，带突刺 | queue time、TTFT p99 | 检查 admission 和 autoscaling |
| 长 prompt sweep | prompt 1K/8K/32K/128K | TTFT、TPOT、attention 占比 | 找 KV 墙拐点 |
| KV 压力 | 提高并发直到 `kv_cache_usage_perc` > 0.9 | preemption rate、p99 | 设置安全水位 |
| prefix 亲和 | sticky vs random routing | hit rate、TTFT | 验证路由策略收益 |

每组都保留相同四类数据：

1. vLLM Prometheus 指标。
2. 客户端 latency 分布和请求长度分布。
3. `nsys` 或 `torch.profiler` 的 30-60 秒采样。
4. 实例配置：模型、精度、TP/PP/EP/DP、`max_num_seqs`、`max_num_batched_tokens`、chunked prefill、CUDA Graph。

没有请求长度分布的 benchmark 基本不可复现。LLM 推理的性能不是单一 QPS 函数，而是 prompt 长度、output 长度、到达过程、prefix 重复率和并行配置的联合函数。

---

## 9. 三个典型案例

### 案例 A：GPU-Util 99%，token/s 只有峰值 1/3

现象：

```text
GPU-Util: 99%
vllm:num_requests_running: 6-10
vllm:num_requests_waiting: 0
MBU: 18%
TTFT/TPOT: 都不差
```

结论：不是卡满，是低 QPS 小 batch。GPU 一直有 kernel，所以 `nvidia-smi` 显示忙；但每步只服务少量请求，带宽利用率低。

处理：缩容省钱，或把多个租户/会话合到同一池，提高 running batch。不要为了低 QPS 去调 attention kernel。

### 案例 B：TTFT p99 飙高，TPOT 正常

现象：

```text
TTFT p99: 4s
TPOT p99: 80ms
vllm:request_queue_time_seconds p99: 3.5s
vllm:kv_cache_usage_perc: 0.45
vllm:num_requests_waiting_by_reason{reason="capacity"}: high
```

结论：不是 decode 慢，也不是 KV 满，是入口到 scheduler 的排队。可能 `max_num_seqs` 太低、router 突刺过载、prefill 池不足。

处理：队列驱动扩容，调 `max_num_seqs`，检查 `max_num_batched_tokens` 是否过小导致 prefill 切得太碎；长 prompt 多时考虑 P/D 分离。

### 案例 C：TPOT p99 周期性尖刺

现象：

```text
TPOT p99: 周期性从 80ms 到 600ms
vllm:kv_cache_usage_perc: 0.95-1.0
rate(vllm:num_preemptions_total[5m]): > 0
vllm:num_requests_waiting: sawtooth
```

结论：KV 贴顶导致抢占级联。V1 默认 preempt 后会让请求回 waiting，后续重算或恢复，TPOT 出现大台阶。

处理：立刻限流或降 `max_num_seqs`，把 `kv_cache_usage_perc` 拉回 0.85 以下；随后用 KV FP8、缩 `max_model_len`、prefix-aware routing 或扩容解决容量。

---

## 10. 长尾（p99）请求怎么治

p50 好看是常态，p99 才是向产品 commit 的数字。LLM 的长尾几乎不是随机抖动，而是结构性的。

### 10.1 队头阻塞：一个长 prefill 拖死一整步

没开或没调好 chunked prefill 时，某请求 32K prompt 的 prefill 可能独占多个大 step，同步在跑的 decode 请求 TPOT 全部被拖长。

治法：chunked prefill。用 `max_num_batched_tokens` 控制单步 prefill 量级；长 prompt 多时配合 `max_num_partial_prefills` / `max_long_partial_prefills`，让短 prompt 不被长 prompt 永久压住。

### 10.2 抢占级联：KV 不够，请求被踢回去

KV 贴顶时，scheduler 会 preempt 请求释放 block。低优先级请求的 TPOT 出现大台阶，释放和重抢又可能产生级联。

治法：保留 KV 余量，不要把 `gpu-memory-utilization` 顶到 0.98；监控 `vllm:kv_cache_usage_perc` 和 `rate(vllm:num_preemptions_total[5m])`；必要时 admission control 先拒新请求。

### 10.3 排队：到达突刺打满入场

`max_num_seqs` 满了，新请求在 waiting 队列里等，`request_queue_time_seconds` 直接进入 TTFT。

治法：autoscaling 以队列深度、queue time、waiting 数为核心信号，而不是 GPU-Util；入口限流把过载挡在外面，不要让请求在队列里腐烂。

### 10.4 公平性：长输出和短交互互相污染

FCFS 下长输出长期占着 slot，短交互被压在后面；反过来，如果短交互过多，批处理任务也会被饿死。

治法：优先级调度只适合有限场景，复杂公平性放到路由层。生产上更常见的是快慢池隔离：短交互、长 batch、工具调用、多模态、LoRA 各自进不同池。

### 10.5 prefix 缓存命中波动

同一会话被 LB 轮询打到不同副本，本可命中的前缀全 miss，TTFT 偶发飙高。

治法：session-sticky / prefix-aware routing。观察 `prefix_cache_hits / prefix_cache_queries` 与 TTFT 的相关性。

### 10.6 客户端重试雪崩

长尾请求触发客户端超时重试，重试又加重负载，正反馈把长尾推成事故。

治法：合理超时、指数退避、server-side admission control。谨慎用 hedging，对冲请求只在有余量时开，过载时会放大流量。

### 10.7 冷启动和 capture 抖动

新副本拉起、权重加载、CUDA Graph capture、JIT/compile cache miss 期间，落到它身上的请求长尾爆炸。

治法：readiness probe 要等 warmup/capture 完成再放流量；扩容时先预热，缩容时 graceful drain。

### 10.8 架构级：P/D 分离根治 prefill 对 decode 的干扰

当 prefill 和 decode 在同一实例里始终互相挤，chunked prefill 只能缓解。prefill / decode 分离部署让 prefill 节点专注高算力长 prompt，decode 节点专注大 batch 高吞吐，TPOT 长尾不再被 prefill 污染。代价是 KV 要跨节点传，复杂度上升，并发足够大、长尾 SLO 严苛时才值得。

---

## 小结

- `nvidia-smi` 的 GPU-Util 只说"忙没忙"，不说"用了几成"。decode 看 **MBU**，prefill 看 **MFU**。
- 当前 vLLM 已经暴露 `estimated_flops/read_bytes/write_bytes` 估算指标，可直接算近似 MFU/MBU；它们来自 `vllm/vllm/v1/metrics/perf.py`。
- 调度瓶颈可以拆成 token budget、seq budget、KV budget。`waiting`、`running`、`kv_cache_usage_perc`、`num_preemptions_total` 的组合比 GPU-Util 更有用。
- 带宽打不满最常见是 batch 太小和 CPU bubble；带宽打满但吞吐仍低，常见是长上下文 KV 墙、通信墙或理论上限。
- 长尾几乎全是结构性的：HOL、抢占、排队、公平性、prefix 波动、重试雪崩、冷启动。chunked prefill、队列驱动扩容、prefix-aware routing、admission control、P/D 分离是主力工具。

---

## 自检（先自答，再看要点）

**1. 监控显示 `nvidia-smi` GPU-Util 99%，但吞吐只有压测峰值的 1/3，老板说"卡满了，加卡"。你怎么反驳并定位？**

要点：Util 99% 只代表 GPU 采样窗口内一直有 kernel，不代表算力/带宽用满。先看 `num_requests_running`、`num_requests_waiting`、近似 MBU/MFU、CUDA Graph 是否命中。若 running 低、MBU 低、waiting 低，多半是小 batch/低 QPS，加卡不解决。

**2. TTFT 长尾爆炸但 TPOT 正常，最可能是哪一类问题？反过来呢？**

要点：TTFT 长尾通常是排队、prefill 拥塞、prefix miss 或冷启动。TPOT 长尾但 TTFT 正常通常是 decode 侧：KV 抢占、通信、长 prefill 插队、长上下文 KV 墙。

**3. `kv_cache_usage_perc=0.96` 但 `num_requests_running` 不高，为什么会这样？**

要点：KV 由上下文长度和并发共同决定。少量超长上下文也能吃满 KV，让 scheduler 无法给新请求分配 block。看 `request_prompt_tokens` 分布、`max_model_len`、preemption rate，不能只看 running 数。

**4. MBU 高、MFU 低、TPOT 仍高，这说明什么？**

要点：decode 可能已经被内存带宽或 KV 读取锁死，不是 CPU bubble。方向是减少 bytes：KV FP8、GQA/MLA、prefix caching、上下文裁剪，或者通过 P/D 分离和通信优化隔离干扰。

**5. 为什么压测必须包含 burst 和长 prompt sweep？**

要点：`request_rate=inf` 只测最大吞吐，不能暴露队列长尾、冷启动、prefix hit 波动和 KV 墙。生产 workload 的长度分布和到达过程会改变瓶颈类型。

---

## 下一步

- 理论底座回看：[`04-optimizations/05-roofline-and-arithmetic-intensity.md`](../04-optimizations/05-roofline-and-arithmetic-intensity.md)。
- 配套监控落地：[`08-monitoring-cookbook.md`](08-monitoring-cookbook.md)。
- 出事时：[`07-incident-playbook.md`](07-incident-playbook.md)、[`09-vllm-doctor-skill.md`](09-vllm-doctor-skill.md)。
- 动手 profile：[`07-hands-on/04-profiling-and-debugging.md`](../07-hands-on/04-profiling-and-debugging.md)。
