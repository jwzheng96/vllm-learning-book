# 05. SLO 与可观测性：怎么"知道你的 LLM 服务好不好"

> **谁该读这一篇？** 负责定义 LLM 服务 SLO、搭建 metrics/log/trace 体系的 SRE / 可观测性工程师。
>
> **前置阅读：** [`02-architecture.md`](../01-overview/02-architecture.md)、[`04-autoscaling-and-capacity.md`](./04-autoscaling-and-capacity.md)
>
> **耗时：** 约 30 分钟
>
> **学完能：**
> 1. 区分 TTFT / TPOT / TTLT / Throughput 四类 SLI，并按业务场景定 p99 SLO
> 2. 把所有 vLLM metric 归到 4 大金信号（Latency/Traffic/Errors/Saturation）
> 3. 写出生产必看的 5-10 条 PromQL 与 dashboard 面板
> 4. 用 OTel trace 直接定位 TTFT/TPOT 突增的根因

> **当前复核（`b23bd73f540175f9e117eaee5029cd7d8df63964`）：** 当前核心 histogram 为 `time_to_first_token_seconds`、`inter_token_latency_seconds`、`request_time_per_output_token_seconds`、`e2e_request_latency_seconds`；请求完成 counter 是带 `finished_reason` 的 `request_success_total`。API/gateway 层 4xx/5xx 需要 HTTP 指标补齐，不能从 engine finished counter 推导全部 availability。

一个 LLM 推理服务"挂了"不像 web 服务那样明显——常常是延迟悄悄上升、token 输出变慢、用户体验下滑。没好的 observability，你只会从用户投诉得知。本节讲清楚 LLM 的 SLO 模型、4 大金信号、metric/log/trace 实战。

---

## 1. LLM 的 4 个核心 SLI（不是 1 个！）

通用服务讲 latency / availability。LLM 至少要分 4 个：

| 指标       | 全称                          | 含义                          | 谁关心        |
| -------- | --------------------------- | --------------------------- | ---------- |
| **TTFT** | Time To First Token         | 从请求发出到第一个 token 返回         | 用户感知响应快慢   |
| **TPOT** | Time Per Output Token       | 之后每个 token 的间隔（也叫 ITL）     | 用户感知输出流畅度  |
| **TTLT** | Time To Last Token          | 总时长（= TTFT + tokens × TPOT）| API 调用方超时计 |
| **Throughput** | tokens/sec, requests/sec | 吞吐                            | 容量规划       |

记住这张图：

```mermaid
gantt
    title 单个请求的时间线
    dateFormat X
    axisFormat %s

    section 时间轴
    TTFT (Time To First Token)   :a, 0, 200
    生成第 1 token                 :milestone, after a, 0
    decode token (TPOT × N)       :b, after a, 800
    末 token                       :milestone, after b, 0
```

TTLT = TTFT + (输出 token 数 - 1) × TPOT。

---

## 2. SLO 怎么定？（从产品等待预算反推）

下面的数值只是一份**待替换的练习表**，不是行业默认值：

| 场景 | TTFT (p99) | TPOT (p99) | TTLT (p99) |
| --- | --- | --- | --- |
| e-commerce 客服 | ≤ 200 ms | ≤ 50 ms | ≤ 5 s |
| 聊天 chatbot | ≤ 500 ms | ≤ 50 ms | ≤ 30 s |
| RAG 长文档 | ≤ 300 ms | ≤ 100 ms | ≤ 3 s |
| agent 工具调用 | ≤ 800 ms | ≤ 80 ms | ≤ 20 s |
| batch 摘要 | n/a | n/a | ≤ 120 s |
| 代码补全 | ≤ 100 ms | ≤ 30 ms | ≤ 3 s |

用用户研究、客户端 deadline、长度分布和容量成本填写目标；同时保留 p50/p95/p99 以区分整体回归和尾部回归。

### 为什么死磕 p99？
LLM 推理延迟是经典 "tail at scale" 问题：

- 大部分请求都不错
- 尾部请求可能遇到更长 prompt、queue、preemption、网络或硬件异常，倍数以观测为准
- 用户对 worst case 体验最敏感

p99 常用于对外尾延迟目标，p50 仍用于基线、成本与整体回归分析。

### 报错率 SLO 示例（必须替换）
- `5xx error rate < 0.1%`
- `timeout rate < 0.5%`
- `model output empty rate < 0.01%`（结构化输出场景）

---

## 3. 4 大金信号（Google SRE 风格）

把所有 LLM metric 归类到这 4 个，不会乱：

### 3.1 Latency（延迟）
- `vllm:time_to_first_token_seconds` (histogram) → TTFT
- `vllm:request_time_per_output_token_seconds` (histogram) → TPOT
- `vllm:e2e_request_latency_seconds` → TTLT
- gateway 端的 HTTP request_duration_seconds（含网络）

### 3.2 Traffic（流量）
- `vllm:request_success_total{finished_reason=...}`（engine 完成原因）；HTTP 总量/4xx/5xx 来自 gateway/API 指标
- `vllm:prompt_tokens_total` / `vllm:generation_tokens_total`
- `vllm:num_requests_running` / `vllm:num_requests_waiting`

### 3.3 Errors（错误）
- HTTP 5xx rate
- `vllm:request_success_total{finished_reason="abort"}`（进入 engine 后 abort；标签值以当前 `/metrics` 为准）
- timeout 比例
- `vllm:num_preemptions_total`（与 queue/KV/TPOT 联合判断；不能单独等同故障）

### 3.4 Saturation（饱和度）
- `vllm:kv_cache_usage_perc`（最关键的饱和度信号）
- `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`
- `DCGM_FI_DEV_GPU_UTIL`（GPU util）
- `DCGM_FI_DEV_MEM_COPY_UTIL`（HBM 带宽利用）
- `node_load1`（CPU）

---

## 4. metric 实战：vLLM 暴露了什么

打 `/metrics` 你会看到几十个 `vllm:*`。挑生产必看的：

```promql
# === 业务层 SLI ===
histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le, model_name))
histogram_quantile(0.99, sum(rate(vllm:request_time_per_output_token_seconds_bucket[5m])) by (le, model_name))
histogram_quantile(0.99, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))

# === 吞吐 ===
sum(rate(vllm:prompt_tokens_total[1m]))      # input tokens/s
sum(rate(vllm:generation_tokens_total[1m]))  # output tokens/s
sum(rate(vllm:request_success_total[1m]))    # successful req/s

# === 饱和度 ===
avg(vllm:kv_cache_usage_perc) by (instance)
sum(rate(vllm:prefix_cache_hits_total[5m])) by (instance)
/
sum(rate(vllm:prefix_cache_queries_total[5m])) by (instance)

# === 调度健康 ===
sum(vllm:num_requests_running) by (model_name)
sum(vllm:num_requests_waiting) by (model_name)
sum(rate(vllm:num_preemptions_total[5m])) by (instance)

# === Scheduler ===
histogram_quantile(0.99, sum by (le) (rate(vllm:iteration_tokens_total_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(vllm:request_queue_time_seconds_bucket[5m])))

# === 投机解码（开了才有）===
sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m]))
/
clamp_min(sum(rate(vllm:spec_decode_num_draft_tokens_total[5m])), 1)
   ↑ 接受率
```

---

## 5. Dashboard 必备的 5 个面板

Grafana 一打开应该看到这五个东西，其他都是补充：

```mermaid
flowchart TB
    P1["Panel 1 · SLO Compliance（按时间段）<br/>· TTFT p50 / p95 / p99（SLO 线标红）<br/>· TPOT p50 / p95 / p99<br/>· Error rate"]
    P2["Panel 2 · Traffic Mix<br/>· 各 model 的 req/s<br/>· input / output tokens 趋势<br/>· prompt 长度分布（heatmap）"]
    P3["Panel 3 · Saturation<br/>· 每实例 GPU cache usage %<br/>· GPU util / HBM bandwidth<br/>· Queue depth"]
    P4["Panel 4 · Cache Effectiveness<br/>· Prefix cache hit rate（按 model）<br/>· Avg cache reuse length"]
    P5["Panel 5 · Stability<br/>· Preempt rate<br/>· Aborted requests<br/>· Pod restarts / OOM kills"]
    P1 --> P2 --> P3 --> P4 --> P5

    classDef panel fill:#eff5ff,stroke:#2563eb,color:#1a1f29;
    class P1,P2,P3,P4,P5 panel;
```

---

## 6. 分布式 Tracing：OpenTelemetry 怎么用

vLLM 集成了 OpenTelemetry。启动参数：

```bash
vllm serve <model> \
    --otlp-traces-endpoint http://otel-collector:4317
```

当前 V1 在请求完成时记录一个 `llm_request` server span，并把 TTFT、E2E、queue、prefill、decode 与 token usage 作为 span attributes；它不是默认生成下图中每个内部阶段的独立 child span。Gateway/EPP/mesh spans 由对应组件接入：

```mermaid
gantt
    title trace_id 7f3a... · 单请求 span 时间线
    dateFormat X
    axisFormat %s ms

    section Edge
    gateway.handle_request    :a, 0, 3
    gateway.auth_check        :b, 1, 1
    extproc.epp.pick_endpoint :c, 2, 1
    mesh.istio_forward        :d, 3, 1
    section vLLM
    llm_request span · queue/prefill/decode attributes :e, 4, 3046
    section Edge
    gateway.response_complete :i, 3050, 5
```

**任何一个 span 异常**（queue_wait 大、prefill 慢、decode 间隔大）都能直接定位。

---

## 7. Logging：不是越多越好

日志量和内容随配置变化。先做分类、敏感字段审查与 retention budget：

| 级别       | 内容                                | 采样率   |
| -------- | --------------------------------- | ----- |
| ERROR    | 异常、OOM、collective failure | 全留并脱敏 |
| WARN     | fallback、恢复、异常状态 | 按事件留存 |
| INFO     | 启动、版本、周期 stats | 按容量设 retention |
| DEBUG    | 高体积诊断 | 临时、精确 scope、到期关闭 |

**Per-request log 不要打全 prompt**：①隐私敏感 ②单条 log 几十 KB。
要打就脱敏 + 截断（前 100 字符）。

结构化 log 用 JSON，便于 Loki / Elasticsearch 检索。

---

## 8. 关键告警

下面是 alert inventory 模板；阈值与 `for` 都由 SLO burn、基线和恢复时间填写：

| 告警                           | 触发条件                            | 严重度 |
| ---------------------------- | ------------------------------- | --- |
| TTFT p99 超 SLO              | 持续 5 分钟                         | P1  |
| TPOT p99 超 SLO              | 持续 5 分钟                         | P1  |
| HTTP availability burn       | multi-window burn policy            | P1  |
| Pod restart 异常             | 超出 rollout/节点事件基线             | P1/P2 |
| Preempt 率突增              | 高于 workload baseline 且 SLO/queue 恶化 | P2  |
| Prefix cache hit rate 变化   | 相对同类 workload 基线显著回退         | P3  |
| KV 压力                     | KV + queue + preemption 联合条件      | P2  |
| Inference Pod 失联          | 超过 scrape/readiness 容忍窗口         | P2  |
| Queue depth 持续高          | queue-time burn 的 leading threshold | P2  |

注意：**TTFT/TPOT 飙升时，先看 cache hit + preempt + queue 三件套**，多半是其中之一。

---

## 9. 生产坑：观测自己也会成为瓶颈

观测看似免费，但有坑：

### 9.1 metric 维度爆炸
按 user_id / request_id 打 metric → 高基数（high cardinality），Prometheus 内存爆。
原则：**业务标签不进 metric label**。用 log + trace 关联。

### 9.2 OTel trace 采样
高 QPS 下全量采样不现实。采样策略：

- 头部采样：按流量与预算设置固定比例
- 尾部采样：先采全，最后看是否慢/错决定丢不丢
- 关键路径采样：错误/慢请求优先，正常流量按预算；验证 collector overload 时降级

### 9.3 log 收集 sidecar 抢 CPU
Fluent-bit / Filebeat 在 LLM Pod 上抢 CPU 会让 Python 调度变慢。Pod 设 `cpuset` 隔离观测进程。

### 9.4 Prometheus scrape 失败
LLM Pod 启动慢，metrics endpoint 几分钟才 ready。scrape 在那期间 down，看起来像故障。
解决：用 `up` metric 区分"启动中"和"故障"。

---

## 10. SLO 仪表盘示例（PromQL）

把下面当模板；先核对 metric、label、bucket 与 gateway 指标再录入 dashboard：

```promql
# TTFT SLO compliance（示例阈值 500ms；跨副本按目标标签聚合）
sum(rate(vllm:time_to_first_token_seconds_bucket{le="0.5"}[5m]))
/
sum(rate(vllm:time_to_first_token_seconds_count[5m]))
# 越接近 1 越好

# Error budget burn rate (1-hour vs 30-day)
# 30-day SLO 99.9% → budget 0.1%
# 如果 1h error rate > 0.1% × 24 × 30 / 1 = 7.2x burn → 报警
(
  sum(rate(gateway_http_requests_total{status=~"5.."}[1h])) /
  sum(rate(gateway_http_requests_total[1h]))
) > (0.001 * 24 * 30)

# Top tenant 只能来自受控的 gateway 低基数/聚合指标；不要给 vLLM metric 加 user_id
topk(10, sum by (tenant_tier) (rate(gateway_http_requests_total[5m])))
```

---

## 11. 模型质量监控（LLM 特有）

LLM 还有一类质量指标，传统服务没有：

- **输出长度分布**：突然变短可能模型问题
- **EOS 命中率**：突然降 = 模型迷失了
- **格式合规率**：JSON 模式下输出能不能解析
- **拒答率**：被安全策略拦截的比例
- **用户反馈**：thumbs up/down，质量 drift 早期信号

这些通常不在 vLLM metric 里，需要在 gateway 或业务层埋点。

---

## 12. 真实场景案例分析

### 案例 1：TTFT p99 突增到 5s
看 trace：

- `queue_wait` 一直 > 2s ← 队列长
- 同时 `num_preemptions_total` 上涨 ← KV 压力
**结论**：流量上来了，KV 不够，请求排队。
**动作候选**：若 load test 证明容量不足，再扩容或 admission control；量化属于需质量/兼容验证的版本变更，不是事故中的即时开关。

### 案例 2：TPOT 抖动严重
看 metric：

- `iteration_tokens_total` 单 step 偶发尖峰
**假设**：step token 分布变化可能来自长 prefill；再用 request mix 与 trace/profile 验证。一次只改 `max_num_batched_tokens`/chunked-prefill 配置之一，观察 TTFT/TPOT/goodput 后决定保留或回滚。

### 案例 3：prefix cache hit rate 突跌
- 模型升级了？tokenizer 变了？
- 路由策略改了？session 不 sticky 了？
- 流量模式变（用户群体变化）？

需要按变更顺序排查。

---

## 小结

- LLM 的 SLI 至少 4 个：TTFT / TPOT / TTLT / Throughput，错误率和质量类指标也要进 SLO。
- 对外目标通常关注 p99/p99.9；p50/p95 仍用于基线、整体回归与容量分析。
- vLLM 的 `kv_cache_usage_perc`、`num_preemptions_total`、`prefix_cache_hits_total / prefix_cache_queries_total` 是排障三件套。
- Dashboard 最少 5 个面板：SLO 合规、流量构成、饱和度、cache 效果、稳定性。
- OTel 可串 gateway/EPP/vLLM request span；当前 vLLM 把 queue/prefill/decode 记录为 request span attributes，需结合 metrics/profile 深挖内部阶段。

## 自检

> 不用照着原文复述，重点是把现象、机制、源码入口和取舍讲顺。

**1. SLO TTFT p99 < 500ms, PromQL 算 "过去 5 分钟落在 SLO 内比例"。**

利用 histogram bucket 算"≤ 500ms 的累积比例"：

```promql
# 方案 A：直接用 le="0.5" bucket（如果有这个边界）
sum(rate(vllm:time_to_first_token_seconds_bucket{le="0.5"}[5m])) by (model_name)
/
sum(rate(vllm:time_to_first_token_seconds_count[5m])) by (model_name)
```

**结果是 0-1 之间的"合规率"**。如果 SLO 是 99%（即 p99 < 500ms 等价于 ≥99% 请求 < 500ms），告警条件：

```promql
(合规率) < 0.99
```

**如果 le 没有正好 0.5 这个 bucket**，方案 B 用 `histogram_quantile` 反查：

```promql
histogram_quantile(0.99,
  sum by (le)(rate(vllm:time_to_first_token_seconds_bucket[5m]))
) < 0.5
```

但 B 不直接给"合规率"，只能判断 p99 是否 < 0.5。

当前锁定 CLI 没有 `--prometheus-histogram-buckets` 这一公共参数。先检查 exposition 是否已有目标边界；没有时在 recording rule/OTel/gateway 层建立可验证的 SLI，而不是写入不存在的 flag。

---

**2. TPOT 抖动，按什么顺序看 metric？分别排除什么假设？**

**根因树（按检查顺序）**：

```
TPOT p99 抖动
│
├─ 1. vllm:num_preemptions_total       ← 排除 "KV 不够导致 preempt"
│      与历史 baseline 和 SLO 同时比较；
│      抢占触发的请求重新 prefill → TPOT 抖
│
├─ 2. vllm:kv_cache_usage_perc          ← 排除 "KV 接近满"
│      高水位是否危险取决于长度分布、headroom 与 preemption
│
├─ 3. vllm:iteration_tokens_total       ← 排除 "step token 数不稳定"
│      histogram 尾部重 = 有些 step 算几千 token（长 prefill 没切）
│      其他 step 算几十 token = step 时长方差大 = TPOT 抖
│
├─ 4. vllm:request_queue_time_seconds   ← 排除 "队列积压拖累"
│      不能直接影响 TPOT（在 batch 内的请求不在 queue），但能解释 TTFT 飙
│
├─ 5. prefix_cache_hits / prefix_cache_queries
│                                      ← 排除 "cache 命中率下降"
│      命中率掉 = 等效更多 prefill 算力 = step 时长涨 = TPOT 抖
│
├─ 6. nvidia-smi (实时 GPU 状态)         ← 排除 "硬件层异常"
│      ECC error、GPU util 异常低、温度过高（throttling）
│
└─ 7. trace 看 forward span 内部         ← 排除 "kernel 层异常"
       某层突然慢（如某个 attention kernel 异常）
```

**实战顺序**：先看 1-3 建立方向；若证据不足再继续到 trace/profile，不能用固定覆盖率替代归因。

---

**3. 为什么不能把 `user_id` 当 metric label？正确做法？**

**不能的原因**：

**Prometheus metric label = 笛卡尔积维度**。每多一个 unique label value，metric 就多一个 time series。

- 用户数 × metric/label 组合会形成数量级很高的 time series
- 实际内存取决于 Prometheus 版本、采样与 churn，但 cardinality 风险确定存在
- query 慢得离谱（GROUP BY 上百万 series）
- Cardinality explosion 是 Prometheus 部署最常见的事故

**正确做法**：

| 需求 | 方案 |
| --- | --- |
| 计费（按用户 token 数） | **OpenTelemetry trace** 或 **直接写日志 / Kafka**，由数据仓库（BigQuery / ClickHouse）聚合；不进 metric |
| 用户级 SLO 监控 | 在日志/trace/数仓聚合；Prometheus 只保留预先定义的低基数 tenant tier |
| 排查某用户问题 | trace 用 `user_id` 作 span attribute（trace 系统支持高基数），按需查询 |
| 用户级限流 | Redis / 内存 counter，不进 Prometheus |

**Prometheus label 选什么**：

- 低基数候选：`model_name`、`finished_reason`、`backend`、`pod`；具体上限由监控容量预算决定
- **绝对不要**：user_id、request_id、prompt（hash 也不行，hash 仍是高基数）

→ **Prometheus 用于聚合趋势，trace / log 用于个体定位**。两套互补。

---

**4. trace 中 `queue_wait` 高 `prefill` 正常，下一步看哪个 metric？**

`queue_wait` 高 = 请求在 waiting queue 里等了很久才被 schedule。下一步看：

**`vllm:num_requests_waiting`**（gauge）

**验证假设**：

- 如果 waiting 也持续高 → 真的是流量超容量，**应该扩容（HPA）**
- 如果 waiting 时高时低（spike）→ 流量突刺，需要更激进的 HPA 或 admission control
- 如果 waiting 低但 queue_wait 仍高 → **`vllm:kv_cache_usage_perc`**——KV 满了，新请求即使被 scheduler"看到"也无法 allocate slot，卡在 waiting

**进一步验证**：

- `rate(vllm:num_preemptions_total[5m])` 高 → 频繁踢人 + 重 admit，导致 queue 抖
- `vllm:num_requests_running` 是否长期低于 max_num_seqs → 如果是，说明不是 batch 满，是 KV 满

**修复路径决策**：

- 单 pod 容量够（waiting 0 spike）→ HPA 扩 pod 数
- 单 pod KV 满 → 调 `--gpu-memory-utilization` 或减 `max_num_seqs`
- workload 突刺导致 → 引入 admission control + rate limit

→ 一个 metric 看完后立刻有下一个具体动作，这是好 metric 设计的标志。

## 下一步

- 下一节：[`06-reliability-and-failure-modes.md`](./06-reliability-and-failure-modes.md)（把观测信号映射到具体故障模式）
- 配套诊断：[`10-gpu-utilization-and-tail-latency.md`](./10-gpu-utilization-and-tail-latency.md)（定了 SLI 之后——为什么实测打不满、长尾怎么从这些 metric 反推根因）
- 想看源码：`vllm/v1/metrics/`（metric 定义与 stat logger）、`vllm/tracing.py`
- 想动手：[`07-hands-on/02-trace-a-request.md`](../07-hands-on/02-trace-a-request.md) 把 OTel collector 串到本地 vLLM

---

## Sources

- [Key metrics for LLM inference — LLM Inference Handbook (BentoML)](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)
- [The P99 Problem: Designing LLM Inference for Real Users](https://agentnativedev.medium.com/the-p99-problem-designing-llm-inference-for-real-users-11deb35bb8d4)
- [LLM Observability: A Complete Guide to Monitoring Production Deployments](https://inference.net/content/llm-observability-monitoring-production-deployments/)
- [Metrics That Matter for LLM Inference — Hivenet](https://www.hivenet.com/post/llm-inference-metrics-ttft-tps)
