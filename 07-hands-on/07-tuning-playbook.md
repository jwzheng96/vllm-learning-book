# 07. 调优 Playbook：从症状到可回滚实验

> **谁该读这一篇？** 面对“慢”“吞吐不够”“GPU 没吃满”但不想靠参数玄学试错的工程师。
>
> **前置阅读：** [`06-benchmark-methodology.md`](06-benchmark-methodology.md)、[`02-trace-a-request.md`](02-trace-a-request.md)、[`04-profiling-and-debugging.md`](04-profiling-and-debugging.md)。
>
> **耗时：** 35 分钟；每个实验按容量和样本数另计。
>
> **学完能：** 从 TTFT/TPOT/throughput/KV/compile/CPU/通信/cache/tail 九类症状构造单变量实验，并给出证据、混杂因素和 rollback。

> **当前源码复核：** 指标与 V1 行为按 `b23bd73f540175f9e117eaee5029cd7d8df63964` 静态核对；参数默认值可能按 usage context 与平台动态计算，先保存 `--help` 和最终 resolved config。

---

## 1. 调优不是“参数列表”

每次变更使用同一张实验卡：

| 字段 | 必填内容 |
| --- | --- |
| 症状 | 哪个指标、哪个 percentile、从何时开始 |
| 证据 | result JSON、metrics、日志、trace/profile 索引 |
| 假设 | 一条可被结果否定的因果解释 |
| 自变量 | 只改一个配置/架构因素 |
| 控制变量 | source/model/workload/hardware/轮次顺序 |
| 预期方向 | 哪些指标应升/降/不变，不预填幅度 |
| stop condition | 错误、OOM、质量或 SLO 达到何值立即停止 |
| rollback | 恢复哪份配置/镜像，如何验证恢复 |
| 结论 | supported / rejected / inconclusive |

不要同时“提高 batch token、打开 prefix cache、换 FP8、改 TP”，否则任何收益都无法归因。

## 2. 统一诊断入口

先保存四组证据：

```bash
curl -fsS "$VLLM_BASE_URL/health" > health.txt
curl -fsS "$VLLM_BASE_URL/metrics" > metrics.txt
vllm bench serve --help > bench-help.txt
```

再从 result JSON/Prometheus 取：

- TTFT、TPOT、ITL、E2E p50/p90/p99；
- throughput、goodput、error/abort/timeout 分类；
- `vllm:num_requests_running`、`vllm:num_requests_waiting`；
- `vllm:kv_cache_usage_perc`、`vllm:num_preemptions_total`；
- `vllm:prompt_tokens_total`、`vllm:generation_tokens_total`；
- `vllm:prefix_cache_hits_total`、`vllm:prefix_cache_queries_total`；
- CPU/GPU util、显存、网络/NCCL、客户端负载；
- startup 各阶段日志与最终选择的 backend/kernel。

先证明是“排队、prefill、decode、output/network、client”哪一段，再进入分支。

## 3. 症状：TTFT 高

### 证据模式

- waiting 与 TTFT 同步升高：先怀疑容量/排队。
- waiting 低但长 prompt TTFT 高：先看 prefill、输入处理、chunking。
- 首轮高、后续低：先看 compile/warmup/cache。
- server TTFT 正常、客户端首字节高：先看 gateway/network/buffering。

### 单变量实验

| 假设 | 只改 | 预期方向 | 混杂因素 | rollback |
| --- | --- | --- | --- | --- |
| prefill 单步工作挤压 decode | `max_num_batched_tokens` 一档 | TTFT/TPOT 尾部重新分配 | 总吞吐和长 prompt step 数也变 | 恢复基线值 |
| 排队来自并发过载 | gateway concurrency/request-rate 上限 | waiting、TTFT p99 降 | admission 会降低 offered load | 恢复限流配置 |
| 输入处理占主导 | tokenizer pool/CPU 资源一个因素 | frontend 时间降，GPU 指标近似不变 | 客户端 tokenization | 恢复资源/配置 |
| 长前缀可复用 | 仅打开 prefix caching | warm-hit TTFT 降 | warmup、block 对齐、tenant 隔离 | 关闭并重启受控实例 |

不要把降低 `max_num_batched_tokens` 写成固定答案：过小会拖慢长 prefill，需扫曲线找 Pareto 点。

## 4. 症状：TPOT / ITL 高或抖动

### 先区分

- 所有请求慢：模型 forward/backend/通信/频率问题。
- 只有高负载慢：batch、抢占、排队或 noisy neighbor。
- 周期性尖峰：compile、GC、同步、监控采样、其他作业。
- streaming server 指标正常、客户端卡顿：proxy buffering/backpressure。

### 单变量实验

| 假设 | 证据 | 只改 | 预期方向 | rollback |
| --- | --- | --- | --- | --- |
| eager/compile 路径差异 | trace 出现 compile/graph gap | `--enforce-eager` A/B | 若 compile 路径是因，steady-state TPOT 或抖动改变 | 恢复原模式 |
| backend 不适配当前 shape | 日志 + profiler 的 kernel 占比 | current `--attention-backend`/config | kernel 时间方向变化 | 撤回强制选择 |
| mixed prefill 干扰 decode | ITL 峰值和大 prompt 同步 | batch token budget 一档 | ITL p99 方向改善，吞吐可能变 | 恢复基线 |
| 通信在关键路径 | nsys/NCCL span 占比高 | TP/拓扑方案一个因素 | NCCL 时间和 TPOT 改变 | 回滚拓扑/副本 |

`--enforce-eager` 是诊断对照，不自动是生产最优配置。

## 5. 症状：吞吐 / goodput 低

吞吐低要先问 offered load 是否足够、输出 token 是否一致、客户端是否成为瓶颈。

```text
offered load 不足？ ──是──> 提高受控 request rate，保持 workload
        │否
GPU 忙且 goodput 低？ ──> 先守 SLO，找 batch/模型/并行 Pareto
        │
GPU 空闲且 queue 高？ ──> CPU/调度/同步/错误/fallback
        │
GPU 空闲且 queue 低？ ──> client/network/offered load
```

单变量候选：`max_num_seqs`、batch token budget、weight quantization、KV dtype、TP/DP 形态、spec decode。每个都必须同时看：

- output token throughput 与 request goodput；
- TTFT/TPOT p99；
- error/quality；
- 显存/KV 容量；
- 实际 backend/fallback；
- 成本而非只看单卡 token/s。

## 6. 症状：KV 压力与 preemption

<!-- vllm-source: {"path":"vllm/v1/core/sched/scheduler.py","symbol":"Scheduler._preempt_request"} -->
[源码锚点：V1 preemption](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/core/sched/scheduler.py#L1208)

当前 V1 抢占释放 KV、把请求放回 waiting 并把 computed tokens 归零，后续采用 recompute；不要沿用 V0 swap 调优建议。

### 证据链

1. `kv_cache_usage_perc` 高位只是容量信号，不单独等于故障；
2. `num_preemptions_total` rate 增长；
3. waiting/TTFT/E2E 或 prompt recompute 成本同步恶化；
4. 请求长度/并发证明工作集确实超过有效 KV 容量。

### 单变量顺序

1. 降低 admission/concurrency，验证是否消除 preemption；
2. 只降低 max model/context 或 workload 长尾，验证工作集；
3. 只调整 KV 可用显存比例，守住 OOM headroom；
4. 平台支持时只改 KV dtype，同时跑质量/容量门禁；
5. 仍不满足时做副本/架构扩容。

rollback 条件：OOM、worker crash、错误率升、质量门禁失败或其他进程失去显存余量。

## 7. 症状：启动 / compile 太慢

把启动拆成：image/container → model download → weight load → distributed init → KV profile/allocation → compile/graph capture → health ready → 首个请求。

| 假设 | 证据 | 单变量实验 | 预期方向 | 混杂因素 |
| --- | --- | --- | --- | --- |
| 下载慢 | artifact/cache log | 预热 immutable model cache | download 阶段降 | 共享 cache 命中 |
| 权重加载慢 | 时间戳 + disk/network | 存储路径一个因素 | load 阶段降 | page cache |
| compile 慢 | compile 日志/trace | compilation config 一个因素 | ready time 改变 | steady-state 也可能变 |
| graph capture 慢 | capture 日志 | eager A/B | ready time 降 | steady-state TPOT/吞吐可能退化 |

冷启动优化不能只看 ready time；同时检查 steady-state、镜像大小、cache provenance 与回滚可用性。

## 8. 症状：CPU / tokenizer 瓶颈

证据：GPU 有空洞、waiting 上升、py-spy/CPU profile 在 tokenizer/input processing/serialization，client/server CPU 饱和。

单变量候选：

- tokenizer pool 配置或 frontend CPU request/limit；
- client 与 server 分离，排除 benchmark 客户端争抢；
- 输入预处理/cache（需正确 tenant/模板 key）；
- 减少不必要的 logprobs/结构化处理；
- gateway buffering/JSON 序列化路径。

预期：frontend/TTFT 降、GPU busy 上升；若 GPU 随后饱和，说明瓶颈已转移，不代表可以无限加 CPU。

## 9. 症状：通信瓶颈

先记录 TP/PP/DP、节点/NUMA、GPU 拓扑、NCCL/driver、网卡与实际 collective。

| 证据 | 假设 | 单变量实验 | 回滚 |
| --- | --- | --- | --- |
| nsys 中 NCCL 占关键路径高 | TP 过大或链路慢 | 同模型可容纳时降低 TP、增加副本 | 恢复并行布局 |
| 单节点正常、跨节点慢 | fabric/路由/MTU/IB | 在授权窗口只改一项网络配置 | 恢复平台配置 |
| 某 rank 长尾 | GPU/NUMA/noisy neighbor | 调换已验证节点/GPU | 回原调度约束 |

不要根据 GPU util 低直接断言 NCCL；需要 collective timeline 或平台 counter。

## 10. 症状：Prefix cache 命中低

命中率用 rate 计算并处理分母为零：

```promql
sum(rate(vllm:prefix_cache_hits_total[5m]))
/
clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])), 1)
```

核对：

- 完整 block 是否相同，尾部变化是否在 block 边界前；
- tokenizer/chat template/special token 是否一致；
- 多模态 hash/额外 key 是否一致；
- 请求是否被路由到持有该 cache 的实例；
- cache 是否因容量压力 eviction；
- tenant 安全策略是否允许共享。

只改一项：规范化模板、prefix-aware routing、KV 容量或 cache 开关。命中上升仍要验证 TTFT/goodput；不能为了命中率跨不可信 tenant 共享 prompt/KV。

## 11. 症状：Tail latency 高

尾延迟是混合分布。先按 input/output length、feature、tenant、route、模型、finish reason、cache hit/miss 分桶，再看 p99。

常见假设：长 prompt head-of-line、突发到达、preemption cascade、compile/graph miss、GC/CPU pause、network retry、某 rank straggler、下游客户端慢。

实验策略：

1. replay 同一 trace/seed；
2. 一次隔离一个桶或 feature；
3. 用 trace 把慢请求分段；
4. 对照 metrics 的同一时间窗；
5. 只改对应机制；
6. 同时检查 p50，避免“牺牲所有请求换 p99”。

rollback 以 goodput/error/quality 与 p50/p99 联合判定。

## 12. 配置变更的生产闭环

```text
baseline evidence
  → canary（相同 workload/流量切片）
  → 自动 gate（quality + error + SLO + cost）
  → 扩大流量
  → 观察一个完整业务周期
  → 固化 config/image/报告
```

任何阶段失败：停止扩流、摘除 canary、恢复上一 immutable image/config、等待旧请求 drain、跑 golden requests，再确认指标回基线。不要在故障现场继续叠加第二个“优化”。

## 13. 面试答题模板

> 我不会先报参数。我先按 TTFT/TPOT/queue/KV/error 把慢分到排队、prefill、decode 或 output/network，再用日志、trace、profile证明假设。实验一次只改一个变量，报告预期方向、混杂因素、stop condition 和 rollback。吞吐必须与 goodput、质量和成本一起看；最终通过 canary 和 golden requests 放量。

## 自检

1. KV usage 95% 但没有 preemption、SLO 正常，需要立刻扩容吗？
2. 降低 batch token budget 后 TTFT 降、吞吐也降，如何选？
3. GPU util 低有哪些完全不同的原因？
4. prefix hit rate 上升为什么可能没有业务收益？
5. 何时 `--enforce-eager` 只是诊断工具？

### 参考答案

1. 不需要立即扩容。KV 95% 但没有 preemption、queue 和 SLO 问题，可能只是健康的高利用率；先看增长趋势、峰值、长上下文分布、故障余量和 capacity forecast。只有接近水位且 goodput/恢复余量不足才扩容。
2. 这是 TTFT 与吞吐的显式 trade-off。按业务 SLO 选择满足 TPOT p99、TTFT 和 goodput 的最大 token budget，不能只追求其中一项；用固定 open-loop 流量做阶梯 A/B。
3. 低 GPU util 可能来自低到达率/小 batch、CPU tokenize 或 scheduler 瓶颈、网络/IPC/NCCL 等待、KV/内存带宽瓶颈、冷启动/compile、错误重试或观测采样误导。要用 queue、token throughput、MBU/MFU、trace 和系统指标分层定位。
4. 命中率只说明 prefix 被复用，不说明 load latency、命中 token 数、SLO 或成本。若命中的是短 prefix，或外部 Store load 占满网络/CPU，hit rate 上升也可能不带来 goodput 收益。
5. `--enforce-eager` 适合隔离 CUDA Graph/compile 是否导致错误、OOM 或长尾；它牺牲启动后的性能和可能的 graph 优势。诊断完成后应恢复生产模式，用同一 workload 验证，而不是把 eager 当成最终优化。

## 下一步

[`08-production-capstone.md`](08-production-capstone.md) 把服务、benchmark、调优、容量与事故处理合成一个可展示项目。
