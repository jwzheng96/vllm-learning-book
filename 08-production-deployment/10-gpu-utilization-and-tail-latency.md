# 10. 全链路性能诊断：GPU 为什么打不满、利用率上不去、长尾怎么治

> **谁该读这一篇？** 已经把服务跑起来、SLO 也定了，但发现"卡很贵、吞吐却不及理论值一半""nvidia-smi 显示 100% 但吞吐很低""p50 很好 p99 爆炸"的人。这章是把 [`04-optimizations/05-roofline-and-arithmetic-intensity.md`](../04-optimizations/05-roofline-and-arithmetic-intensity.md) 的理论上限，对到生产实测差距上，做一次从客户端到 SM 的全链路归因。
>
> **前置阅读：** [`05-slo-and-observability.md`](05-slo-and-observability.md)（TTFT/TPOT/SLI 体系）、[`04-optimizations/05-roofline-and-arithmetic-intensity.md`](../04-optimizations/05-roofline-and-arithmetic-intensity.md)（存算比/roofline，是本章的理论底座）、[`02-core-concepts/02-continuous-batching.md`](../02-core-concepts/02-continuous-batching.md)
>
> **耗时：** 约 25 分钟
>
> **学完能：**
> 1. 解释为什么 `nvidia-smi` 的 GPU-Util 接近 100% 也不代表算力/带宽被用满，并改用 MBU / MFU 衡量。
> 2. 画出请求从客户端到 SM 的全链路，指出每一段会"漏掉"吞吐的地方。
> 3. 拿到一组 vLLM metric，用一张决策表定位瓶颈是排队、batch 太小、CPU overhead、KV 压力还是通信。
> 4. 系统列出长尾延迟的根因，并对每个根因给出 vLLM 里的具体处置手段。

理论上限（roofline）告诉你这张卡**最多**能跑多快；这章讲的是为什么实测**总是**比理论低，以及差距藏在链路的哪一段。生产里大部分"GPU 很贵但不够用"的问题，根因不在卡，在链路。

---

## 1. 先破一个误区：`nvidia-smi` 的利用率是个谎言

最常见的误判：`nvidia-smi` 显示 `GPU-Util 100%`，于是认定"GPU 已经满了，只能加卡"。

`GPU-Util` 的真实定义是 **"过去采样窗口内，至少有一个 kernel 在 SM 上跑的时间占比"**。它只回答"GPU 闲没闲着"，**不回答"算力/带宽用了几成"**。一个 batch=1 的 decode kernel 把 SM 占着（Util=100%），但实际只用了 ~0.3% 的算力（回顾上一章：AI=1，离拐点 300 差两个数量级）。**Util 100% 和算力打满是两件完全不同的事。**

生产上真正要看的是两个"打满率"：

| 指标 | 定义 | 怎么算（粗略） | decode 关注 |
| --- | --- | --- | --- |
| **MFU**（Model FLOPs Utilization）| 实际有效 FLOPs / 峰值 FLOPs | `2 · 参数量 · 总token数 / 时长 / 峰值TFLOPS` | prefill 看它 |
| **MBU**（Model Bandwidth Utilization）| 实际 HBM 读写 / 峰值带宽 | `(权重字节 + KV字节) · 步频 / 峰值带宽` | **decode 看它** |

判据来自上一章的存算比：**decode 是 memory-bound，健康标志是 MBU 高（70%+）而不是 MFU 高**；prefill 是 compute-bound，健康标志是 MFU 高。如果 decode 的 MBU 只有 30%，说明带宽没喂满——这才是"打不满"，而且加卡解决不了，要从链路找原因。

> 实操：MBU/MFU 没有现成 metric，但可以用 vLLM 的 `vllm:iteration_tokens_total`（每步 token 数）、步频、模型参数量手算，或用 `nsys` / `torch.profiler` 看 kernel 级的 achieved occupancy 和 DRAM throughput（[`07-hands-on/04-profiling-and-debugging.md`](../07-hands-on/04-profiling-and-debugging.md)）。

---

## 2. 全链路：吞吐在哪几段漏掉

一个请求从客户端到 SM，要穿过下面每一段。**任何一段慢，GPU 都会饿着**——它不是被算力限制，而是没活干在等上游：

```mermaid
flowchart LR
    C[客户端] --> LB[网关/LB<br/>路由+排队]
    LB --> TOK[tokenize<br/>HTTP 解析]
    TOK --> WQ[waiting 队列]
    WQ --> SCH[Scheduler<br/>组 batch + token budget]
    SCH --> PREP[输入拼装<br/>block table / sampling meta]
    PREP --> LAUNCH[kernel launch<br/>CPU→GPU]
    LAUNCH --> SM[SM 计算<br/>+ HBM 访存]
    SM --> SAMP[sampler<br/>detokenize]
    SAMP --> NET[流式回传]
    classDef cpu fill:#fef3c7,stroke:#b45309,color:#1a1f29;
    classDef gpu fill:#dcfce7,stroke:#15803d,color:#1a1f29;
    class C,LB,TOK,WQ,SCH,PREP,LAUNCH,SAMP,NET cpu;
    class SM gpu;
```

注意：**黄色的全是 CPU / 网络段，只有一段绿色是 GPU 真正干活**。如果黄色段加起来的时间能和绿色段相比，GPU 就会出现"空泡"——算完一步，等 CPU 准备下一步。这在小 batch、快 decode（如 8B 模型）时尤其致命：单步 GPU 时间可能才 5 ms，而 Python 侧组 batch + launch 也要几 ms，**bubble 占比可达 30-50%**。

vLLM 针对这条链路的关键武器：

- **Async scheduling**：第 n+1 步的调度/输入拼装与第 n 步的 GPU 计算重叠，消掉 CPU bubble（`vllm/v1/core/sched/async_scheduler.py`）。
- **CUDA Graph**：把整张 decode forward 的 kernel launch 序列录成一张图，重放时几乎零 launch 开销——专治 decode 的 CPU-launch-bound（[`04-optimizations/03-cudagraph-and-compile.md`](../04-optimizations/03-cudagraph-and-compile.md)）。
- **持久化 InputBatch**：batch 状态在步间复用，不每步重建（`vllm/v1/worker/gpu_input_batch.py`）。

---

## 3. 为什么"带宽打不满"（MBU 低）——逐项归因

MBU 低 = decode 步里 HBM 没在满速读。常见原因，按出现频率排：

1. **batch 太小（最常见）**。低 QPS 或并发上不去时，`vllm:num_requests_running` 长期个位数。decode 是 memory-bound，batch 小不影响单步带宽利用率本身，但会让你**远没用满吞吐天花板**——同样 42 ms 读一遍权重，只服务 3 个请求 vs 200 个请求。解决：提高并发（前提是 KV 装得下）、用更激进的 `max-num-seqs`、autoscaling 兜底低负载（[`04-autoscaling-and-capacity.md`](04-autoscaling-and-capacity.md)）。

2. **CPU bubble 没消掉**。CUDA Graph 没开（被某些采样参数/特性强制 eager）、或 async scheduling 失效。症状：GPU 实测步频明显低于"单步 GPU 纯计算时间"的倒数。查 CUDA Graph 是否真正命中（`enforce_eager` 没被某条路径打开），profiler 看 launch gap。

3. **KV 碎片 / 容量不足**。block 不够，请求被 preempt 或卡在 waiting，running batch 上不去。看 `vllm:gpu_cache_usage_perc` 是否贴顶、`vllm:num_preemptions_total` 是否在涨（详见 §5）。

4. **TP 通信占了带宽时间**。TP=8 时每层两次 AllReduce，如果走 PCIe 而非 NVLink，通信能吃掉每步可观时间，HBM 读取被通信穿插打断。确认 NVLink 拓扑、`NCCL_*` 配置（[`05-distributed/01-tp-pp-ep.md`](../05-distributed/01-tp-pp-ep.md)）。

5. **长上下文的 KV 墙**。回顾上一章 §5：长上下文下读 KV 主导，batch 摊不薄。这种情况下 MBU 其实可能不低（带宽确实在忙着读 KV），但吞吐天花板被 KV 字节锁死。区分方法：看耗时里 attention kernel 占比，若 attention ≫ Linear，就是 KV 墙，要上 MLA / KV-FP8 / GQA。

> 一句话判据：**MBU 低且 batch 小 → 拉并发 / 开 CUDA Graph；MBU 高但吞吐低 → KV 墙或纯粹理论上限到了（该量化或加卡）。**

---

## 4. 为什么"GPU 利用率（SM）不高"

这里的"利用率"指真正的算力占用（MFU 或 profiler 里的 achieved occupancy），不是 `nvidia-smi`。SM 闲置的来源：

- **物理性闲置（不是 bug）**：decode 本来就 memory-bound，SM 等数据是合理的。这种情况"SM 利用率低"是 roofline 决定的，不该去硬提 SM——该提的是 MBU 和吞吐。**不要把 memory-bound 的低 SM 占用当成性能 bug 去调。**
- **bubble 性闲置（要修）**：CPU 准备下一步时 SM 真空转。见 §2，靠 async + CUDA Graph 消除。
- **同步点闲置**：TP AllReduce、PP 的流水气泡（bubble）、sampler 里的 host-device 同步（如 `.item()`、动态形状）都会让 SM 停。检查采样路径有没有引入 GPU→CPU 同步。
- **prefill/decode 互相踩**：不开 chunked prefill 时，一个长 prefill 独占整步，其他请求 decode 全停，宏观上看 SM 忙但吞吐塌（[`02-core-concepts/05-chunked-prefill.md`](../02-core-concepts/05-chunked-prefill.md)）。
- **kernel 选型不佳**：attention backend 没选对、shape 没对齐导致 Tensor Core 没吃满（[`03-code-walkthrough/05-attention-backends.md`](../03-code-walkthrough/05-attention-backends.md)）。

---

## 5. 诊断决策表：从 metric 反推瓶颈

把 §3 §4 落成可执行的判断。先拉这几个 vLLM metric，再对表：

| 现象（metric） | 最可能的瓶颈 | 下一步动作 |
| --- | --- | --- |
| `num_requests_waiting` 高、`num_requests_running` 低、`gpu_cache_usage_perc` 不满 | **被 `max-num-seqs` 卡住**或调度保守 | 提高 `max-num-seqs`；检查是否人为限并发 |
| `waiting` 高、`gpu_cache_usage_perc` 贴顶（>0.95）、`num_preemptions_total` 在涨 | **KV 不够**，发生抢占级联 | 降并发 / 上 KV 量化 / 加卡 / 缩 `max-model-len` |
| `running` 健康但吞吐低、TTFT 还行、TPOT 偏高 | **CPU bubble / CUDA Graph 没命中** | 确认 CUDA Graph + async scheduling 生效，profiler 看 launch gap |
| TTFT 高、TPOT 正常 | **排队 / prefill 拥塞**（`request_queue_time_seconds` 高）| 开/调 chunked prefill、加 prefill 容量、P/D 分离 |
| TPOT 长尾、`iteration_tokens_total` 方差大 | **长 prefill 插队踩 decode** | 调小 chunked prefill 的 `max-num-batched-tokens` |
| `prefix_cache_hit_rate` 掉、TTFT 同步涨 | **prefix 缓存命中塌方**（路由没做 cache 亲和）| prefix-aware 路由（[`02-smart-routing-and-load-balancing.md`](02-smart-routing-and-load-balancing.md)）|
| 各项都正常、MBU 已 70%+ 但吞吐仍不够 | **到理论上限了** | 量化 / 换 MLA 模型 / 加卡，没有免费午餐 |

核心思路：**TTFT 异常往 prefill/排队 找，TPOT 异常往 decode/batch/KV 找，两者都正常但吞吐低往 CPU bubble 和理论上限找。**

---

## 6. 长尾（p99）请求怎么治

p50 好看是常态，p99 才是向产品 commit 的数字（[`05-slo-and-observability.md`](05-slo-and-observability.md) §2）。LLM 的长尾几乎不是随机抖动，而是**结构性**的——下面每条都有明确根因和处置：

### 6.1 队头阻塞（HOL）：一个长 prefill 拖死一整步
没开 / 没调好 chunked prefill 时，某请求 8K prompt 的 prefill 独占 200 ms，同步在跑的所有 decode 这一步 TPOT 全部 +200 ms。
- **治**：chunked prefill（V1 默认开），用 `max-num-batched-tokens` 控制单步 prefill 量级，把长 prefill 切片均摊。这是长尾治理第一刀。

### 6.2 抢占级联：KV 不够，请求被踢回去重算
KV 贴顶时，低优先级请求被 preempt（V1 默认 recompute），它的 TPOT 出现一个大台阶，且释放-重抢可能级联。
- **治**：留 KV 余量（别把 `gpu-memory-utilization` 顶到 0.98）、KV 量化扩容、admission control 限制入场并发；监控 `vllm:num_preemptions_total` 作为长尾先行指标。

### 6.3 排队：到达突刺打满入场
`max-num-seqs` 满了，新请求在 waiting 队列里等，`request_queue_time_seconds` 直接进 TTFT 长尾。
- **治**：autoscaling 以**队列深度 / waiting 数**为信号而非 GPU-Util（[`04-autoscaling-and-capacity.md`](04-autoscaling-and-capacity.md)）；入口限流把过载挡在外面而不是让它在队列里腐烂。

### 6.4 公平性：长输出请求饿死短请求（或反之）
FCFS 下一个超长生成请求长期占着 slot；纯按到达顺序则突发短请求被压在长请求后面。
- **治**：优先级调度（`PRIORITY` policy），给交互式请求高优先级、batch 任务低优先级（[`03-code-walkthrough/02b-scheduling-policies.md`](../03-code-walkthrough/02b-scheduling-policies.md)）；按业务分**独立的快/慢池**（短交互 vs 长 batch 各一组副本），避免互相污染长尾。

### 6.5 prefix 缓存命中波动
同一会话被 LB 轮询打到不同副本，本可命中的前缀全 miss，TTFT 偶发飙高。
- **治**：session-sticky / prefix-aware 路由（[`02-smart-routing-and-load-balancing.md`](02-smart-routing-and-load-balancing.md)）。

### 6.6 客户端重试雪崩
长尾请求触发客户端超时重试，重试又加重负载，正反馈把长尾推成事故。
- **治**：合理超时 + 退避 + 限流；服务端 admission control。谨慎用 hedging（对冲请求）——它能压长尾但放大负载，过载时反而火上浇油，只在有余量时开。详见 [`06-reliability-and-failure-modes.md`](06-reliability-and-failure-modes.md)。

### 6.7 冷启动 / 模型抖动
新副本拉起、权重加载、CUDA Graph capture 期间，落到它身上的请求长尾爆炸。
- **治**：就绪探针要等 capture 完成再放流量；预热（warmup）后再进 LB；优雅 drain（[`04-autoscaling-and-capacity.md`](04-autoscaling-and-capacity.md)）。

### 6.8 架构级:P/D 分离根治 prefill 对 decode 的干扰
当 prefill 和 decode 在同一实例里始终互相挤，chunked prefill 也只是缓解。彻底的做法是 **prefill / decode 分离部署**：prefill 节点专注高算力低并发，decode 节点专注大 batch 高吞吐，TPOT 长尾不再被 prefill 污染（[`05-distributed/02-disaggregated.md`](../05-distributed/02-disaggregated.md)）。代价是 KV 要跨节点传，复杂度上升——并发足够大、长尾 SLO 严苛时才值得。

---

## 小结

- `nvidia-smi` 的 GPU-Util 只说"忙没忙"，不说"用了几成"。decode 看 **MBU**、prefill 看 **MFU**。
- GPU "饿着"往往不是算力不够，是**链路上某段 CPU/网络把它喂慢了**——async scheduling、CUDA Graph、持久 InputBatch 就是堵这些洞的。
- 带宽打不满最常见是 **batch 太小**和 **CPU bubble**；其次是 KV 不足、TP 通信、长上下文 KV 墙。
- SM 利用率低要先区分**物理性闲置（memory-bound 的正常现象，别硬调）**和 **bubble 性闲置（要修）**。
- 长尾几乎全是结构性的：HOL、抢占、排队、公平性、prefix 波动、重试雪崩、冷启动——每条都有对应处置，chunked prefill / 优先级调度 / 队列驱动扩容 / prefix 路由 / P/D 分离是主力工具。

---

## 自检（先自答，再看要点）

**1. 监控显示 `nvidia-smi` GPU-Util 99%，但吞吐只有压测峰值的 1/3，老板说"卡满了，加卡"。你怎么反驳并定位？**

要点：Util 99% 只代表 SM 没闲着，不代表算力/带宽用满；很可能是小 batch decode 把 SM 占着但 MBU 很低。先看 `num_requests_running`（是不是 batch 太小）、`num_requests_waiting`（是不是被 `max-num-seqs` 卡住或 KV 不够）、确认 CUDA Graph 命中。多半是并发/链路问题，加卡不解决。

**2. TTFT 长尾爆炸但 TPOT 正常，最可能是哪一类问题？反过来呢？**

要点：TTFT 长尾 → prefill 拥塞或排队（看 `request_queue_time_seconds`、长 prefill 队头阻塞、prefix miss、冷启动）。TPOT 长尾而 TTFT 正常 → decode 侧（抢占、长 prefill 插队踩 decode、batch 抖动），先看 `num_preemptions_total` 和 `iteration_tokens_total` 方差。

**3. 为什么说"低 QPS 时单卡 GPU 利用率低"通常不是 bug？**

要点：decode 是 memory-bound，低 QPS → batch 小 → AI≪拐点 → SM 物理性闲置，这是 roofline 决定的。该做的不是硬提 SM 占用，而是靠 autoscaling 缩容省钱、或合并流量提高 batch。把它当 bug 去调内核是方向错误。

**4. 你能加的"长尾治理"手段里，哪个该最先上、哪个要最谨慎？**

要点：最先上 chunked prefill（V1 默认，治队头阻塞，几乎无副作用）；最谨慎是请求 hedging/对冲——它压长尾但放大负载，过载时会加剧雪崩，只在有余量时开，且要配限流。

---

## 下一步

- 理论底座回看：[`04-optimizations/05-roofline-and-arithmetic-intensity.md`](../04-optimizations/05-roofline-and-arithmetic-intensity.md)（本章的"理论上限"出处）。
- 配套监控落地：[`08-monitoring-cookbook.md`](08-monitoring-cookbook.md)（本章 metric 的 PromQL / 告警 / dashboard）、[`05-slo-and-observability.md`](05-slo-and-observability.md)。
- 出事时：[`07-incident-playbook.md`](07-incident-playbook.md)（抢占级联、重试雪崩等的 runbook）、[`09-vllm-doctor-skill.md`](09-vllm-doctor-skill.md)（把诊断流程自动跑）。
- 想动手 profile：[`07-hands-on/04-profiling-and-debugging.md`](../07-hands-on/04-profiling-and-debugging.md)（torch.profiler / nsys 看 launch gap 和 DRAM throughput）。
