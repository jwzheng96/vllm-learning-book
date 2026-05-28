# 02. Trace 一个请求：把内部跑明白

> **谁该读这一篇？** 想从"读过 Scheduler 源码"升级到"看着日志能解释每个 step 在干什么"的学习者；调试线上请求异常时需要 hook 进单步行为的工程师。
>
> **前置阅读：** [`07-hands-on/01-setup.md`](01-setup.md)（环境已经能跑通最小 demo），[`03-code-walkthrough/02-scheduler.md`](../03-code-walkthrough/02-scheduler.md)（Scheduler 源码概览），[`03-code-walkthrough/03-kv-cache-manager.md`](../03-code-walkthrough/03-kv-cache-manager.md)（KV 分配函数定位）。
>
> **耗时：** 约 20 分钟。
>
> **学完能：**
> 1. 打开 `VLLM_LOGGING_LEVEL=DEBUG` 和 stat logger 实时看 step 状态
> 2. 在 Scheduler / KVCacheManager 里加 print，验证一个请求的每步决策
> 3. 用 `/metrics` 端点读 Prometheus 指标，把数字翻译成请求行为
> 4. 用 torch.profiler 看出 prefill / decode 各自的 kernel 时间分布

让 vLLM 把它每一步在干什么打印出来。看一次比读 10 遍文档管用。

---

## 1. 准备：打开调试日志

vLLM 用 Python 标准 logging。最快开 debug：

```bash
VLLM_LOGGING_LEVEL=DEBUG .venv/bin/python hello_vllm.py 2>&1 | tee vllm.log
```

或代码内：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

输出 5 万行不是夸张。先用最小例子（1 个 prompt、5 个 token）。

---

## 2. 打开 stat logger 看 step

vLLM 自带每 step 的统计输出。在创建 LLM 时加：

```python
llm = LLM(
    model="facebook/opt-125m",
    enforce_eager=True,
    disable_log_stats=False,    # 默认是 False，意为开启
)
```

或 server：`vllm serve ... --no-disable-log-stats`

你会看到周期性的日志：

```
Engine: Avg prompt throughput: ... tokens/s
        Avg generation throughput: ... tokens/s
        Running: 3 reqs   Waiting: 0 reqs
        GPU KV cache usage: 12.3%
        Prefix cache hit rate: 0.0
```

每一项对应代码里某个 metric。

---

## 3. 手动加 print：观察 Scheduler 决策

实验：在 `vllm/v1/core/sched/scheduler.py` 的 `schedule()` 末尾加：

```python
def schedule(self) -> SchedulerOutput:
    output = self._do_schedule(...)   # 假设主逻辑已抽出
    print(f"[SCHED] step n_running={len(self.running)} "
          f"n_waiting={len(self.waiting)} "
          f"total_tokens={output.total_num_scheduled_tokens} "
          f"new={[r.request_id for r in output.scheduled_new_reqs]} "
          f"preempted={list(output.preempted_req_ids)}")
    return output
```

跑：

```python
llm = LLM("facebook/opt-125m", enforce_eager=True)
llm.generate(["A"*100, "B"*200, "C"*50], SamplingParams(max_tokens=30))
```

观察 print，你能看出：

- 第一个 step 是 prefill（total_tokens 等于所有 prompt 长度之和或 chunk）
- 后续 step 是 decode（total_tokens ≈ n_running）
- 短请求先 finish，长请求继续

---

## 4. 观察 KV 分配

在 `vllm/v1/core/kv_cache_manager.py` 的 `allocate_slots` 末尾加：

```python
print(f"[KV] req={request.request_id} "
      f"new_tokens={num_new_tokens} "
      f"block_table_len={len(request.block_table)} "
      f"free_blocks={self.block_pool.get_num_free_blocks()}")
```

跑同样 demo，观察：

- 第一次 allocate 是大头（prompt 长度对应的所有 block）
- 后续 decode 大部分时候 block_table_len 不变（block 没满）
- 偶尔跨 block 边界时 block_table_len += 1

---

## 5. 观察 Prefix Caching 命中

```python
# 跑一遍：3 次完全相同的 prompt
prompts = ["System: be concise. User: hi"] * 3
```

如果 prefix caching 工作正常，从第 2 个请求开始：

- profile run 阶段：`block_table_len` 立即 = prompt 的 block 数
- 但 `allocate_slots` 内部应该命中了 cache，free_blocks 几乎没减

可以在 `block_pool.py` 的 hash 命中分支加 print：

```python
print(f"[CACHE HIT] hash={block_hash} block_id={existing_block.block_id}")
```

---

## 6. 用 Prometheus 看 metric

启动 server 时 vLLM 默认在 `/metrics` 暴露：

```bash
vllm serve facebook/opt-125m --port 8000 --enforce-eager
# 另开窗口
curl http://localhost:8000/metrics | grep -E 'vllm_(num_|prefix_|gpu_)' | head -20
```

关键指标（面试可引用）：

- `vllm:num_requests_waiting`
- `vllm:num_requests_running`
- `vllm:num_preemptions_total`
- `vllm:prefix_cache_hit_rate`
- `vllm:gpu_cache_usage_perc`
- `vllm:iteration_tokens_total` (histogram)
- `vllm:time_to_first_token_seconds` (histogram)
- `vllm:time_per_output_token_seconds` (histogram)

---

## 7. 用 torch profiler 看 GPU

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    llm.generate(prompts, params)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

你会看到每个 CUDA kernel 的时间。预期看到：

- prefill 阶段：matmul（QKV、MLP）占大头
- decode 阶段：attention 和访存类 kernel 占比上升

---

## 8. 一份实验作业清单

跑完下面 5 个，你对 vLLM 有"肉身记忆"：

1. **空 vs 满 batch**：单个 prompt vs 100 个并发 prompt，看 throughput / GPU util / KV usage 差距
2. **开关 prefix caching**：相同 prompt 重复 10 次，对比 TTFT
3. **不同 max-num-batched-tokens**：设 1024/4096/16384，看 TPOT 方差
4. **enforce-eager 开关**：开和关分别跑，看 startup time 和 decode latency
5. **加 fp8 KV**：`--kv-cache-dtype fp8`，看 `# GPU blocks` 是否翻倍

每个实验记一条结论。这就是你的"实战材料"，面试时可以拿出来讲。

---

## 9. 小结

- 三档观察工具：`VLLM_LOGGING_LEVEL=DEBUG`（全量日志） / stat logger（周期摘要） / `/metrics`（Prometheus 指标），从粗到细。
- 在 Scheduler 的 `schedule()` 末尾和 KVCacheManager 的 `allocate_slots` 加 print，是最快验证调度 / KV 分配假设的方式。
- prefix caching 命中时 `free_blocks` 几乎不减，这是判断 cache 真在工作最直接的信号。
- torch.profiler 能区分 prefill（matmul-bound）和 decode（attention + 访存-bound）的 kernel 占比，是后续 profiling 章节的入门工具。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. 同 prompt 跑 3 次（开 prefix caching），第 2 次 `free_blocks` 变化多少？为什么？**

假设 prompt 占 N 个 block。

**第 1 次**：

- prefix 命中 0 个（首次，cache 为空）
- alloc N 个 block prefill + 后续 decode 累积
- `free_blocks` 减少 N + decode 增量

**第 2 次（同 prompt）**：

- prefix 命中 **N-1 个 block**（最后一个 block 因为只用了部分 token，下次匹配时 hash 不同——见 [`02-core-concepts/04-prefix-caching.md`](../02-core-concepts/04-prefix-caching.md) §5.1）。注意：若是完美 block 对齐的 prompt，可能命中全部 N 个 block
- 命中的 block ref_cnt++ → **它们不从 free_queue 取出**（已被引用），所以 free_blocks 不减
- 只需要 alloc decode 累积的少量 block

→ **第 2 次的 `free_blocks` 比第 1 次少减约 N-1 个**。具体来说：

- 第 1 次结束 `free_blocks` = `initial - (N + decode_blocks)`
- 第 2 次 prefill 启动：`free_blocks` 减 1 或 2（最后一个非对齐 block + decode 第一个 block）

**判断 cache 真在工作**：第 2 次开始时如果 `free_blocks` 几乎不变，说明 prefix 命中了。如果还在减 N 个 block，要么命中率为 0（hash 没匹配），要么 cache 已被 evict。

---

**2. 5k 长 prompt + 50 token 短 prompt 同时进，前 3 step 的 `[SCHED]` 输出 + chunked prefill 怎么救短请求？**

假设 `max_num_batched_tokens = 2048`。

```
[SCHED] step 1:
  num_scheduled_tokens = {req_short: 50, req_long: 1998}
  total = 2048
  解释: 短请求一次性 prefill 完 50 token；长请求 prefill 第 1 个 chunk (1998 token)

[SCHED] step 2:
  num_scheduled_tokens = {req_short: 1, req_long: 2047}
  total = 2048
  解释: 短请求开始 decode 第 1 个新 token；长请求 prefill 第 2 个 chunk

[SCHED] step 3:
  num_scheduled_tokens = {req_short: 1, req_long: 955}
  total = 956
  解释: 短请求 decode 第 2 个 token；长请求剩余 prefill (5000-1998-2047=955) 跑完
        总 token < budget 因为长请求 prefill 已完
```

**chunked prefill 救短请求的原理**：

- 每步都给短请求 1 个 decode token slot（除了 step 1 它在 prefill）
- 长请求被切成 chunk（1998, 2047, 955），不是 5000 一次性
- 每步 GPU 时长稳定约 2048 token forward
- 短请求每步都能 decode → TPOT 平稳

**反例（不开 chunked prefill）**：

```
step 1: {req_short: 50, req_long: 5000} → total 5050, 单步几百 ms
  这一步内 req_long 5000 token prefill 全跑
  req_short 也"跟着算了 50 个 prefill token + 0 decode"
step 2: 全员 decode
```

短请求第一个 token 要等长请求 5000 token prefill 跑完——TTFT 飙到几百 ms。chunked prefill 把这压到 ~50 ms。

---

**3. `num_preemptions_total` 与 `gpu_cache_usage_perc` 因果关系。**

**因果链**：

```
请求并发 ↑
  → KV 占用 ↑                    （vllm:gpu_cache_usage_perc 升高）
    → 接近 100% 时 alloc 失败
      → 触发 preempt              （vllm:num_preemptions_total 累加）
        → 被踢请求 KV 释放
          → KV 占用回落
```

**典型曲线**：

```
gpu_cache_usage:    ____......^^^^____...^^^^____
                                ↑           ↑
preemptions:        ____________╱___________╱___
                              一次 spike   再一次
```

`preemptions` 几乎只在 `cache_usage > 0.9` 时增长——存在强因果。如果 `preemptions` 涨而 `cache_usage` 不高，多半是 PRIORITY 模式下优先级反演（高优请求挤低优）。

**告警规则**：

```promql
# 容量预警
vllm:gpu_cache_usage_perc > 0.9 for 5m → warning

# 抢占速率
rate(vllm:num_preemptions_total[5m]) > 0.5 → warning

# 两者关联表明真饿 KV
gpu_cache_usage > 0.95 AND preempt_rate > 1 → critical (扩容)
```

详见 [`08-production-deployment/08-monitoring-cookbook.md`](../08-production-deployment/08-monitoring-cookbook.md)。

---

**4. torch.profiler prefill vs decode 排前 3 kernel + 差异。**

**Prefill 阶段排前 3**（典型）：

1. `flash_attn_varlen_func` 或 `_flash_attn_v3_fwd_kernel` — attention 主 kernel
2. `gemm_kernel`（cuBLAS / cutlass）— Q/K/V/O 投影 + MLP 矩阵乘
3. `silu_and_mul_kernel`（fused）— SwiGLU MLP 激活

特点：**compute-bound**——大 batch × 长 seq，算力打满。kernel 时长正比 token 数。

**Decode 阶段排前 3**（典型）：

1. `paged_attention_v2_kernel` 或 `flash_attn_with_kvcache` — 单 query attention，纯 memory-bound
2. `gemm_kernel`（GEMM）— Linear 投影，每步 batch_size × hidden_size 的矩阵乘
3. `_rms_norm_kernel` 或 `reshape_and_cache_kernel` — LayerNorm + KV 写入 cache

特点：**memory-bound**——每 token 1 个 query，算力闲着，主要读 weight 和 KV。kernel 时长几乎与 batch_size 无关（只要还在 memory-bound 区）。

**关键差异**：

| 维度 | prefill | decode |
| --- | --- | --- |
| attention kernel | flash_attn_varlen (大 Q × 长 K) | paged_attention (Q=1 × 长 K) |
| 算力利用 | 80%+ (compute-bound) | 30-50% (memory-bound) |
| HBM 带宽 | 中 | 接近峰值 |
| GEMM 占比 | 50%+ | 30-40% |
| kernel 时长 / token | 0.05 ms (Llama-7B) | 0.2-0.5 ms |

→ 这就是为什么 vLLM 设计**continuous batching + chunked prefill 把两者混跑**——decode 用算力空隙，prefill 用 GPU 满载，互补。

## 下一步

- 下一节：[`07-hands-on/03-mini-experiments.md`](03-mini-experiments.md)（把"我看到了 X"升级成"我测量出 Y"）
- 想看源码：`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`
- 想动手：本节末尾的"实验作业清单"5 条都做一遍
- 想从生产视角理解：[`08-production-deployment/05-slo-and-observability.md`](../08-production-deployment/05-slo-and-observability.md)（同样的 metric 在 Grafana 里怎么搭面板）
