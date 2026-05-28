# 03. 5 个 Mini 实验：把直觉变成数字

> **谁该读这一篇？** 想在面试和技术分享里讲"我跑过 X，看到 Y"而不是只复述论文的工程师；带新人时希望给出可复现验证清单的 mentor。
>
> **前置阅读：** [`07-hands-on/01-setup.md`](01-setup.md)（环境已装好），[`07-hands-on/02-trace-a-request.md`](02-trace-a-request.md)（会用 stat logger 和 metric），[`02-core-concepts/04-prefix-caching.md`](../02-core-concepts/04-prefix-caching.md) + [`02-core-concepts/05-chunked-prefill.md`](../02-core-concepts/05-chunked-prefill.md)（理解实验背后概念）。
>
> **耗时：** 约 30 分钟阅读 + 1-2 小时跑完 5 个实验。
>
> **学完能：**
> 1. 复现 prefix caching 对 TTFT 的提升（量化数字）
> 2. 验证 `max-num-batched-tokens` 大小如何影响 TPOT 方差
> 3. 看到 FP8 KV cache 让 num_blocks 接近翻倍
> 4. 故意制造 KV 压力观察 Scheduler 的 preempt 行为
> 5. 测量 ngram 投机解码在不同 workload 下的吞吐收益

读再多笔记不如自己测一次。下面 5 个实验都基于 `facebook/opt-125m` 或 `Qwen2.5-0.5B`（小模型省 GPU），但结论可以推广到大模型。跑完后，每个实验记一段 200 字以内的"我观察到 X，所以 Y"。这就是面试可拿出来讲的"实战证据"。

---

## 实验 1：Prefix Caching 对 TTFT 的真实影响

### 目标
量化"重复 system prompt"在 prefix cache 开/关下的差距。

### 脚本

```python
# experiment1_prefix_cache.py
import time
from vllm import LLM, SamplingParams

SYSTEM = "你是一个助手。" * 200      # 大约 1k tokens
USER_QUERIES = [f"用户问题 {i}" for i in range(10)]

def run(enable_prefix_caching: bool):
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        enforce_eager=True,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.5,
    )
    params = SamplingParams(max_tokens=50, temperature=0)
    prompts = [SYSTEM + q for q in USER_QUERIES]

    # 第一次跑：填 cache
    llm.generate(prompts[:1], params)

    # 计时：后续 9 次
    t0 = time.perf_counter()
    llm.generate(prompts[1:], params)
    dt = time.perf_counter() - t0
    print(f"prefix_caching={enable_prefix_caching}: 9 requests in {dt*1000:.0f}ms")

run(enable_prefix_caching=False)
run(enable_prefix_caching=True)
```

### 预期结果
开启后第 2-10 次请求 TTFT 显著下降（70-90%）。

### 自测题
- 如果改成 `temperature=0.7`（每次输出不同），prefix cache 还能命中吗？
- 如果把 SYSTEM 改成 `"你是一个助手。" * 200 + str(time.time())`（每次微小不同），命中率会怎样？

---

## 实验 2：max-num-batched-tokens 对 TPOT 抖动的影响

### 目标
观察 token budget 大小对单步延迟方差的影响。

### 脚本

```bash
# 把 max-num-batched-tokens 设小（不易混入长 prefill）
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --max-num-batched-tokens 1024 \
    --port 8001 &

# 设大（一个长 prefill 可能占满整步）
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --max-num-batched-tokens 16384 \
    --port 8002 &
```

```python
# experiment2_tpot.py：同时发短请求 + 一个长请求
import asyncio
import time
import httpx

async def short_req(client, port):
    t0 = time.perf_counter()
    await client.post(f"http://localhost:{port}/v1/completions", json={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Hi, how are you?",
        "max_tokens": 30,
        "temperature": 0,
    })
    return time.perf_counter() - t0

async def long_req(client, port):
    await client.post(f"http://localhost:{port}/v1/completions", json={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "重复内容 " * 4000,    # ~8k tokens
        "max_tokens": 10,
        "temperature": 0,
    })

async def run(port):
    async with httpx.AsyncClient(timeout=60) as client:
        # 同时发 5 个短请求 + 1 个长请求
        tasks = [short_req(client, port) for _ in range(5)] + [long_req(client, port)]
        results = await asyncio.gather(*tasks)
    print(f"port {port}: short req latencies = {[f'{r*1000:.0f}ms' for r in results[:5]]}")

asyncio.run(run(8001))   # 小 budget
asyncio.run(run(8002))   # 大 budget
```

### 预期结果
- port 8001（小 budget）：短请求延迟均匀
- port 8002（大 budget）：长请求那一步把短请求拖慢 100ms+

### 自测题
- 哪种配置更适合 chatbot？哪种更适合批量推理？

---

## 实验 3：FP8 KV Cache 真的让 num_blocks 翻倍吗？

### 目标
验证启动日志里的 "# GPU blocks: NNNN" 是否符合预期。

### 脚本

```bash
# baseline
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enforce-eager \
    --gpu-memory-utilization 0.5 2>&1 | grep -E "GPU blocks|KV cache" &

sleep 60 && pkill -f "vllm serve" && sleep 5

# fp8 KV
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --kv-cache-dtype fp8 2>&1 | grep -E "GPU blocks|KV cache"
```

### 预期结果
fp8 模式 num_blocks **约为 baseline 的 2 倍**。

不严格 2×，因为还有其他显存（activation buffer、CUDA graph workspace 等）固定。

### 自测题
- 显存利用率从 0.5 改成 0.9，num_blocks 是否线性放大？

---

## 实验 4：观察 Scheduler 的 preempt 行为

### 目标
故意制造 KV 压力，观察 preempt 发生。

### 脚本

```python
# experiment4_preempt.py
import asyncio
import httpx

# 启动 server 时强制小 KV：
# vllm serve Qwen/Qwen2.5-0.5B-Instruct \
#     --enforce-eager --gpu-memory-utilization 0.2 \
#     --max-num-seqs 256

async def long_req(client, i):
    return await client.post("http://localhost:8000/v1/completions", json={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": f"请求 {i} " + "上下文 " * 500,
        "max_tokens": 500,
        "temperature": 0.7,
    })

async def run():
    async with httpx.AsyncClient(timeout=120) as client:
        # 同时发 200 个长请求
        tasks = [long_req(client, i) for i in range(200)]
        await asyncio.gather(*tasks)

asyncio.run(run())
```

同时另开终端：
```bash
watch -n 0.5 'curl -s localhost:8000/metrics | grep -E "vllm:num_preemptions|vllm:num_running|vllm:num_waiting"'
```

### 预期结果
- `vllm:num_preemptions_total` 持续上升
- `num_running` 上下抖动（被 preempt 又 admit）

### 自测题
- 如果改成 `--scheduling-policy priority` 并给一半请求高 priority，会有什么变化？

---

## 实验 5：投机解码的接受率与吞吐

### 目标
量化投机解码在 chat workload 下的实际收益。

### 脚本

```bash
# baseline
python benchmarks/benchmark_throughput.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num-prompts 100 \
    --input-len 256 \
    --output-len 256 \
    --enforce-eager

# 用 ngram spec
python benchmarks/benchmark_throughput.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num-prompts 100 \
    --input-len 256 \
    --output-len 256 \
    --enforce-eager \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4}'
```

### 预期结果
- ngram 在 prompt 重复多的场景：吞吐 ×1.3-1.5
- 普通 chat：收益较小（acceptance rate 低）

### 自测题
- 如果你换成 `{"method": "eagle", ...}` + 一个 EAGLE 模型，效果怎样？
- 为什么 batch_size 越大，投机解码收益越小？

---

## 实验报告模板

每个实验跑完，把下面 3 句话填好放进笔记里：

```
实验 X：[一句话目标]
观察：[最关键的 1-2 个数字]
结论：[这告诉我们关于 vLLM 的什么]
踩坑：[过程中遇到的意外，怎么解决的]
```

例如：

> **实验 1：Prefix Caching 对 TTFT 的影响**
> 观察：开启后第 2-10 次请求平均 TTFT 从 320ms 降到 45ms。
> 结论：chatbot 场景 system prompt 占 80%+ 计算，prefix caching 是必开。
> 踩坑：第一次没看到效果，发现是 `enable_prefix_caching` 写错；测试时还要排除 model load 时间。

这些数据是面试时最大的差异化。**普通候选人讲概念，你讲数字**。

---

## 进阶实验（如果有大卡）

1. **不同量化方法吞吐对比**：FP16 vs FP8 vs AWQ-INT4 vs GPTQ-INT4 on Llama-2-7B
2. **TP scaling**：TP=1/2/4 跑同一模型，看延迟与吞吐曲线
3. **Async scheduler 对 CPU overhead 的影响**：开关 async，看 CPU profile
4. **Disaggregated prefill 模拟**：在同一台机器上跑两个实例 + NIXL 模拟器

每个进阶实验都能产出一篇技术博客级别的内容。

---

## 小结

- 5 个实验分别验证了 prefix caching、token budget、FP8 KV、preempt、投机解码这 5 个 vLLM 核心机制。
- 实验脚本都用小模型（OPT-125m / Qwen-0.5B）就能跑，但结论对大模型一样适用。
- "目标 / 预期 / 自测题"三段式让每个实验都有"可复现 + 可推理"的双重价值。
- 实验报告模板（目标 / 观察 / 结论 / 踩坑）是面试自我介绍中最有效的素材结构。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. 实验 1 结论：prefix caching 在 chatbot 场景 TTFT 降低 X%。**

典型结果（system prompt ≈ 500 token，用户 query ≈ 50 token，无 cache vs 有 cache）：

> "在 system prompt = 500 token + 用户 query = 50 token 的 chatbot workload 下，开启 prefix caching 让**首次同 prompt 之后的请求 TTFT 从 ~180ms 降到 ~30ms，降低约 83%**。降幅与 (cached_tokens / total_prompt_tokens) 比值正相关——cached 占比越高，降幅越接近 95%。"

**详细数字依赖**：
- system prompt 越长 → 降幅越大（500-token prompt 降 83%，2000-token prompt 降 95%+）
- 命中率（连续请求间是否同 prefix）决定平均收益
- 第一次请求 TTFT 不变（cache 还没建立）

**面试可引申**：这是 RAG / chatbot 场景的主要优化路径，比 quantization / spec decode 收益还大。

---

**2. 实验 2 长请求让短请求多等多少 ms + `max-num-batched-tokens` 建议。**

典型实验 2 设置：长请求 prompt=8192，短请求 prompt=50，同时进。

**不开 chunked prefill（或 budget 极大）**：
- 长请求一次 forward 跑 8192 token → ~250ms
- 短请求 TTFT = 排队 + 长请求 prefill 时长 ≈ **250 ms**

**开 chunked prefill, `max-num-batched-tokens=2048`**：
- 长请求被切成 4 chunk，每步 2048 token → ~60ms / step
- 短请求 step 1 就能并行 prefill（50 + 1998 = 2048 内）→ TTFT ≈ **60 ms**

→ **短请求多等的 ms = 250 - 60 = 190 ms**（4× 减少）。

**`max-num-batched-tokens` 取值建议**（参考表）：
- 4096-8192：通用 chatbot，平衡 TTFT 和 throughput
- 2048：TPOT 敏感（code completion），切更小 chunk
- 16384+：离线 batch，不在意 TPOT 抖动
- < 1024：极端 TPOT 要求（agent 多轮交互），但每 chunk 太小 schedule overhead 占比上升

**调参逻辑**：先按业务 SLO 选初值 → 跑 benchmark → 看 TPOT p99 是否达标 → 不达标减半 / 翻倍

---

**3. 实验 3 baseline vs fp8 num_blocks 比值是否严格 2×？什么是固定项拉低比值？**

**答**：**不严格 2×**，典型实测 1.6-1.8×。

理由：单 block 字节数严格减半（K/V 各 1 byte vs 2 byte），但**可用 KV 显存不是全部显存**：

```
total_hbm = 80 GB (H100)
   minus 模型权重         (~16 GB Llama-3-8B BF16)
   minus CUDA buffer       (~2 GB)
   minus activation 预算   (~4-6 GB, 与 max_num_batched_tokens 相关)
   minus CUDA Graph buffer (~1-2 GB)
   minus profiling 留 5%   (~4 GB)
= KV 可用                  (~50 GB)

→ KV blocks (BF16) = 50 / single_block_bf16
→ KV blocks (FP8)  = 50 / single_block_fp8 = 50 / (single_block_bf16 / 2) = 100 / single_block_bf16

ratio = 100 / 50 = 2.0  ← 理论值
```

**但实际**：FP8 KV 启用后 attention kernel 需要额外 `k_scale, v_scale` per layer → ~MB 级开销，影响微小。

**真正拉低比值的因素**：
- 模型权重始终占固定显存
- profile_run 用大 batch 测峰值，FP8 时激活仍 BF16/FP16 → 激活预算不变
- CUDA Graph capture sizes 与 dtype 无关

→ **比值近 2× 但不严格 2×**，这是预期行为，不是 bug。

---

**4. 实验 4 看 `num_preemptions_total` 增长率，KV 再降一半会怎样？**

实验 4 通常用 `--gpu-memory-utilization 0.5` 故意压缩 KV 空间观察 preempt。

**当前 preempt 速率**例：约 1-5 次/s（在并发饱和时）。

**KV 再降一半**：
- num_blocks 减半 → 同时能装的并发请求减半
- 同样并发流量下：
  - `kv_cache_usage_perc` 持续 100%
  - `num_preemptions_total` 增长率 **指数级上升**（不是 2×）
  - 因为每次 preempt 释放出来的空间立刻被下一个排队请求占满，触发下一次 preempt
- **症状**：throughput 崩溃，TPOT 抖动剧烈（10×+），出现"刚 admit 又被踢"的振荡

**临界条件**：
- 单请求平均 KV 占用 = `K KB / 请求`
- 可用 KV = `M KB`
- 稳定不 preempt 的并发上限 ≈ `M / K`
- 一旦 offered concurrency > 这个上限，preempt 必然发生
- KV 减半 = 上限减半 = 提前 2× 触发 preempt 风暴

**生产经验**：留至少 30% KV 安全边际，宁可少几个并发也不要在 KV 边缘走。

---

**5. 实验 5 ngram spec decode 在什么 workload 收益最高？为什么大 batch 收益下降？**

**ngram 收益最高的 workload**：
- **高重复性文本**：code completion（变量名、API、boilerplate 反复出现）
- **结构化输出**：JSON / XML / SQL（语法 token 高度可预测）
- **长输出 + 模板化**：写报告、邮件模板、文档生成
- **多轮对话同 system prompt**：前缀重复带来 ngram 库丰富

**ngram 接受率典型范围**：
- code/JSON：30-50%
- 通用对话：10-20%
- 创意写作（高随机性）：5-10%

**为什么 batch_size 大时收益下降？**

batch_size 与 GPU 状态的关系：

| batch_size | GPU 状态 | spec decode 收益 |
| --- | --- | --- |
| 1-4 | memory-bound（算力闲）| **高** —— 多算 N 个 token 几乎免费 |
| 8-32 | 接近 compute-bound | 中 |
| 64+ | compute-bound（算力满）| **低甚至负** —— 每个 token 都要算力 |

**算力账**：
- 小 batch：target 跑 1 token 的算力 = 跑 5 token 的算力（GPU 闲着，多算免费）
- 大 batch：target 跑 1 个新 token 的实际算力 ≈ batch_size 个 token 的工作量。多算 N 个 token 就是真实地多花 N × batch_size 的算力
- spec decode 加速比 ≈ `1 + acceptance_rate × (N - 1)` × (1 - overhead)
- 大 batch 下 overhead 上升 + 算力成本上升 → 实际加速比可能 < 1

**生产建议**：
- 用 `vllm:num_requests_running` 当 batch_size 代理
- batch_size > 32 时自动关 spec decode（动态开关）
- 或换成 MTP（DeepSeek-V3 内置）—— 几乎零额外开销，大 batch 也能开

加分点：EAGLE 在小 batch 下加速比 2-3×，大 batch 下可能反而拖慢 20%。决策权在监控数据，不在文档建议。

## 下一步

- 下一节：[`07-hands-on/04-profiling-and-debugging.md`](04-profiling-and-debugging.md)（从"我能跑出数字"升级成"我能定位 kernel-level 异常"）
- 想看源码：`benchmarks/benchmark_throughput.py`、`benchmarks/benchmark_serving.py`、`vllm/v1/core/sched/scheduler.py`
- 想动手：把每个实验改成"对比 2 个 vLLM 版本"——这能直接产出社区 PR 的回归测试材料
- 想从生产视角理解：[`08-production-deployment/05-slo-and-observability.md`](../08-production-deployment/05-slo-and-observability.md)（同样的指标在生产怎么报警）
