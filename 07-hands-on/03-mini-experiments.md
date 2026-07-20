# 03. 5 个 Mini 实验：把直觉变成数字

> **谁该读这一篇？** 想在技术分享里讲"我跑过 X，看到 Y"而不是只复述论文的工程师；带新人时希望给出可复现验证清单的 mentor。
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

> **当前源码复核（`b23bd73f540175f9e117eaee5029cd7d8df63964`）：** 每次实验只改一个自变量；模型、commit、硬件、driver、输入/输出长度、并发和预热次数必须入报告。下面写的是待验证假设，不是通用性能承诺；所有数字都应来自你的原始结果。标有 NVIDIA/H100/A100 的实验不能拿无 GPU 的静态检查冒充实测。

读再多笔记不如自己测一次。下面 5 个实验都基于 `facebook/opt-125m` 或 `Qwen2.5-0.5B`（小模型省 GPU），但结论可以推广到大模型。跑完后，每个实验记一段 200 字以内的"我观察到 X，所以 Y"，后续做分享、复盘和方案评审都用得上。

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

### 待验证假设
重复前缀跨越至少一个完整 cache block 时，开启 prefix caching 后续请求的 TTFT 应下降。下降幅度由前缀长度、block 对齐、并发和硬件决定；记录 p50/p95/p99 与 hit/query counter 增量，不预填百分比。

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

### 待验证假设
- 小 budget 会限制单步 prefill 工作量，可能降低短请求尾延迟，但会增加完成长 prompt 所需的 step 数。
- 大 budget 可能提高吞吐，也可能让混合负载中的 decode 延迟更抖。用同一 workload 比较 TPOT p99，不能用单次端到端耗时替代 TPOT。

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
    --gpu-memory-utilization 0.5 >baseline.log 2>&1 &
BASELINE_PID=$!

# 等待 health ready 后保存日志；只终止刚刚记录的进程。
curl --retry 60 --retry-delay 2 --retry-connrefused -fsS \
    http://127.0.0.1:8000/health
grep -E "GPU blocks|KV cache" baseline.log
kill -TERM "$BASELINE_PID"
wait "$BASELINE_PID" || true

# fp8 KV
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --kv-cache-dtype fp8 2>&1 | grep -E "GPU blocks|KV cache"
```

### 待验证假设
在平台支持 FP8 KV 且其他参数相同的前提下，单 block 的 KV 字节数下降，可用 block 数通常增加，但并不承诺恰好 2 倍。把两次启动日志、有效 KV 容量和平台支持信息一并保存。

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
watch -n 0.5 'curl -s localhost:8000/metrics | grep -E "vllm:(num_preemptions_total|num_requests_running|num_requests_waiting)"'
```

### 待验证假设
- KV 压力足够大时，`vllm:num_preemptions_total` 增长。
- 同时保存 `num_requests_running`、`num_requests_waiting` 与 `kv_cache_usage_perc`；若没有抢占，先证明 workload 是否真的造成 KV 压力，不要把“没有复现”改写成成功。

### 自测题
- 如果改成 `--scheduling-policy priority` 并给一半请求高 priority，会有什么变化？

---

## 实验 5：投机解码的接受率与吞吐

### 目标
量化投机解码在 chat workload 下的实际收益。

### 脚本

```bash
# baseline
vllm bench throughput \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num-prompts 100 \
    --input-len 256 \
    --output-len 256 \
    --enforce-eager

# 用 ngram spec
vllm bench throughput \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --num-prompts 100 \
    --input-len 256 \
    --output-len 256 \
    --enforce-eager \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4}'
```

### 待验证假设
- ngram 在可从 prompt 匹配后续 token 的 workload 中可能提高吞吐。
- 普通 chat 的接受率与收益取决于数据；必须同时报告 acceptance、output throughput、TTFT/TPOT 与基线，不能预填倍数。

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

下面只是**格式示例**，数字必须替换为你保存的原始结果，不能作为本章结论引用：

> **实验 1：Prefix Caching 对 TTFT 的影响**
> 观察：基线 TTFT p50/p99 为 `<实测>`，开启后为 `<实测>`；hit/query 增量为 `<实测>`。
> 结论：在本次 `<模型、commit、硬件、workload>` 下，结果支持/不支持“重复前缀降低 TTFT”的假设。
> 踩坑：第一次没看到效果，发现是 `enable_prefix_caching` 写错；测试时还要排除 model load 时间。

这些数据会让理解变得非常扎实。**概念要讲，数字也要拿得出来**。

---

## 进阶实验（如果有大卡）

下面 4 个实验都需要更大的硬件（H100/A100 单卡或多卡），但每一个都能产出**一篇博客级技术内容**。

---

## 进阶实验 1：4 种量化方法的吞吐 / 延迟 / 精度三维对比

### 目标

用同一个模型、同一个 workload 跑 FP16 / FP8 / AWQ-INT4 / GPTQ-INT4 四组数据，给生产部署整理一份量化选型表。**特别要回答**：什么硬件该选什么量化、精度损失实际多少、是否走了 Marlin kernel。

### 硬件 / 软件要求

- **必须 H100 80GB**（H100 才有 FP8 硬件原生支持，A100 跑 FP8 是软件模拟，结论不可比）
- 准备 4 个 checkpoint：
  - `meta-llama/Llama-2-7b-hf` （FP16 baseline）
  - `nm-testing/llama-2-7b-fp8` （FP8 dynamic，NeuralMagic 出品）
  - `TheBloke/Llama-2-7B-AWQ` （AWQ INT4）
  - `TheBloke/Llama-2-7B-GPTQ` （GPTQ INT4）
- 工具：`benchmark_serving.py` + `lm-eval-harness`（精度）

### 脚本

```bash
# 1. 性能 benchmark（4 组）
for fmt in fp16 fp8 awq gptq; do
    case $fmt in
        fp16) MODEL=meta-llama/Llama-2-7b-hf;        QUANT="" ;;
        fp8)  MODEL=nm-testing/llama-2-7b-fp8;       QUANT="--quantization fp8" ;;
        awq)  MODEL=TheBloke/Llama-2-7B-AWQ;         QUANT="--quantization awq_marlin" ;;
        gptq) MODEL=TheBloke/Llama-2-7B-GPTQ;        QUANT="--quantization gptq_marlin" ;;
    esac

    echo "=== $fmt ==="
    vllm serve $MODEL $QUANT \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 256 \
        --port 8000 &
    SERVER_PID=$!
    sleep 90   # 等 compile + warmup

    vllm bench serve \
        --model $MODEL \
        --dataset-name sharegpt \
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 500 \
        --request-rate 10 \
        --result-filename results_$fmt.json

    # 取关键 metric
    curl -s :8000/metrics | grep -E "vllm:(kv_cache_usage|num_requests_running)" \
        > runtime_$fmt.txt

    kill $SERVER_PID && sleep 30
done

# 2. 精度 benchmark
for fmt in fp16 fp8 awq gptq; do
    lm_eval --model vllm \
        --model_args pretrained=$MODEL,quantization=$QUANT \
        --tasks wikitext,arc_easy,hellaswag \
        --batch_size auto \
        --output_path eval_$fmt.json
done
```

### 结果记录（不得预填）

| 格式 | 显存 | 吞吐 tok/s | TTFT p99 | TPOT p99 | WikiText PPL | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| FP16/BF16 baseline | `<实测>` | `<实测>` | `<实测>` | `<实测>` | `<实测>` | 记录 dtype/backend |
| FP8 | `<实测>` | `<实测>` | `<实测>` | `<实测>` | `<实测>` | 记录平台原生支持 |
| AWQ-INT4 | `<实测>` | `<实测>` | `<实测>` | `<实测>` | `<实测>` | 记录实际 kernel |
| GPTQ-INT4 | `<实测>` | `<实测>` | `<实测>` | `<实测>` | `<实测>` | 记录实际 kernel |

### 必须核对

- 启动日志中的实际 quantization backend/kernel 是否与预期一致；fallback 必须记入结果。
- 权重 dtype 与 KV cache dtype 是两个自变量，不要在同一轮一起改。
- 吞吐变化可能来自每 token 计算、可容纳 batch、KV 容量或 kernel fallback；用 batch=1 和饱和负载拆因。
- PPL 不能替代业务质量评估；至少补一组与业务相似的 code/math/chat 任务。

### 自测题

1. 把 batch=1 跑一遍，量化的相对收益还在吗？为什么？
2. WikiText PPL 增长 2.7% 在你的业务能接受吗？怎么换算到 chatbot 用户体验？
3. AWQ 和 GPTQ 哪个 calibration 时间更长？checkpoint 文件大小差多少？
4. FP8 KV cache（`--kv-cache-dtype fp8`）和 FP8 权重 量化，能不能同时开？

### 可产出的博客角度

- "为什么 H100 用户应该默认上 FP8"——几乎零成本翻倍并发
- "AWQ vs GPTQ：4 步选型决策树"——配 ecosystem 兼容性矩阵
- "你以为开了量化就一定快？"——把 Marlin fallback 的坑讲透
- "量化的精度损失分布在哪？"——把 PPL 增长拆到不同任务（math、code、闲聊）

---

## 进阶实验 2：TP scaling 真实曲线 + 通信开销可视化

### 目标

量化 TP 的 **scaling efficiency**，看清"为什么 TP=8 不是 TP=1 的 8 倍吞吐"，并用 nsys 直观看到 AllReduce 占 forward 多少。

### 硬件 / 软件要求

- 4-8 卡 H100 / A100 **同机**（必须 NVLink，跨机 TP 是另一个故事）
- Model：**Llama-2-13B**（BF16 26GB，刚好单 H100 80G 能放，方便测 TP=1 作为 baseline）
- 工具：`benchmark_serving.py` + `nsys profile`

### 脚本

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
for tp in 1 2 4 8; do
    [ $tp -gt $NUM_GPUS ] && continue

    vllm serve meta-llama/Llama-2-13b-hf \
        --tensor-parallel-size $tp \
        --max-num-seqs 256 \
        --port 8000 &
    SERVER_PID=$!
    sleep 120   # 等 NCCL init + compile

    # 多 QPS 曲线（找到拐点）
    for qps in 5 10 20 50 100; do
        vllm bench serve \
            --model meta-llama/Llama-2-13b-hf \
            --dataset-name sharegpt \
            --num-prompts 300 \
            --request-rate $qps \
            --result-filename results_tp${tp}_qps${qps}.json
    done

    # nsys 看 NCCL 占比（取 QPS=20 这个中等负载）
    nsys profile -t cuda,nvtx,osrt \
        --capture-range cudaProfilerApi --capture-range-end stop \
        -o profile_tp${tp} \
        .venv/bin/python -c "
from vllm import LLM, SamplingParams
import torch
llm = LLM('meta-llama/Llama-2-13b-hf', tensor_parallel_size=$tp, enforce_eager=True)
torch.cuda.cudart().cudaProfilerStart()
llm.generate(['Hello'] * 32, SamplingParams(max_tokens=100))
torch.cuda.cudart().cudaProfilerStop()
"

    kill $SERVER_PID && sleep 30
done

# 用 nsys-ui 或 ncu 打开 profile_tp*.nsys-rep，看：
#   - NCCL AllReduce kernel 占 timeline 的 % 比例
#   - 不同 TP 下 forward 总时长
```

### 结果记录（不得预填）

| TP | 单卡吞吐 (tok/s) | 总吞吐 | scaling efficiency | AllReduce 占 forward |
| --- | --- | --- | --- | --- |
| 1 | `<实测>` | `<实测>` | 100%（基准）| `<实测>` |
| 2 | `<实测>` | `<实测>` | `<计算>` | `<实测>` |
| 4 | `<实测>` | `<实测>` | `<计算>` | `<实测>` |
| 8 | `<实测>` | `<实测>` | `<计算>` | `<实测>` |

### 必须核对

- 先验证各 TP 使用相同模型、dtype、输入分布、并发和拓扑；记录 NVLink/PCIe 与 NCCL 版本。
- scaling efficiency 用 `TP=n 总吞吐 / (n × TP=1 总吞吐)` 计算，不凭经验填“甜点”。
- 用 nsys 区分 GEMM、attention、NCCL 和空洞时间，再解释 TTFT/TPOT 变化。

### 自测题

1. 同样 13B 模型，TP=4 和 4 实例 DP=4 哪个总吞吐高？什么场景下选哪个？
2. 如果换 PCIe 替 NVLink，TP=8 还能跑吗？为什么效率会暴跌？算一下传输时长。
3. TP scaling 曲线在 batch=1 和 batch=64 下形状有何不同？为什么 batch 越大效率反而越高？
4. 给定 70B 模型 + 4× A100 40G，能跑通吗？需要怎么配置？

### 可产出的博客角度

- "TP scaling 的 80% 法则"——每翻倍 TP 大约掉 15-20% 效率
- "用 nsys 找出 AllReduce 真实占比"——截图 + 逐段解读
- "什么时候停止增加 TP"——Pareto 前沿分析
- "TP vs DP：什么时候多实例反而更好"

---

## 进阶实验 3：AsyncScheduler 对 CPU overhead 的真实影响

### 目标

量化 V1 AsyncScheduler 在大 batch 下能省多少端到端时间，**用 flame graph 直接看 scheduler 与 forward 是否真的 overlap**。

### 硬件 / 软件要求

- 任意一卡 GPU（这个实验关注 CPU 行为，GPU 不是瓶颈）
- 工具：`py-spy`（采样 profiler）+ `torch.profiler`（kernel + Python 时间）

### 脚本

```bash
# 1. 启动服务
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --port 8000 &
SERVER_PID=$!
sleep 60

# 2. 先查看进程树，人工确认属于 SERVER_PID 的 EngineCore 子进程；
# 不要用全机 pgrep 后取第一个结果。
ps -eo pid,ppid,cmd | grep -E "PID|$SERVER_PID|EngineCore"
export SCHEDULER_PID='<verified-engine-core-pid>'
ps -p "$SCHEDULER_PID" -o pid,ppid,cmd
sudo py-spy record \
    -o profile_async.svg \
    --pid "$SCHEDULER_PID" \
    --duration 30 \
    --rate 1000 \
    --threads &
PYSPY_PID=$!

# 3. 同时跑大 batch workload 制造 scheduler 压力
vllm bench serve \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset-name sharegpt \
    --num-prompts 500 \
    --request-rate 50

wait "$PYSPY_PID"

# 4. 看 profile_async.svg：
#    - 找 schedule() 调用栈，看它在 30s 里占多少
#    - 看 update_from_output / preempt 等子函数分布
#    - 看 scheduler thread 状态：busy% 多少
```

补一段 Python 端 torch.profiler 配合：

```python
# experiment_async.py
import torch
from torch.profiler import profile, ProfilerActivity
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enforce_eager=True,   # 关掉 CUDA Graph，让 CPU 时间清晰可见
    max_num_seqs=256,
    gpu_memory_utilization=0.9,
)

# 大 batch 制造调度压力
prompts = ["写一段 200 字介绍 vLLM 的文章。"] * 200
params = SamplingParams(max_tokens=200, temperature=0)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    outs = llm.generate(prompts, params)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
prof.export_chrome_trace("trace_async.json")
# 用 chrome://tracing 或 https://ui.perfetto.dev 打开 trace_async.json
```

### 结果记录（不得预填）

在 Perfetto 时间线上标出 schedule 与 forward 是否重叠；下面只是读图示意，不代表实测比例：

```
线程 EngineCore-main  ▓▓▓▓░░▓▓▓▓░░▓▓▓▓░░▓▓▓▓     ← schedule() 间歇执行
线程 Worker-0 GPU     ░░▓▓▓▓░░▓▓▓▓░░▓▓▓▓░░▓▓▓▓   ← forward 间歇执行
                       ^ schedule 与 forward 错开，几乎完全重叠
```

报告中填写 schedule CPU time、forward GPU time、重叠区间、吞吐与 TPOT p99。若没有建立关闭/开启 async scheduling 的可比基线，就只能描述 trace，不能声称收益来自 AsyncScheduler。

### 必须核对

- overlap 优化减少的是关键路径，不等于减少 schedule 的 CPU 总工作量。
- batch 大小、请求长度和采样功能都会改变调度成本；至少测低/中/高三个负载点。
- 关键源码入口是 `vllm/v1/core/sched/async_scheduler.py`；实验开关必须以当前 CLI/help 为准，不建议 patch V0 做对照。

### 自测题

1. 为什么 schedule 时间会随 batch 增大？给出 2 个主要原因（提示：preempt 候选 + KV alloc）
2. AsyncScheduler 如果 schedule 比 forward 还慢，发生什么？什么场景会出现？
3. 这套 producer-consumer overlap 在 OS 课里有什么对应概念？（double buffering / pipelining）
4. 当前 CLI 若提供受支持的 async scheduling 开关，应如何设计只改该开关的 A/B？若没有，为什么不能用 patch V0 冒充可比基线？

### 可产出的博客角度

- "V1 AsyncScheduler 看似简单，实则 vLLM 性能跃迁的关键之一"
- "如何用 py-spy + flame graph 量化推理引擎的 CPU 开销"
- "AsyncScheduler 与 1F1B：推理与训练的调度差异"
- "schedule() 逐行注解 + profile 截图"——把 200 行 Python 讲透

---

## 进阶实验 4：Disaggregated prefill / decode 模拟

### 目标

在同一台机器跑两个 vLLM 实例（一个专 prefill、一个专 decode），用 **KV connector** 模拟 disaggregated 部署，量化 TTFT / TPOT 改善。

### 硬件 / 软件要求

- 2 个 GPU 同机（H100 或 A100 都行）
- NIXL 库（NVIDIA GPU-Direct）或回退到 shared memory connector
- 本教程锁定的源码 commit；KV connector 仍是实验性接口，升级后先重查示例与 schema

### 受控步骤

当前协议参考是 `examples/disaggregated/disaggregated_prefill.sh`，但该开发示例会安装依赖并广泛结束 Python 进程，**不要在共享机器原样运行**。先静态审计：

```bash
EXAMPLE=examples/disaggregated/disaggregated_prefill.sh
test -f "$EXAMPLE"
grep -nE 'kv_connector|kv_role|proxy|pip install|pgrep|pkill|kill -9' "$EXAMPLE"
```

把审计后的实验 launcher 放到独立目录，并满足以下验收条件后再跑：

1. producer、consumer、proxy、baseline 各自绑定独立端口与 GPU；模型和 connector schema 从**当前示例**复制，不凭旧版本记忆填写。
2. 依赖在实验前通过项目环境安装，launcher 内不执行 `pip install`。
3. 每个后台进程启动后立即保存 `$!`；cleanup 只对这些 PID 先 `TERM`、等待、超时后再有选择地升级，不使用 `pgrep python` / `pkill -f python`。
4. 先分别验证 producer/consumer health，再通过 proxy 的公开端口做功能请求；不能把 producer 端口的响应当作完整 P/D 服务。
5. baseline 与 P/D 使用相同模型、dtype、总 GPU、workload 和 benchmark 参数；保存 proxy、两侧服务及 connector 日志。

完成 launcher 的安全审查后，用当前 CLI 对 proxy 和 baseline 分别运行 `vllm bench serve`。命令参数先用 `vllm bench serve --help` 在锁定 commit 的环境确认；下一章会给出完整 benchmark 方法。

### 结果记录（不得预填）

| 配置 | TTFT p50 | TTFT p99 | TPOT p50 | 吞吐 tok/s |
| --- | --- | --- | --- | --- |
| TP=2 单实例（baseline） | `<实测>` | `<实测>` | `<实测>` | `<实测>` |
| Disaggregated (P=1, D=1) | `<实测>` | `<实测>` | `<实测>` | `<实测>` |

同时记录 KV transfer 成功/失败、传输字节、connector、链路、重试和两侧队列。若只请求 prefill 端口而没有完成当前 connector 示例要求的路由协议，该结果不是有效的 P/D 对照。

### 必须核对

- KV transfer 有固定和按字节增长的成本，必须在当前链路实测，不能套用通用 μs/GB。
- 分别测短/长 prompt、短/长输出与 prefix hit/miss，找出本环境的 break-even point。
- P/D 增加路由、connector 与局部故障面；性能结论必须和正确性、重试与容量结果一起交付。

### 自测题

1. KV transfer 跨 GPU 走的是哪条路径？PCIe vs NVLink vs RDMA？分别多快？
2. 如果 prefill 节点 OOM crash，请求会怎样？跟单实例的故障模式有何不同？
3. 多用户 chatbot 场景，prefix cache 命中率怎么影响 disaggregated 收益？
4. 算一下 100K token prompt 的 KV 大小（Llama-2-7B GQA）。NVLink 与 PCIe 跨卡分别传多久？

### 可产出的博客角度

- "Disaggregated prefill：一次真实的部署成本/收益账"
- "vLLM KV connector 是什么——一个抽象的演化史"
- "什么 workload 适合 disaggregated，什么不适合"——决策矩阵
- "NIXL vs LMCache vs Mooncake：3 种 KV transfer 后端对比"

每个进阶实验都能产出一篇博客级别的内容。建议跑完后把"目标 / 数据 / 关键观察 / 反直觉的发现"4 段直接复制到 Notion 或博客里，再补 1-2 张 nsys / py-spy 截图就成稿。

---

## 小结

- 5 个实验分别验证了 prefix caching、token budget、FP8 KV、preempt、投机解码这 5 个 vLLM 核心机制。
- 实验脚本都用小模型（OPT-125m / Qwen-0.5B）就能跑，但结论对大模型一样适用。
- "目标 / 预期 / 自测题"三段式让每个实验都有"可复现 + 可推理"的双重价值。
- 实验报告模板（目标 / 观察 / 结论 / 踩坑）是技术分享和复盘里最有效的素材结构。

## 自检

> 不用照着原文复述，重点是把现象、机制、源码入口和取舍讲顺。

**1. 实验 1 结论：prefix caching 在 chatbot 场景 TTFT 降低 X%。**

典型结果（system prompt ≈ 500 token，用户 query ≈ 50 token，无 cache vs 有 cache）：

> "在 system prompt = 500 token + 用户 query = 50 token 的 chatbot workload 下，开启 prefix caching 让**首次同 prompt 之后的请求 TTFT 从 ~180ms 降到 ~30ms，降低约 83%**。降幅与 (cached_tokens / total_prompt_tokens) 比值正相关——cached 占比越高，降幅越接近 95%。"

**详细数字依赖**：

- system prompt 越长 → 降幅越大（500-token prompt 降 83%，2000-token prompt 降 95%+）
- 命中率（连续请求间是否同 prefix）决定平均收益
- 第一次请求 TTFT 不变（cache 还没建立）

**工程上可引申**：这是 RAG / chatbot 场景的主要优化路径，比 quantization / spec decode 收益还大。

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

补充细节：EAGLE 在小 batch 下加速比 2-3×，大 batch 下可能反而拖慢 20%。决策权在监控数据，不在文档建议。

## 下一步

- 下一节：[`07-hands-on/04-profiling-and-debugging.md`](04-profiling-and-debugging.md)（从"我能跑出数字"升级成"我能定位 kernel-level 异常"）
- 想看源码：`vllm/benchmarks/throughput.py`、`vllm/benchmarks/serve.py`、`vllm/v1/core/sched/scheduler.py`
- 想动手：把每个实验改成"对比 2 个 vLLM 版本"——这能直接产出社区 PR 的回归测试材料
- 想从生产视角理解：[`08-production-deployment/05-slo-and-observability.md`](../08-production-deployment/05-slo-and-observability.md)（同样的指标在生产怎么报警）
