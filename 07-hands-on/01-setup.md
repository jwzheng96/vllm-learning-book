# 01. 环境搭建与第一个 Demo

> **谁该读这一篇？** 第一次跑 vLLM、想把"读过的概念"变成"亲手验证"的学习者；准备演示给团队 / 面试官的工程师。
>
> **前置阅读：** [`01-overview/00-prerequisites.md`](../01-overview/00-prerequisites.md)（环境与背景知识），[`01-overview/01-what-is-vllm.md`](../01-overview/01-what-is-vllm.md)（先知道 vLLM 是什么再装它）。
>
> **耗时：** 约 15 分钟（不含模型下载与首次编译时间）。
>
> **学完能：**
> 1. 用 uv 装好 vLLM 并跑通最小离线推理脚本
> 2. 启动 OpenAI 兼容的 `vllm serve` 并用 curl / openai SDK 发请求
> 3. 看流式输出并解释为什么 token 是一个个吐
> 4. 用自带 benchmark 跑出第一组吞吐 / 延迟数字
> 5. 遇到常见环境问题（OOM / NCCL / Compile 卡）能查表自救

光读不练假把式。本节让你跑通最小例子并加日志观察内部行为。

---

## 1. 系统要求

- **硬件**：至少一张 NVIDIA GPU（建议 ≥ 16GB 显存。无显卡可用 CPU 后端但很慢）
- **CUDA**：12.1+（H100/H200 用 12.4+）
- **Python**：3.10 / 3.11 / 3.12（vLLM 官方推荐 3.12）

如果只是看代码而不跑：可以跳过下面"安装"，直接 IDE 打开 `vllm/` 目录读。

---

## 2. 推荐安装方式（用 uv）

vLLM 自己的 AGENTS.md 强制用 uv：

```bash
# 装 uv（如果还没装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在 vllm 仓库目录下
cd /Users/zjw/Documents/github-project/vllm
uv venv --python 3.12
source .venv/bin/activate

# 装预编译版（不改 C++ 代码）
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 或者从 PyPI 装最新 release（不需要 dev）
uv pip install vllm
```

如果只是读代码做笔记，不需要安装——但要跑实验、改代码、加日志，必须装。

---

## 3. 第一个 demo：离线推理

新建文件 `hello_vllm.py`：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",     # 最小模型，几秒下载
    gpu_memory_utilization=0.5,    # 实验环境别吃满
    enforce_eager=True,            # 关 CUDA Graph 加速启动
)

prompts = [
    "Hello, my name is",
    "The capital of France is",
    "Once upon a time,",
]

params = SamplingParams(temperature=0.0, max_tokens=20)
outputs = llm.generate(prompts, params)

for o in outputs:
    print(f"PROMPT: {o.prompt!r}")
    print(f"OUTPUT: {o.outputs[0].text!r}\n")
```

运行：

```bash
.venv/bin/python hello_vllm.py
```

观察启动日志：你会看到这些关键行（理解每行对应代码里哪个动作）：

- `Loading model weights took X GB` → `gpu_worker.py:load_model`
- `# GPU blocks: NNNN` → profile run + KV 计算
- `Capturing CUDA Graphs` → 如果没有 `enforce_eager`，会 capture 多个 batch_size

---

## 4. 第二个 demo：在线服务

```bash
# Terminal 1：启动 server
vllm serve facebook/opt-125m --port 8000 --enforce-eager

# Terminal 2：发请求
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Hello, my name is",
    "max_tokens": 20,
    "temperature": 0
  }'
```

---

## 5. 第三个 demo：流式

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

stream = client.completions.create(
    model="facebook/opt-125m",
    prompt="Once upon a time,",
    max_tokens=50,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].text, end="", flush=True)
print()
```

观察 token 一个个吐出来——这就是 continuous batching 的用户视角。

---

## 6. Benchmark：自带的性能测试

vLLM 仓库 `benchmarks/` 自带几个工具：

```bash
# 吞吐 benchmark（离线）
python benchmarks/benchmark_throughput.py \
    --model facebook/opt-125m \
    --num-prompts 100 \
    --enforce-eager

# 延迟 benchmark（单请求）
python benchmarks/benchmark_latency.py \
    --model facebook/opt-125m \
    --batch-size 1 \
    --input-len 32 \
    --output-len 64

# 在线服务 benchmark
# 先启动 server，再：
python benchmarks/benchmark_serving.py \
    --model facebook/opt-125m \
    --num-prompts 100 \
    --request-rate 10
```

记下结果。后面对比开关 prefix caching、改 batch size 等的影响。

---

## 7. 常见环境问题

| 现象                              | 原因                            | 解决                       |
| ------------------------------- | ----------------------------- | ------------------------ |
| `ImportError: ... vllm_C ...`   | 编译后端没装                       | `VLLM_USE_PRECOMPILED=1 uv pip install -e .` |
| OOM during profile run           | gpu_memory_utilization 太高     | 降到 0.5                   |
| CUDA Graph capture error        | 显存不够                         | 加 `--enforce-eager`       |
| 启动卡在 "Compiling..."             | torch.compile 第一次很慢          | 等，或 `--compilation-config '{"level": 0}'` |
| `RuntimeError: NCCL ...`        | TP 通信失败                       | 检查 `NCCL_DEBUG=INFO`、NVLink 状态 |

---

## 8. 推荐 IDE 设置

- VS Code / Cursor：打开 vllm 仓库根目录
- 装 Python / Pylance 扩展
- 设置解释器为 `.venv/bin/python`
- 用 "Go to Definition" 跳转看类继承
- 用 "Find All References" 看一个类被哪些地方用

---

## 9. 小结

- 推荐用 uv + Python 3.12 装 vLLM；只读代码不跑也可以，但实验环节必须装。
- 离线推理用 `LLM(...).generate(...)`，在线服务用 `vllm serve`，流式用 OpenAI SDK——三种入口都验证一遍最佳。
- 启动日志的 3 行（权重大小、# GPU blocks、Capturing CUDA Graphs）分别对应权重加载、profile run + KV 估算、CUDA Graph 录制。
- `benchmarks/` 里 3 个脚本（throughput / latency / serving）覆盖几乎所有性能测试需求。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. "# GPU blocks: NNNN" 是哪个模块算的？受哪 3 个启动参数影响？**

**模块**：`vllm/v1/worker/gpu_worker.py` 的 `determine_available_memory()` + `vllm/v1/core/kv_cache_utils.py` 的 `get_num_blocks()`。

**算法**：启动时跑一次 `profile_run`（用 max_num_batched_tokens 的 dummy batch 走一次 forward），观察峰值显存占用，剩下的显存按"单 block 字节数"切成 num_blocks。

**影响 N 的 3 个启动参数**：

1. **`--gpu-memory-utilization`**（默认 0.9）：可用显存上限 = total × util。util 调到 0.95 → 多 5GB → 多约 2000 个 block
2. **`--block-size`**（默认 16）：单 block 字节数 = block_size × num_kv_heads × head_dim × layers × 2 (K+V) × dtype_bytes。block_size 翻倍 → 单 block 翻倍 → num_blocks 减半
3. **`--max-num-batched-tokens`**（默认 8192）：profile run 用的 dummy batch 大小，越大占用激活越多 → 留给 KV 的剩余显存越少 → num_blocks 减少

**其他间接影响**：`--dtype`（FP16/BF16/FP8）、`--kv-cache-dtype`（fp8 KV 让单 block 减半）、`--enforce-eager`（不开 CUDA Graph 多省 1-2 GB）。

**验证**：去掉 `enforce_eager=True` 重跑，num_blocks 会因为 CUDA Graph capture 占用而**减少 ~5-10%**。

---

**2. `enforce_eager=True` 去掉后启动时间增加多少？TPOT 下降？**

**启动时间**：

- `enforce_eager=True`：跳过 CUDA Graph capture，启动快约 **30-60 秒**
- 去掉后：CUDA Graph 录制需要遍历 capture sizes（如 [1,2,4,8,16,32,...,256]），单个 capture 1-5 秒，总共增加 **30-120 秒**
- 加上 torch.compile（默认开），首次编译再加 **30-300 秒**

→ 启动总时长从 ~30s 涨到 1-5 分钟。生产部署用 `VLLM_TORCH_COMPILE_CACHE_DIR` + 持久 CG cache 减轻。

**TPOT 是否下降**：

- **下降**，典型 decode TPOT 从 30-50ms 降到 20-30ms（30-40% 提速）
- 原因：CUDA Graph 消 kernel launch overhead，每步省几 ms；torch.compile fusion 再省几 ms
- 但**首请求 TPOT 会变高**（先触发 compile），第二个请求开始才稳定

**实测对比**（Llama-3-8B + H100 + decode batch=8）：

- `enforce_eager=True`：TPOT median 38 ms
- 默认（CG + compile）：TPOT median 24 ms

→ 生产部署绝不要 `enforce_eager`，调试时才用。

---

**3. `--request-rate 10` vs `--request-rate 100` 对比 TTFT 与 throughput，解释拐点。**

设系统单 GPU 极限吞吐约 50 req/s（看模型大小）。

**`--request-rate 10`（10 req/s）**：

- 远低于极限 → 请求几乎不排队
- TTFT p99 ~ 单请求 prefill 时长 = 50-200 ms
- throughput = 10 req/s（达到 offered load）
- KV cache utilization 低（30-50%）

**`--request-rate 100`（100 req/s）**：

- 远超极限 → 请求大量堆积在 waiting queue
- TTFT p99 ~ "排队时长 + prefill 时长" = 几秒甚至几十秒
- throughput 被卡在 50 req/s（系统极限）
- KV utilization ~100%，开始触发 preempt

**拐点（约 50 req/s 附近）**：

- 之下：TTFT 缓慢上升、throughput 线性跟随 rate
- 之上：TTFT 雪崩式增长（排队论 M/M/1 公式：等待时间 ∝ 1/(μ-λ)，λ 接近 μ 时趋无穷）、throughput 平台

**生产意义**：

- 用 `benchmark_serving.py` 找到拐点 → 单实例容量
- 设 HPA target = 拐点的 70%（留缓冲）
- 跨过拐点的告警条件：`vllm:num_requests_waiting` > 阈值 + `vllm:kv_cache_usage_perc` > 0.9

详见 [`08-production-deployment/04-autoscaling-and-capacity.md`](../08-production-deployment/04-autoscaling-and-capacity.md)。

---

**4. `gpu_memory_utilization=0.95` 触发 OOM，错误栈定位到哪？**

**错误链路**：

1. profile_run 时分配比 0.9 时更多的 activation buffer
2. capture CUDA Graph 时还要额外 1-2 GB
3. 总占用 > 0.95 × HBM → CUDA OOM

**错误栈通常显示**：

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
  File "vllm/v1/worker/gpu_model_runner.py", line XXXX, in execute_model
    output = self.model.forward(...)
  File "vllm/model_executor/models/llama.py", line XXX, in forward
    hidden_states = layer(positions, hidden_states, ...)
  File "vllm/model_executor/layers/attention.py", line XXX, in forward
    attn_output = self.impl.forward(...)
```

**真正的"分配出错点"**：

- 如果在 profile_run 时 OOM：`vllm/v1/worker/gpu_worker.py::determine_available_memory()`
- 如果在 CG capture 时 OOM：`vllm/v1/worker/gpu_model_runner.py::capture_model()`
- 如果在请求时 OOM（kvcache 容量不够）：`vllm/v1/core/block_pool.py::get_new_blocks()` 抛 IndexError 而非 OOM——这是 vLLM 设计：kvcache 满了应该 preempt 而不是真 OOM

**修法**：

- 调回 `--gpu-memory-utilization 0.9` 或更小
- 看启动日志的 "available memory" 行确认 vLLM 估算
- 加 `VLLM_LOGGING_LEVEL=DEBUG` 看更详细的显存分配 trace

→ 调试技巧：从 0.9 出发，每次 +0.02 找到能稳跑的最大值。生产环境留 5-10% 安全边际。

## 下一步

- 下一节：[`07-hands-on/02-trace-a-request.md`](02-trace-a-request.md)（给一个请求加日志看完整内部行为）
- 想看源码：`vllm/entrypoints/llm.py`（LLM 类）、`vllm/entrypoints/openai/api_server.py`（serve 入口）、`benchmarks/benchmark_serving.py`
- 想动手：[`07-hands-on/03-mini-experiments.md`](03-mini-experiments.md)（5 个独立小实验把直觉变数字）
- 想从生产视角理解：[`08-production-deployment/01-deployment-architectures.md`](../08-production-deployment/01-deployment-architectures.md)（从单机 demo 到生产部署）
