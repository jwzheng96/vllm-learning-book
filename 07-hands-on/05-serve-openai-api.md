# 05. 从零启动 OpenAI-Compatible API 服务

> **谁该读这一篇？** 第一次把 vLLM 当服务使用的读者；需要一份可复现、可清理、能产出面试证据的最小实验手册的工程师。
>
> **前置阅读：** [`01-setup.md`](01-setup.md)、[`03-code-walkthrough/08-input-processing-and-tokenization.md`](../03-code-walkthrough/08-input-processing-and-tokenization.md)、[`03-code-walkthrough/09-output-processing-and-streaming.md`](../03-code-walkthrough/09-output-processing-and-streaming.md)。
>
> **耗时：** 30–60 分钟（模型下载不计）。
>
> **学完能：** 启动/连接服务，验证 health/models/chat/completion/streaming/concurrency/metrics/error，安全停止，并交付一页证据报告。

> **静态复核：** 命令和指标按 `b23bd73f540175f9e117eaee5029cd7d8df63964` 核对；教程没有代你执行 GPU/CPU benchmark。

---

## 1. 先选择路线

| 路线 | 适合 | 前提 | 本章是否启动本地服务 |
| --- | --- | --- | --- |
| A. NVIDIA GPU | Linux + 受支持 CUDA/driver | vLLM 环境、可用显存、模型权限 | 是 |
| B. 受支持 CPU | vLLM 当前支持的 Linux CPU 平台 | 对应 CPU wheel/build 与足够 RAM | 是，但只做功能验证 |
| C. 远端 endpoint | macOS、无受支持 GPU/CPU、不能下载模型 | 获得授权测试 URL/key | 否 |

macOS 不作为本地运行 vLLM 的默认路线；用路线 C，或 SSH 到受支持 Linux 主机。不要为了完成教程在宿主机随意安装另一平台的二进制依赖。

路线 A/B 先执行当前安装指南对应的安装，再记录：

```bash
vllm --version
.venv/bin/python -c 'import vllm; print(vllm.__version__)'
```

版本命令失败就停止，不要继续把后续 404/连接失败误判成模型问题。

## 2. 定义实验变量

所有路线都使用环境变量；真实 key 不进入 Markdown、shell history 截图或日志。

```bash
export MODEL_ID='your-org/your-instruct-model'
export VLLM_BASE_URL='http://127.0.0.1:8000'
export VLLM_API_KEY='replace-with-a-temporary-lab-key'
export VLLM_LAB_DIR="$(mktemp -d)"
printf 'artifacts=%s\n' "$VLLM_LAB_DIR"
```

远端路线把 `VLLM_BASE_URL` 改成授权 endpoint，并从安全 secret store 注入 key。报告中只写 `key configured: yes`，不写值。若远端不要求鉴权，也保留变量并由客户端逻辑决定是否发送 header。

## 3. 启动服务（路线 A/B）

<!-- vllm-source: {"path":"vllm/entrypoints/cli/serve.py","symbol":"ServeSubcommand.cmd"} -->
[源码锚点：ServeSubcommand.cmd](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/entrypoints/cli/serve.py#L50)

先用最少参数建立基线；不要一开始同时加量化、TP、spec decode 和自定义 backend。

```bash
vllm serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  >"$VLLM_LAB_DIR/server.log" 2>&1 &
export VLLM_SERVER_PID=$!
printf '%s\n' "$VLLM_SERVER_PID" >"$VLLM_LAB_DIR/server.pid"
```

CPU 路线只应按当前 CPU 安装/平台文档增加必要参数。NVIDIA 路线若 OOM，先换更小模型或降低 `--gpu-memory-utilization`，不要盲目提高它。

观察启动：

```bash
tail -f "$VLLM_LAB_DIR/server.log"
```

看到服务 ready 后 Ctrl-C 只退出 `tail`，不会停止 server。

## 4. 验证 health 与 models

<!-- vllm-source: {"path":"vllm/entrypoints/serve/instrumentator/health.py","symbol":"health"} -->
[源码锚点：health endpoint](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/entrypoints/serve/instrumentator/health.py#L23)

```bash
curl -fsS "$VLLM_BASE_URL/health" | tee "$VLLM_LAB_DIR/health.txt"
curl -fsS "$VLLM_BASE_URL/v1/models" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  | tee "$VLLM_LAB_DIR/models.json"
```

成功证据：health 为 2xx，models 列表含你要调用的 served model ID。请求体的 `model` 应使用这个 ID，不要想当然照抄 Hugging Face ID。

## 5. Chat completion

```bash
curl -fsS "$VLLM_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"用两句话解释 continuous batching\"}],\"max_tokens\":64,\"temperature\":0}" \
  | tee "$VLLM_LAB_DIR/chat.json"
```

如果 `/v1/models` 返回的 ID 与 `MODEL_ID` 不同，定义 `SERVED_MODEL_ID` 并在后续请求使用它。

验收：HTTP 2xx、choice 非空、finish reason 可解释、usage token 数存在或符合服务配置。

## 6. Completion（仅模型/服务支持时）

```bash
curl -fsS "$VLLM_BASE_URL/v1/completions" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"PagedAttention 的核心是\",\"max_tokens\":32,\"temperature\":0}" \
  | tee "$VLLM_LAB_DIR/completion.json"
```

Chat-only 或自定义 task 不支持 completion 时，记录 `not supported by selected model/route`，不要把预期 4xx 写成实验失败。

## 7. Streaming 与 usage

```bash
curl -N -sS "$VLLM_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"按 1 到 10 编号输出\"}],\"max_tokens\":96,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
  | tee "$VLLM_LAB_DIR/stream.sse"
```

检查首个 delta、连续 chunk、finish reason、final usage 与 `[DONE]`。不要按 chunk 数计 token。

## 8. 并发请求

用标准库发送四个并发请求，避免先安装额外 benchmark 客户端：

```bash
.venv/bin/python - <<'PY'
import concurrent.futures, json, os, urllib.request

url = os.environ["VLLM_BASE_URL"] + "/v1/chat/completions"
key = os.environ["VLLM_API_KEY"]
model = os.environ["MODEL_ID"]

def send(index):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"返回请求编号 {index}"}],
        "max_tokens": 24,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    with urllib.request.urlopen(request, timeout=120) as response:
        return index, response.status, json.load(response)["choices"][0]["message"]["content"]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(send, range(4)):
        print(result)
PY
```

把输出重定向到 `$VLLM_LAB_DIR/concurrency.txt` 也可以。验收是四个 request ID/编号可区分、无静默丢失；这不是吞吐 benchmark。

## 9. 检查当前 metrics

<!-- vllm-source: {"path":"vllm/v1/metrics/loggers.py","symbol":"PrometheusStatLogger.__init__"} -->
[源码锚点：Prometheus metrics registration](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/v1/metrics/loggers.py#L451)

如果这个私有方法名在后续版本变化，语义锚点刷新门禁会阻止文档静默漂移。

```bash
curl -fsS "$VLLM_BASE_URL/metrics" >"$VLLM_LAB_DIR/metrics.txt"
grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)' "$VLLM_LAB_DIR/metrics.txt"
grep -E '^vllm:(prompt_tokens|generation_tokens|num_preemptions)_total' "$VLLM_LAB_DIR/metrics.txt"
grep -E '^vllm:(time_to_first_token_seconds|inter_token_latency_seconds|e2e_request_latency_seconds)_' "$VLLM_LAB_DIR/metrics.txt" | head
```

含义：

- running/waiting 是 gauge；
- KV 使用是比例 gauge；
- prompt/generation/preemption 是 counter，Prometheus 暴露 `_total`；
- TTFT/ITL/E2E 是 histogram，看 `_bucket/_count/_sum`，由 PromQL 算 rate/quantile。

空闲时 running/waiting 回 0 是正常的。为了观察非零值，要在并发请求进行时抓取。

## 10. 故意触发四类错误

### Invalid model

把 `model` 改为 `definitely-not-served`，期望 4xx 与结构化错误。

### Invalid input

发送空 `messages`、互斥字段或超过模型限制的输入，记录是哪一层拒绝。不要构造会占用巨大内存的 body。

### Invalid auth

```bash
curl -sS -o "$VLLM_LAB_DIR/auth-error.json" -w '%{http_code}\n' \
  "$VLLM_BASE_URL/v1/models" -H 'Authorization: Bearer redacted-invalid-key'
```

本地 route 应非 2xx；远端 route 只在得到授权时做该测试，避免触发安全告警。

### Limit / overload

优先用网关或测试环境明确配置的 token/concurrency limit。没有明确限额时只记录 `not exercised`，不要用无界并发制造事故。

错误实验记录 HTTP status、错误 code/type、request ID、是否进入 EngineCore、是否可重试；日志中删除 Authorization header 和 prompt 敏感内容。

## 11. 停止、drain 与清理

实验 server 不应使用 `pkill -f`，它可能误杀同机其他用户的服务。

```bash
kill -TERM "$VLLM_SERVER_PID"
wait "$VLLM_SERVER_PID"
```

生产 drain 应先从 LB/readiness 摘流，等 running/waiting 降到 0 或达到 deadline，再 TERM。验证端口关闭：

```bash
if curl -fsS "$VLLM_BASE_URL/health" >/dev/null 2>&1; then
  echo 'server still reachable; inspect before cleanup'
else
  echo 'server stopped'
fi
```

证据目录默认保留供报告使用。确认报告完成后再删除**打印出的那个精确临时路径**；不要在教程里给出宽泛递归删除命令。

## 12. 一页证据报告

```markdown
# vLLM serving lab evidence

- source/version:
- route: NVIDIA GPU / supported CPU / remote endpoint
- model + served model ID:
- environment: hardware, driver/runtime, Python, vLLM
- startup command: key redacted
- health/models evidence:
- chat/completion/streaming/concurrency results:
- metrics observed and exact names:
- negative tests: status + retry decision
- shutdown/drain evidence:
- hardware verification: none / indexed run ID
- limitations and next experiment:
```

只有填写 hardware、命令、时间、原始 artifact 索引和结果后，才能声称 hardware verified。

## 13. 面试表达

> 我会先用最小参数启动 `vllm serve`，通过 `/health` 和 `/v1/models` 验证 readiness/served ID，再覆盖 chat、stream、并发、usage 与负例。可观测性上看 running/waiting、KV usage、token counters、TTFT/ITL/E2E histograms。慢请求按输入处理、EngineCore queue、GPU、output/network 分段。停止时先摘流 drain，再按记录的 PID TERM，避免丢请求或误杀其他实例。

## 小结

- 先选受支持本地路线或远端路线；macOS 不假定本地运行。
- 最小基线通过后再加优化参数。
- 功能、streaming、并发、metrics、错误与 shutdown 都是服务验收的一部分。
- key 永不进入报告或日志，硬件声明必须有索引证据。

## 自检

1. `/health` 成功但 `/v1/models` 没目标模型，下一步查什么？
2. counter、gauge、histogram 在 `/metrics` 文本里如何区分？
3. 为什么并发功能测试不是 benchmark？
4. 客户端中断后如何确认 KV 最终释放？
5. 为什么不使用 `pkill -f "vllm serve"`？

### 参考答案

1. `/health` 只说明 HTTP 进程存活；`/v1/models` 没目标模型通常要查模型加载日志、model ID/alias、revision、权重挂载和 readiness 条件。确认服务没有只启动了健康端口而尚未完成模型初始化。
2. counter 通常单调递增并带 `_total`，gauge 可上下波动，histogram 会有 `_bucket/_count/_sum`。读取 `/metrics` 时还要确认 label、单位和当前版本名称，不能把任意带数字的行当成同一类型。
3. 功能测试关注正确性、HTTP、streaming 和错误路径；benchmark 还必须固定模型、长度、到达率、warmup、并发、客户端和硬件，并报告 p50/p99、goodput 与 token throughput。功能测试通过不代表容量可用。
4. 发送取消后，观察 request finished/aborted、KV usage、running/waiting、preemption 和 connector load/save 是否收敛；必要时用重复请求和时间窗口确认 block 最终归还。客户端断开本身不是释放证据。
5. `pkill -f` 可能误杀同机其他 vLLM、测试进程或 supervisor，跳过优雅 drain 和子进程清理。应记录 PID/进程组，使用服务管理器或明确的优雅 shutdown，再按证据处理残留进程。

## 下一步

- [`06-benchmark-methodology.md`](06-benchmark-methodology.md)：把“能服务”升级为可复现实验。
- [`04-profiling-and-debugging.md`](04-profiling-and-debugging.md)：当指标异常时逐层 profile。
