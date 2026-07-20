# 06. Benchmark 方法论：让性能数字可复现、可比较

> **谁该读这一篇？** 需要给配置选型、容量规划、回归门禁或面试项目提供可信性能证据的工程师。
>
> **前置阅读：** [`05-serve-openai-api.md`](05-serve-openai-api.md)、[`03-mini-experiments.md`](03-mini-experiments.md)。
>
> **耗时：** 40 分钟阅读；真实 endpoint 实验通常需要 1–3 小时。
>
> **学完能：** 区分 open-loop/closed-loop，设计单变量实验，正确解释 TTFT/TPOT/ITL/E2E/throughput/goodput，并交付可复核的原始 artifact。

> **当前源码复核：** 命令按 `b23bd73f540175f9e117eaee5029cd7d8df63964` 的 benchmark CLI 核对；本章不声称在该 SHA 上完成硬件实测。

---

## 1. 先把问题写成决策

差的 benchmark 问“vLLM 有多快”；可执行的 benchmark 问：

> 在固定模型、revision、硬件、软件栈和请求分布下，配置 A 相比 B 能否在错误率不升高、TTFT p99 ≤ 800 ms、TPOT p99 ≤ 50 ms 的约束内，提高 goodput 或降低每百万 token 成本？

800/50 ms 只是写法示例，必须替换为你的 SLO。没有决策、约束和 rollback 条件，数字再多也只是截图。

## 2. Open-loop 与 closed-loop

| 模式 | 客户端行为 | 主要控制量 | 适合回答 | 常见误读 |
| --- | --- | --- | --- | --- |
| Open-loop | 按到达过程发请求，不等前一批完成 | `--request-rate`、`--burstiness` | 给定外部流量时何时排队/失守 | 客户端跟不上时实际到达率低于配置 |
| Closed-loop | 保持一定并发，完成后再补请求 | concurrency | 固定并发下的饱和吞吐/延迟 | 隐藏真实排队，不能代表独立用户到达 |
| 混合约束 | 到达率 + 并发上限 | `--request-rate` + `--max-concurrency` | 网关限流后的系统行为 | 达到并发上限后实际 request rate 会下降 |

当前 `vllm bench serve` 的 `--request-rate inf` 会在开始时发送全部请求，不等于生产 open-loop。Poisson 到达用有限 request rate 和默认 burstiness；burstiness 变化本身也是一个自变量。

## 3. 固定实验指纹

每次运行至少记录：

- source SHA、vLLM version、容器 image digest；
- model ID + immutable revision、tokenizer ID/revision、chat template hash；
- quantization/weight dtype、KV dtype、attention backend；
- GPU/CPU 型号与数量、显存、拓扑、driver、CUDA、PyTorch；
- TP/PP/DP/EP、`max_num_seqs`、`max_num_batched_tokens`、KV/compile 配置；
- endpoint、客户端主机、网络路径、benchmark CLI 完整参数；
- dataset 名称/hash、seed、样本数、输入/输出 token 分布；
- warmup、运行时长、开始/结束 UTC、后台负载；
- 原始 JSON、server log、metrics snapshot、错误样本（敏感字段脱敏）。

缺少任一会影响结论的字段时，报告写 `not-recorded`，不要猜。

## 4. 请求分布比平均长度重要

平均 1K token 可以是“全都 1K”，也可以是“90% 100 token + 10% 9.1K”。两者的 KV、prefill、尾延迟完全不同。

至少报告：

| 维度 | 最少统计 |
| --- | --- |
| input tokens | p10/p50/p90/p99/max |
| requested output tokens | p10/p50/p90/p99/max |
| actual output tokens | p10/p50/p90/p99/max + finish reason |
| prefix | 可共享长度分布、hit/query 增量 |
| arrival | request rate、burstiness、并发上限 |
| features | streaming、logprobs、structured output、LoRA、多模态占比 |

Tokenizer 必须与服务一致。只按字符数构造“约 8K token”会让跨语言比较失真；保存 tokenizer 后的真实长度。

## 5. 指标定义

| 指标 | 定义 | 用途 | 陷阱 |
| --- | --- | --- | --- |
| TTFT | 发出请求到收到首 token | 交互首响应 | 包含排队、prefill、网络；非 streaming 无法可靠拆分 |
| TPOT | 首 token 后，每输出 token 的平均时间 | 持续生成速度 | 输出 ≤1 token 时无意义 |
| ITL | 相邻输出 token 间隔 | 卡顿/抖动 | 不能只报均值 |
| E2E | 请求开始到完成 | 总等待 | 强依赖输入/输出长度 |
| throughput | 每秒完成请求或处理 token | 资源效率 | 离开 SLO 的吞吐可能没有业务价值 |
| goodput | 同时满足成功与指定 SLO 的请求率 | 容量决策 | SLO 键/单位必须写清 |
| error rate | 失败/总请求 | 正确性约束 | timeout、4xx、5xx、client abort 要分类 |

所有 latency 至少给 p50/p90/p99；样本过少时 p99 不稳定，应增加样本或报告置信区间/重复轮次，不假装精确。

## 6. 当前 CLI 命令族

<!-- vllm-source: {"path":"vllm/benchmarks/serve.py","symbol":"add_cli_args"} -->
[源码锚点：serve benchmark CLI](https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/benchmarks/serve.py#L1478)

先留存 help：

```bash
vllm bench serve --help > bench-serve-help.txt
vllm bench throughput --help > bench-throughput-help.txt
vllm bench latency --help > bench-latency-help.txt
```

三类命令：

```bash
# 在线：回答到达率、排队、SLO/goodput
vllm bench serve --help

# 离线吞吐：回答同一进程尽可能快处理固定 workload 的能力
vllm bench throughput --help

# 单请求/固定 batch 延迟：用于模型执行基线，不代表生产排队
vllm bench latency --help
```

选错命令会测到不同系统边界，不能互相替代。

## 7. 真实 endpoint：可复制的基线

先按上一章安全启动服务，并定义：

```bash
export MODEL_ID='your-served-model-id'
export VLLM_BASE_URL='http://127.0.0.1:8000'
export BENCH_DIR="$(mktemp -d)"
printf 'bench artifacts=%s\n' "$BENCH_DIR"
```

若启用 API key，用工具当前支持的安全注入方式；不要把真实 key放进 command history、metadata 或结果 JSON。

功能 smoke 后跑低负载基线：

```bash
vllm bench serve \
  --backend openai \
  --base-url "$VLLM_BASE_URL" \
  --endpoint /v1/completions \
  --model "$MODEL_ID" \
  --dataset-name random \
  --input-len 256 \
  --output-len 128 \
  --num-warmups 10 \
  --num-prompts 200 \
  --request-rate 1 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result \
  --save-detailed \
  --result-dir "$BENCH_DIR" \
  --result-filename baseline-qps1.json \
  --metadata source_sha=b23bd73f540175f9e117eaee5029cd7d8df63964 experiment=baseline
```

`random` 适合机制对照，不代表真实业务。第二组应使用经过授权、脱敏且可 hash 的业务分布或 trace；若不能带出数据，保存分桶统计与生成规则。

## 8. 找容量拐点，而不是只跑最高 QPS

对固定 workload 逐步增加 request rate，例如 1/2/4/8……，每档独立运行并在两档之间恢复到空闲：

```bash
for rate in 1 2 4 8; do
  vllm bench serve \
    --backend openai \
    --base-url "$VLLM_BASE_URL" \
    --endpoint /v1/completions \
    --model "$MODEL_ID" \
    --dataset-name random \
    --input-len 256 --output-len 128 \
    --num-warmups 10 --num-prompts 400 \
    --request-rate "$rate" \
    --goodput ttft:800 tpot:50 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result --save-detailed \
    --result-dir "$BENCH_DIR" \
    --result-filename "open-loop-qps${rate}.json" \
    --metadata source_sha=b23bd73f540175f9e117eaee5029cd7d8df63964 request_rate="$rate"
done
```

把 800/50 换成你的毫秒 SLO。拐点是 goodput 不再线性增长、queue/TTFT 尾部快速恶化或错误开始增加的位置；不是 GPU util 第一次到 100% 的位置。

## 9. Warmup、重复与顺序效应

- 冷启动（下载/加载/compile）单独测，不混入 steady-state。
- warmup 覆盖 tokenizer、kernel、CUDA graph/compile 与 cache；记录它是否也填充 prefix cache。
- 每个配置至少重复 3 轮，轮次顺序 A/B/B/A 或随机化，避免温度、缓存、邻居负载单向漂移。
- prefix caching 实验要区分 cold miss、warm hit，并保存 hit/query counter。
- 每轮开始前证明 queue 归零；需要冷 cache 时用受控的新实例，而不是广泛杀进程或清理共享缓存。

## 10. 比较有效性检查

只有以下条件都满足，A/B 才能进入结论：

1. 除一个自变量外配置指纹一致；
2. 成功请求数、实际 token 分布、finish reason 可比；
3. 无 OOM、fallback、模型质量退化或错误率变化被忽略；
4. 客户端 CPU/网络没有先饱和；
5. server 日志确认 backend/quantization/parallelism 与预期一致；
6. 每轮原始结果和 UTC 可追溯；
7. 结论范围限定为本次环境/workload。

若不满足，结论写“inconclusive”，并列出下一次只需补的证据。

## 11. 无 GPU：练习结果分析

把下面示例保存为 `example-result.json`。它是教学数据，不是 vLLM 实测：

```json
{
  "label": "teaching-only",
  "source_sha": "b23bd73f540175f9e117eaee5029cd7d8df63964",
  "request_rate": 4,
  "completed": 98,
  "failed": 2,
  "request_throughput": 3.7,
  "request_goodput": 3.1,
  "percentiles_ttft_ms": {"50": 110, "90": 360, "99": 910},
  "percentiles_tpot_ms": {"50": 31, "90": 46, "99": 67},
  "slo_ms": {"ttft": 800, "tpot": 50}
}
```

用 Python 标准库做离线检查，不导入 vLLM、不需要 GPU：

```bash
python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("example-result.json").read_text())
total = data["completed"] + data["failed"]
error_rate = data["failed"] / total if total else 0
print(f"error_rate={error_rate:.2%}")
print(f"goodput_ratio={data['request_goodput'] / data['request_throughput']:.2%}")
for metric in ("ttft", "tpot"):
    p99 = data[f"percentiles_{metric}_ms"]["99"]
    limit = data["slo_ms"][metric]
    print(metric, "PASS" if p99 <= limit else "FAIL", p99, limit)
PY
```

应读出：错误率 2%，goodput 小于 throughput，TTFT/TPOT p99 均未满足示例 SLO。真正报告还要检查 schema 是否与当前 CLI 输出一致，不能让这段教学 schema 伪装成原生结果。

## 12. Artifact 目录与交付

建议结构：

```text
run-20260720T181306Z/
├── manifest.json
├── command.txt
├── bench-serve-help.txt
├── result.json
├── server.log
├── metrics-before.txt
├── metrics-during.txt
├── metrics-after.txt
├── errors-redacted.jsonl
└── analysis.md
```

manifest 保存每个文件的 SHA-256。密钥、完整 prompt、用户 ID、Authorization header 不进入 artifact；需要复核内容时保存脱敏/合成输入和数据生成器 revision。

完整报告从 [`templates/experiment-report.md`](templates/experiment-report.md) 复制。

## 面试表达

> 我先把性能问题写成带 SLO 的决策，再固定 source/model/tokenizer/硬件/workload 指纹。在线容量用 open-loop request-rate 扫描并计算 goodput，固定并发只做补充；每个配置预热、重复、保存原始 JSON/metrics/log。比较只改一个变量，同时检查实际 token、错误、fallback 和客户端瓶颈。最后把结论限制在本环境，并给出 rollback 条件。

## 自检

1. `--request-rate inf` 为什么不是生产 open-loop？
2. throughput 上升但 goodput 下降，应该如何解释？
3. 为什么 input/output 平均长度相同仍不能保证可比？
4. prefix caching A/B 的 warmup 如何避免污染？
5. 何时应把结果标为 inconclusive？

## 下一步

[`07-tuning-playbook.md`](07-tuning-playbook.md) 把指标症状映射为单变量调优实验。
