# vLLM 实验报告模板

> 默认值 `not-recorded` / `not-run` 必须保留到获得证据后再替换；不得把它们解释为通过。

## 1. 决策摘要

| 字段 | 值 |
| --- | --- |
| experiment ID | `exp-YYYYMMDD-sequence` |
| owner / reviewer | `not-recorded` / `not-recorded` |
| decision | `not-decided` |
| status | `planned` |
| conclusion | `not-run` |
| applicable scope | `not-recorded` |
| rollback trigger | `not-recorded` |

一句话问题：在 `<固定环境/workload>` 下，只改变 `<自变量>`，是否能在 `<质量/错误/SLO>` 约束内改善 `<目标指标>`？

## 2. 假设

- Mechanism hypothesis: `not-recorded`
- Falsifying outcome: `not-recorded`
- Expected direction（不填幅度）：
  - TTFT p99: `up / down / unchanged / not-applicable`
  - TPOT p99: `up / down / unchanged / not-applicable`
  - goodput: `up / down / unchanged / not-applicable`
  - error/quality: `must-not-regress`
- Confounders: `not-recorded`

## 3. Immutable 指纹

| 维度 | Baseline | Candidate |
| --- | --- | --- |
| vLLM SHA / version | `not-recorded` | `not-recorded` |
| image digest | `not-recorded` | `not-recorded` |
| model ID + revision | `not-recorded` | `not-recorded` |
| tokenizer/template hash | `not-recorded` | `not-recorded` |
| weight / KV dtype | `not-recorded` | `not-recorded` |
| quantization/backend | `not-recorded` | `not-recorded` |
| TP/PP/DP/EP | `not-recorded` | `not-recorded` |
| hardware/topology | `not-recorded` | `not-recorded` |
| driver/CUDA/PyTorch | `not-recorded` | `not-recorded` |
| resolved config artifact | `artifacts/not-recorded` | `artifacts/not-recorded` |

唯一允许变化的字段：`not-recorded`。其他差异：`none-recorded`。

## 4. Workload

| 字段 | 值 | 单位/证据 |
| --- | ---: | --- |
| dataset + hash / generator revision | `not-recorded` | SHA-256 / commit |
| seed | `not-recorded` | integer |
| num warmups / prompts / repeats | `not-recorded` | requests / rounds |
| arrival model | `not-recorded` | open-loop / closed-loop / mixed |
| request rate / burstiness | `not-recorded` | req/s / ratio |
| max concurrency | `not-recorded` | requests |
| input tokens p10/p50/p90/p99/max | `not-recorded` | tokens |
| output tokens p10/p50/p90/p99/max | `not-recorded` | tokens |
| streaming/features | `not-recorded` | booleans/percentages |
| client host/network | `not-recorded` | topology |

敏感信息处理：`not-recorded`。

## 5. 命令与时间

- Baseline command artifact: `artifacts/baseline-command.txt`
- Candidate command artifact: `artifacts/candidate-command.txt`
- Help/config artifacts: `artifacts/cli-help.txt`, `artifacts/resolved-config.txt`
- UTC intervals: `not-recorded`
- Warmup/cache state: `not-recorded`
- Stop conditions: `not-recorded`
- Rollback command/runbook: `not-recorded`

命令文件必须 redacted；Authorization、secret、原始用户 prompt 不得进入仓库。

## 6. 结果

| 指标 | Unit | Baseline median [min,max] | Candidate median [min,max] | Delta | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| completed / failed | requests | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| request throughput | req/s | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| output throughput | token/s | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| request goodput | req/s | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| TTFT p50/p90/p99 | ms | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| TPOT p50/p90/p99 | ms | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| ITL p50/p90/p99 | ms | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| E2E p50/p90/p99 | ms | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| preemption rate | req/s | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| KV usage peak | ratio | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| quality score | task unit | `not-run` | `not-run` | `not-run` | `not-evaluated` |
| cost proxy | currency/M token | `not-run` | `not-run` | `not-run` | `not-evaluated` |

公式：

```text
delta_pct = (candidate - baseline) / baseline × 100%
error_rate = failed / attempted
goodput_ratio = request_goodput / request_throughput
```

## 7. 证据索引

| Artifact | Baseline | Candidate | SHA-256 | Contains sensitive data |
| --- | --- | --- | --- | --- |
| benchmark JSON | `artifacts/baseline.json` | `artifacts/candidate.json` | `not-recorded` | `no-recorded` |
| server log | `artifacts/baseline.log` | `artifacts/candidate.log` | `not-recorded` | `redacted-recorded` |
| metrics | `artifacts/baseline-metrics.txt` | `artifacts/candidate-metrics.txt` | `not-recorded` | `no-recorded` |
| trace/profile | `not-collected` | `not-collected` | `not-recorded` | `not-assessed` |
| quality results | `artifacts/baseline-quality.json` | `artifacts/candidate-quality.json` | `not-recorded` | `redacted-recorded` |

## 8. 有效性与结论

- One variable changed: `not-evaluated`
- Workload/token distributions comparable: `not-evaluated`
- No unexpected fallback/OOM/error: `not-evaluated`
- Client/network not saturated first: `not-evaluated`
- Repeats/order sufficient: `not-evaluated`
- Quality and safety gates pass: `not-evaluated`
- Hardware verification run ID: `none-recorded`

Conclusion: `supported / rejected / inconclusive / not-run`。

Decision and scope: `not-decided`。

## 9. Rollback 与复核

- Rollback executed/tested: `not-run`
- Golden request after rollback: `not-run`
- Metrics returned to baseline: `not-run`
- Reviewer sign-off + UTC: `not-recorded`
- Next experiment with one new variable: `not-decided`

## Acceptance rubric

| 项 | Pass 条件 | 状态 |
| --- | --- | --- |
| Reproducibility | immutable 指纹、命令、数据 hash、seed、UTC 完整 | `not-evaluated` |
| Valid comparison | 单变量、实际 token/错误/轮次可比 | `not-evaluated` |
| Evidence | 原始 JSON/log/metrics 与 SHA-256 可访问 | `not-evaluated` |
| Safety | secret/redaction/stop/rollback 有证据 | `not-evaluated` |
| Decision | 结论、适用范围、反证与限制明确 | `not-evaluated` |
