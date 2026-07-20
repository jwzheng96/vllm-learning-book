# vLLM 容量计划模板

> 未测字段使用 `not-measured`；估算必须标 `estimated`，不能伪装成 benchmark。

## 1. Scope

| 字段 | 值 |
| --- | --- |
| service / owner / review UTC | `not-recorded` |
| source/image/model revision | `not-recorded` |
| region / failure domains | `not-recorded` |
| planning horizon | `not-recorded` |
| workload contract link | `artifacts/not-recorded` |
| experiment evidence link | `artifacts/not-recorded` |

## 2. Demand

| Scenario | Offered req/s | Admitted req/s | Input token/s | Output token/s | Input p99 | Output p99 | Burst duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| steady | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-recorded` |
| peak | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-recorded` |
| degraded/failover | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-measured` | `not-recorded` |

```text
admitted_rps = offered_rps × (1 - policy_rejection_fraction)
input_token_rate = admitted_rps × mean_actual_input_tokens
output_token_rate = admitted_rps × mean_actual_output_tokens
```

分布证据：`not-recorded`。季节性/增长假设：`not-recorded`。

## 3. SLO 与单副本能力

| Gate | Target | Measurement window | Evidence |
| --- | ---: | --- | --- |
| availability | `not-recorded` | `not-recorded` | `not-recorded` |
| TTFT p99 | `not-recorded ms` | `not-recorded` | `not-recorded` |
| TPOT p99 | `not-recorded ms` | `not-recorded` | `not-recorded` |
| E2E p99 | `not-recorded ms` | `not-recorded` | `not-recorded` |
| error rate | `not-recorded %` | `not-recorded` | `not-recorded` |
| quality | `not-recorded unit` | `not-recorded` | `not-recorded` |

Per-replica goodput at all gates: `not-measured req/s`。

## 4. Replica 计算

| Input | Value | Unit | Type |
| --- | ---: | --- | --- |
| peak admitted request rate | `not-measured` | req/s | measured/estimated |
| per-replica goodput@SLO | `not-measured` | req/s | measured |
| headroom fraction | `not-recorded` | ratio | policy |
| failure-domain reserve | `not-recorded` | replicas | policy |

```text
required_replicas = ceil(peak_admitted_rps / per_replica_goodput_at_SLO)
headroom_replicas = ceil(required_replicas × headroom_fraction)
planned_replicas = required_replicas + headroom_replicas + failure_domain_reserve
```

Calculated result: `not-calculated`。Rounding/assumptions: `not-recorded`。

## 5. Memory / KV 校验

| Component | Per replica | Unit | Evidence/type |
| --- | ---: | --- | --- |
| model weights | `not-measured` | GiB | startup log/measured |
| non-KV runtime/compile workspace | `not-measured` | GiB | measured |
| usable KV capacity | `not-measured` | GiB or tokens | resolved config/log |
| safety headroom | `not-recorded` | GiB | policy |
| peak active-context demand | `not-estimated` | GiB or tokens | estimated + calibrated |

```text
approx_kv_bytes_per_token_per_layer = 2 × num_kv_heads × head_size × dtype_bytes
approx_request_kv = active_context_tokens × num_layers × kv_bytes_per_token_per_layer
```

Hybrid attention、sliding window、MLA、KV connector、block fragmentation 修正：`not-assessed`。校准 run ID：`none-recorded`。

## 6. Autoscaling 与 cold start

| Item | Value | Unit/evidence |
| --- | ---: | --- |
| detect window | `not-recorded` | seconds |
| scale decision metric | `not-recorded` | queue/goodput/custom |
| pod scheduling | `not-measured` | seconds |
| model download/load | `not-measured` | seconds |
| compile/capture | `not-measured` | seconds |
| ready-to-serve total | `not-measured` | seconds |
| burst lead time | `not-recorded` | seconds |

若 cold start > burst lead time，预案：`not-decided`（warm pool / predictive scale / admission / smaller fallback）。

## 7. Failure scenarios

| Scenario | Remaining capacity | SLO impact | Automatic action | Human action |
| --- | --- | --- | --- | --- |
| one replica lost | `not-calculated` | `not-assessed` | `not-recorded` | `not-recorded` |
| one node/failure domain lost | `not-calculated` | `not-assessed` | `not-recorded` | `not-recorded` |
| model storage unavailable | `not-calculated` | `not-assessed` | `not-recorded` | `not-recorded` |
| traffic exceeds plan | `not-calculated` | `not-assessed` | `not-recorded` | `not-recorded` |

Admission/degradation order：`not-recorded`。RTO/RPO：`not-recorded`。

## 8. Cost

| Scenario | Replicas | Instance cost/hour | Utilization | Cost/M output tokens | Confidence |
| --- | ---: | ---: | ---: | ---: | --- |
| steady | `not-calculated` | `not-recorded` | `not-measured` | `not-calculated` | low/medium/high |
| peak | `not-calculated` | `not-recorded` | `not-measured` | `not-calculated` | low/medium/high |
| failover reserve | `not-calculated` | `not-recorded` | `not-measured` | `not-calculated` | low/medium/high |

```text
cost_per_million_output_tokens = total_cost_per_second / output_tokens_per_second × 1_000_000
```

## 9. Decision / triggers

- Approved planned replicas: `not-decided`
- Scale-up trigger + window: `not-decided`
- Scale-down safety window: `not-decided`
- Re-benchmark trigger（source/model/hardware/workload change）：`not-decided`
- Owner / reviewer / expiry UTC: `not-recorded`

## Acceptance rubric

| 项 | Pass 条件 | 状态 |
| --- | --- | --- |
| Demand | 分布、peak/burst/failover、有来源 | `not-evaluated` |
| Capacity | 使用 goodput@SLO，不用裸最大吞吐 | `not-evaluated` |
| Memory | KV/weights/runtime/headroom 有证据 | `not-evaluated` |
| Resilience | failure domain、cold start、admission、RTO | `not-evaluated` |
| Cost | 单位、公式、confidence 完整 | `not-evaluated` |
