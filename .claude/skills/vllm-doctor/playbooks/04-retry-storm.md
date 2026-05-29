# Playbook 04: 客户端重试雪崩 (Retry Storm)

## Symptom Reconfirm
- **必须同时**：
  - 请求 QPS 1 分钟内涨 2-5×
  - `request_failed_rate > 0.1/s` 或 5xx 显著上升
  - **token 生成速率没同步涨**（说明请求是无效重试，不是真流量）
- 区分点：与"真流量峰值"区别在于 `prompt_tokens_total / request_success_total` 比例突变（真流量这个比例稳定）

## Triage Commands

```bash
# 1. QPS vs 生成速率（10 分钟）
curl -sS "$PROM_URL/api/v1/query_range?query=sum(rate(vllm:request_success_total[1m]))&start=$(($(date +%s)-600))&end=$(date +%s)&step=15" \
  > "$INCIDENT_DIR/evidence/qps-10m.json"
curl -sS "$PROM_URL/api/v1/query_range?query=sum(rate(vllm:generation_tokens_total[1m]))&start=$(($(date +%s)-600))&end=$(date +%s)&step=15" \
  > "$INCIDENT_DIR/evidence/gen-tps-10m.json"

# 2. 错误码分布
curl -sS "$PROM_URL/api/v1/query?query=sum(rate(vllm:request_failed_total[5m]))by(reason)" \
  > "$INCIDENT_DIR/evidence/error-by-reason.json"

# 3. Gateway / 网关层日志（不同部署位置不同，下面只是示例）
kubectl logs -n ${VLLM_NAMESPACE:-vllm} -l app=vllm-gateway --tail=200 \
  > "$INCIDENT_DIR/evidence/gateway.log"
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| QPS 翻倍但 token 生成率不变 + 多数 5xx | 客户端没有退避，直接死循环重试 |
| `request_failed_total` 主因是 `429` 但 QPS 也涨 | 网关已经限流但客户端无视，继续重试 |
| QPS 涨但 GPU 空闲 | 网关丢请求或上游路由错配 |
| QPS 涨且 KV 飙到 0.95 | 真实流量 + 客户端踩雷 → 看 01 |

## Remediate

- **L1（直接做）**：
  - 抓证据
- **L2（直接做）**：
  - 网关把整体 rate limit 砍到当前 QPS 的 0.5×
  - 5xx 改 503（明确告诉客户端"过载"）而不是 500（语义模糊会被无脑重试）
  - 网关 `max_tokens <= 2048`，挡住"恶意长输出"
  - 关键客户端走白名单，其他降级
- **L3（弹确认）**：
  - 联系出问题的客户端 owner，要求其 SDK 升级或下线
  - 极端情况下：网关临时阻断特定 client-id（影响：被阻断方完全不可用）

## Verification

- 3 分钟内 `request_failed_rate < 0.01/s`
- QPS 回到雪崩前基线 ± 20%
- 5xx 错误码归零

## Long-term

- 所有 SDK 必须实现指数退避 + 随机抖动（1s→2s→4s→8s + jitter）
- 重试预算上限：单个请求最多重试 3 次
- 网关默认开启 admission control（KV > 0.85 直接 429）
- 客户端区分错误类型：429 / 503 = 重试，4xx = 不重试
- 见 `reference/checklist-prelaunch.md` 第 11 条

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md L180-201 + 07-incident-playbook.md case 4 -->
