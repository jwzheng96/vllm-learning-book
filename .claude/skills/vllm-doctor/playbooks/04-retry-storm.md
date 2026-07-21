# Playbook 04: 客户端重试雪崩 (Retry Storm)

## Symptom Reconfirm
- **必须同时**：
  - 网关 request-attempt rate 相对同类时段基线突增
  - 429/5xx/timeout 或客户端 attempt count 同步上升
  - **token 生成速率没同步涨**（说明请求是无效重试，不是真流量）
- vLLM 没有通用的 HTTP 失败 counter；错误码与尝试次数必须从实际网关/客户端指标取证。

## Triage Commands

```bash
# 1. QPS vs 生成速率（10 分钟）
curl -sS "$PROM_URL/api/v1/query_range?query=sum(rate(vllm:request_success_total[1m]))&start=$(($(date +%s)-600))&end=$(date +%s)&step=15" \
  > "$INCIDENT_DIR/evidence/qps-10m.json"
curl -sS "$PROM_URL/api/v1/query_range?query=sum(rate(vllm:generation_tokens_total[1m]))&start=$(($(date +%s)-600))&end=$(date +%s)&step=15" \
  > "$INCIDENT_DIR/evidence/gen-tps-10m.json"

# 2. 错误码分布：只有部署明确提供网关查询时才执行
if [ -n "${GATEWAY_STATUS_QUERY:-}" ]; then
  curl -sS -G --data-urlencode "query=${GATEWAY_STATUS_QUERY}" \
    "$PROM_URL/api/v1/query" > "$INCIDENT_DIR/evidence/error-by-status.json"
fi

# 3. Gateway / 网关层日志（不同部署位置不同，下面只是示例）
kubectl logs -n ${VLLM_NAMESPACE:-vllm} -l app=vllm-gateway --tail=200 \
  > "$INCIDENT_DIR/evidence/gateway.log"
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| attempt rate 上升但 token 生成率不变 + 多数 5xx | 缺少退避是候选原因；用 client/request ID 关联验证 |
| 429 上升但同一 client 的 attempt 仍升 | 客户端可能忽略限流；检查 retry-after 与 SDK policy |
| QPS 涨但 GPU 空闲 | 网关丢请求或上游路由错配 |
| attempt 与成功 token 同升、KV 越过本池水位 | 可能是真实流量叠加重试 → 联合看 01 |

## Remediate

- **L1（只读）**：
  - 抓证据
- **L2（需逐命令批准）**：按已定义租户预算收紧 admission；让过载错误符合既有 API 契约；按业务长度分布限制输入/输出；每项都展示受影响主体和回滚。
- **L3（需逐命令批准）**：临时阻断已确认的 client ID 或切流到降级池；这是外部可用性变更，必须单独批准并设过期时间。

## Verification

- 在完整重试窗口内，同一逻辑请求的 attempt count 回到批准预算
- attempt/success/token rate 恢复到同类时段健康区间
- 429/5xx/timeout 满足该服务错误预算；不能要求所有 5xx 永远为零

## Long-term

- SDK 按 API 契约实现指数退避、随机抖动、deadline 与总尝试预算
- 幂等性、错误类型和 `Retry-After` 共同决定能否重试；不要写死“所有 429/503 都重试”
- admission control 使用租户预算、queue 与本池容量实验，不照抄固定 KV 阈值
- 见 `reference/checklist-prelaunch.md` 第 11 条

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md + 07-incident-playbook.md case 4 -->
