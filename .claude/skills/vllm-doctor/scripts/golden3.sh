#!/usr/bin/env bash
# golden3.sh — Phase 1：拉 Golden 3+ 指标，输出 JSON
# 支持 VLLM_DOCTOR_FIXTURE 走离线 fixture
set -u
set -o pipefail

PROM_URL="${PROM_URL:-http://prometheus.monitoring:9090}"

# Fixture mode（dry-run / 没集群时用）
if [ -n "${VLLM_DOCTOR_FIXTURE:-}" ]; then
  if [ ! -f "$VLLM_DOCTOR_FIXTURE" ]; then
    echo "[golden3] fixture 文件不存在: $VLLM_DOCTOR_FIXTURE" >&2
    exit 1
  fi
  cat "$VLLM_DOCTOR_FIXTURE"
  exit 0
fi

# instant query helper
q() {
  local query="$1"
  curl -sS --max-time 5 -G --data-urlencode "query=${query}" \
    "${PROM_URL}/api/v1/query" \
  | python3 -c "
import json,sys
try:
  d=json.load(sys.stdin)
  res=d.get('data',{}).get('result',[])
  if not res:
    print('null'); sys.exit(0)
  v=res[0].get('value',[None,None])[1]
  print(v if v not in (None,'') else 'null')
except Exception:
  print('null')
"
}

ttft_p99_ms=$(q 'histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)) * 1000')
queue=$(q 'sum(vllm:num_requests_waiting)')
kv_usage=$(q 'max(vllm:kv_cache_usage_perc)')
throughput=$(q 'sum(rate(vllm:generation_tokens_total[1m]))')
running=$(q 'sum(vllm:num_requests_running)')
prefix_cache_hit_rate=$(q 'sum(rate(vllm:prefix_cache_hits_total[5m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])), 1e-9)')
preempt_rate_per_sec=$(q 'sum(rate(vllm:num_preemptions_total[5m]))')
if [ -n "${GATEWAY_ERROR_QUERY:-}" ]; then
  request_failed_rate=$(q "$GATEWAY_ERROR_QUERY")
else
  request_failed_rate=null
fi
if [ -n "${FORMAT_COMPLIANCE_QUERY:-}" ]; then
  format_compliance_rate=$(q "$FORMAT_COMPLIANCE_QUERY")
else
  format_compliance_rate=null
fi

for required_value in \
  "$ttft_p99_ms" "$queue" "$kv_usage" "$throughput" "$running" \
  "$prefix_cache_hit_rate" "$preempt_rate_per_sec"; do
  if [ "$required_value" = "null" ] || [ -z "$required_value" ]; then
    echo "[golden3] required vLLM metric is missing; refusing to classify" >&2
    exit 1
  fi
done

cat <<JSON
{
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "ttft_p99_ms": ${ttft_p99_ms},
  "queue": ${queue},
  "kv_usage": ${kv_usage},
  "throughput": ${throughput},
  "running": ${running},
  "prefix_cache_hit_rate": ${prefix_cache_hit_rate},
  "preempt_rate_per_sec": ${preempt_rate_per_sec},
  "request_failed_rate": ${request_failed_rate},
  "format_compliance_rate": ${format_compliance_rate}
}
JSON
