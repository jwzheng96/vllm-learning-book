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
kv_usage=$(q 'max(vllm:gpu_cache_usage_perc)')
throughput=$(q 'sum(rate(vllm:generation_tokens_total[1m]))')
running=$(q 'sum(vllm:num_requests_running)')
prefix_cache_hit_rate=$(q 'avg(vllm:gpu_prefix_cache_hit_rate)')
preempt_rate_per_sec=$(q 'sum(rate(vllm:num_preemptions_total[5m]))')
request_failed_rate=$(q 'sum(rate(vllm:request_failed_total[5m]))')
format_compliance_rate=$(q 'avg(vllm:format_compliance_rate)')

null_to_zero() { [ "$1" = "null" ] || [ -z "$1" ] && echo "0" || echo "$1"; }

ttft_p99_ms=$(null_to_zero "$ttft_p99_ms")
queue=$(null_to_zero "$queue")
kv_usage=$(null_to_zero "$kv_usage")
throughput=$(null_to_zero "$throughput")
running=$(null_to_zero "$running")
prefix_cache_hit_rate=$(null_to_zero "$prefix_cache_hit_rate")
preempt_rate_per_sec=$(null_to_zero "$preempt_rate_per_sec")
request_failed_rate=$(null_to_zero "$request_failed_rate")
# format_compliance_rate 默认 1（很多部署没采，没采当合规处理）
[ "$format_compliance_rate" = "null" ] || [ -z "$format_compliance_rate" ] && format_compliance_rate=1

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
