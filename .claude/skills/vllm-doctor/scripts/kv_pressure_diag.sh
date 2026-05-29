#!/usr/bin/env bash
# kv_pressure_diag.sh — 抢占速率 / 长尾请求 / KV 历史采样
# 用于 playbook 01-preempt-cascade 的 Triage 阶段
set -u
set -o pipefail

PROM_URL="${PROM_URL:-http://prometheus.monitoring:9090}"
VLLM_NAMESPACE="${VLLM_NAMESPACE:-vllm}"
VLLM_SERVICE_LABEL="${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm}"

OUT_DIR="${1:-./evidence/kv}"
mkdir -p "$OUT_DIR"

prom_range() {
  local q="$1"
  local out="$2"
  local end=$(date +%s)
  local start=$((end - 600))  # 最近 10 分钟
  curl -sS --max-time 10 -G \
    --data-urlencode "query=${q}" \
    --data-urlencode "start=${start}" \
    --data-urlencode "end=${end}" \
    --data-urlencode "step=15" \
    "${PROM_URL}/api/v1/query_range" > "$out"
}

prom_range 'sum(vllm:gpu_cache_usage_perc)' "$OUT_DIR/kv-usage-10m.json"
prom_range 'sum(rate(vllm:num_preemptions_total[1m]))' "$OUT_DIR/preempt-rate-10m.json"
prom_range 'sum(vllm:num_requests_waiting)' "$OUT_DIR/queue-10m.json"
prom_range 'sum(vllm:num_requests_running)' "$OUT_DIR/running-10m.json"
prom_range 'histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le))' "$OUT_DIR/ttft-p99-10m.json"

# 长尾请求：仍在 running 但已经跑超过 N 秒
prom_range 'max(vllm:request_inference_time_seconds)' "$OUT_DIR/longest-running-10m.json"

# 当前 max_num_seqs / max_num_batched_tokens 配置（从 env 取）
first_pod=$(kubectl get pods -n "$VLLM_NAMESPACE" -l "$VLLM_SERVICE_LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "$first_pod" ]; then
  kubectl exec -n "$VLLM_NAMESPACE" "$first_pod" -- bash -lc \
    "env | grep -E 'MAX_NUM_SEQS|MAX_NUM_BATCHED_TOKENS|GPU_MEMORY_UTILIZATION|KV_CACHE_DTYPE' | sort" \
    > "$OUT_DIR/vllm-env.txt" 2>&1 || true
fi

# 提取最新一个采样点做摘要
extract_latest() {
  python3 -c "
import json,sys
try:
  d=json.load(open('$1'))
  res=d.get('data',{}).get('result',[])
  if not res: print('null'); sys.exit(0)
  vs=res[0].get('values',[])
  if not vs: print('null'); sys.exit(0)
  print(vs[-1][1])
except Exception:
  print('null')
"
}

kv_now=$(extract_latest "$OUT_DIR/kv-usage-10m.json")
preempt_now=$(extract_latest "$OUT_DIR/preempt-rate-10m.json")
queue_now=$(extract_latest "$OUT_DIR/queue-10m.json")
longest=$(extract_latest "$OUT_DIR/longest-running-10m.json")

cat <<JSON
{
  "out_dir": "$OUT_DIR",
  "kv_usage_now": ${kv_now:-null},
  "preempt_rate_now": ${preempt_now:-null},
  "queue_now": ${queue_now:-null},
  "longest_running_seconds": ${longest:-null}
}
JSON
