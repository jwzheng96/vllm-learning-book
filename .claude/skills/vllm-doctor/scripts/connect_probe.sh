#!/usr/bin/env bash
# connect_probe.sh — Phase 0：探测 kubectl / prometheus / GPU 可达性
# stdout: JSON
# 任一关键组件不可达 → exit 1 + 在 stderr 打可执行的修复提示
set -u
set -o pipefail

VLLM_NAMESPACE="${VLLM_NAMESPACE:-vllm}"
VLLM_SERVICE_LABEL="${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm}"
PROM_URL="${PROM_URL:-http://prometheus.monitoring:9090}"

fail=0
declare -A status

# 1) kubectl
if kubectl config current-context > /dev/null 2>&1; then
  status[kubectl_context]="ok"
  ctx=$(kubectl config current-context)
else
  status[kubectl_context]="missing"
  ctx=""
  fail=1
  echo "[connect_probe] kubectl current-context 失败。导出 KUBECONFIG 或运行: aws eks update-kubeconfig --name <cluster>" >&2
fi

# 2) namespace + pod 可见
if [ "$ctx" != "" ]; then
  if kubectl get pods -n "$VLLM_NAMESPACE" -l "$VLLM_SERVICE_LABEL" --no-headers 2>/dev/null | head -1 | grep -q .; then
    status[pods]="ok"
    pod_count=$(kubectl get pods -n "$VLLM_NAMESPACE" -l "$VLLM_SERVICE_LABEL" --no-headers 2>/dev/null | wc -l | tr -d ' ')
  else
    status[pods]="empty"
    pod_count=0
    fail=1
    echo "[connect_probe] namespace=$VLLM_NAMESPACE label=$VLLM_SERVICE_LABEL 下没有 pod。检查 VLLM_NAMESPACE / VLLM_SERVICE_LABEL。" >&2
  fi
else
  status[pods]="skipped"
  pod_count=0
fi

# 3) prometheus
if curl -sS --max-time 5 "$PROM_URL/api/v1/query?query=up" 2>/dev/null | grep -q '"status":"success"'; then
  status[prom]="ok"
else
  status[prom]="unreachable"
  fail=1
  echo "[connect_probe] Prometheus ${PROM_URL} 不可达。检查 PROM_URL 或开 kubectl port-forward svc/prometheus 9090:9090。" >&2
fi

# 4) GPU 可达性（pod 内 nvidia-smi）
gpu_status="unknown"
gpu_count=0
if [ "${status[pods]}" = "ok" ]; then
  first_pod=$(kubectl get pods -n "$VLLM_NAMESPACE" -l "$VLLM_SERVICE_LABEL" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -n "$first_pod" ]; then
    if kubectl exec -n "$VLLM_NAMESPACE" "$first_pod" -- nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 | grep -qE '^[0-9]+'; then
      gpu_status="ok"
      gpu_count=$(kubectl exec -n "$VLLM_NAMESPACE" "$first_pod" -- nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
    else
      gpu_status="nvidia-smi-failed"
      echo "[connect_probe] pod $first_pod 内 nvidia-smi 失败，GPU 可能已掉链。" >&2
      fail=1
    fi
  fi
fi
status[gpu]="$gpu_status"

# 输出 JSON
cat <<JSON
{
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "kubectl_context": "${status[kubectl_context]}",
  "context": "${ctx}",
  "namespace": "${VLLM_NAMESPACE}",
  "service_label": "${VLLM_SERVICE_LABEL}",
  "pods": "${status[pods]}",
  "pod_count": ${pod_count},
  "prom_url": "${PROM_URL}",
  "prom": "${status[prom]}",
  "gpu": "${gpu_status}",
  "gpu_count": ${gpu_count}
}
JSON

exit $fail
