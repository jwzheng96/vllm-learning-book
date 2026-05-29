#!/usr/bin/env bash
# nccl_diag.sh — 全 worker py-spy dump + NCCL 环境 / NVLink CRC 抓取
# 用于 playbook 02-nccl-hang 的 Triage 阶段
# 输出目录通过 $1 传入（一般是 $INCIDENT_DIR/evidence/nccl/）
set -u
set -o pipefail

VLLM_NAMESPACE="${VLLM_NAMESPACE:-vllm}"
VLLM_SERVICE_LABEL="${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm}"

OUT_DIR="${1:-./evidence/nccl}"
mkdir -p "$OUT_DIR"

# 取出所有 vllm worker pod
pods=$(kubectl get pods -n "$VLLM_NAMESPACE" -l "$VLLM_SERVICE_LABEL" -o jsonpath='{.items[*].metadata.name}')
if [ -z "$pods" ]; then
  echo "[nccl_diag] 没找到 pod，先看 connect_probe 输出" >&2
  exit 1
fi

for pod in $pods; do
  echo "==> $pod" | tee -a "$OUT_DIR/summary.txt"
  pod_dir="$OUT_DIR/$pod"
  mkdir -p "$pod_dir"

  # 找 worker 进程的 PID（vllm 主进程通常是 python -m vllm 或 multiprocessing worker）
  pids=$(kubectl exec -n "$VLLM_NAMESPACE" "$pod" -- bash -lc "ps -eo pid,cmd --no-headers | awk '/vllm|engine_core|model_executor/ && !/awk/ {print \$1}'" 2>/dev/null || true)

  if [ -z "$pids" ]; then
    echo "  (no vllm-like PID found)" | tee -a "$OUT_DIR/summary.txt"
    continue
  fi

  for pid in $pids; do
    # py-spy dump 不会 stop 进程，安全
    kubectl exec -n "$VLLM_NAMESPACE" "$pod" -- bash -lc "py-spy dump --pid $pid" \
      > "$pod_dir/pyspy-$pid.txt" 2>&1 || echo "  py-spy on $pid failed (need cap_sys_ptrace)" >> "$OUT_DIR/summary.txt"

    # 关键栈匹配
    if grep -Eq 'ncclAllReduce|c10d::ProcessGroupNCCL|work\.wait|cuStreamSynchronize' "$pod_dir/pyspy-$pid.txt" 2>/dev/null; then
      echo "  PID $pid: NCCL_WAIT_LIKELY" | tee -a "$OUT_DIR/summary.txt"
    else
      echo "  PID $pid: NOT_IN_NCCL" | tee -a "$OUT_DIR/summary.txt"
    fi
  done

  # NVLink 错误计数
  kubectl exec -n "$VLLM_NAMESPACE" "$pod" -- bash -lc "nvidia-smi nvlink -e 2>/dev/null || true" \
    > "$pod_dir/nvlink-errors.txt" 2>&1
  if grep -Eq '(CRC|Replay|Recovery) Error.*: [1-9]' "$pod_dir/nvlink-errors.txt" 2>/dev/null; then
    echo "  NVLink: CRC/Replay errors NON-ZERO (硬件嫌疑)" | tee -a "$OUT_DIR/summary.txt"
  fi

  # InfiniBand 状态（如果有）
  kubectl exec -n "$VLLM_NAMESPACE" "$pod" -- bash -lc "ibstat 2>/dev/null || true" \
    > "$pod_dir/ibstat.txt" 2>&1

  # 当前 NCCL 环境
  kubectl exec -n "$VLLM_NAMESPACE" "$pod" -- bash -lc "env | grep -E '^NCCL_' | sort" \
    > "$pod_dir/nccl-env.txt" 2>&1
done

# 汇总分类
all_wait=$(grep -c "NCCL_WAIT_LIKELY" "$OUT_DIR/summary.txt" || true)
none_wait=$(grep -c "NOT_IN_NCCL" "$OUT_DIR/summary.txt" || true)
nvlink_bad=$(grep -c "NVLink: CRC/Replay errors NON-ZERO" "$OUT_DIR/summary.txt" || true)

cat <<JSON
{
  "out_dir": "$OUT_DIR",
  "pods_scanned": $(echo "$pods" | wc -w | tr -d ' '),
  "pids_in_nccl_wait": ${all_wait},
  "pids_not_in_nccl": ${none_wait},
  "nvlink_errors_nonzero_pods": ${nvlink_bad}
}
JSON
