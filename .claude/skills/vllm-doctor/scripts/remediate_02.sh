#!/usr/bin/env bash
# remediate_02.sh — Playbook 02：NCCL Hang 的整改清单生成器
#
# 用法同 remediate_01.sh。
#
# 关键策略：
#   全部 worker 都卡 → L3 重启整个 LeaderWorkerSet（不是单 pod，因为 NCCL group 必须全员重建）
#   单 worker 卡 / NVLink CRC 高 → L3 taint 节点 + 重建
#   预防：L2 注入 NCCL_TIMEOUT / NCCL_BLOCKING_WAIT 让下次"卡 → crash → 自动重启"

set -u
set -o pipefail

VLLM_NAMESPACE="${VLLM_NAMESPACE:-vllm}"
VLLM_DEPLOYMENT="${VLLM_DEPLOYMENT:-vllm}"
VLLM_LWS="${VLLM_LWS:-vllm}"

# 由调用方提供（从 nccl_diag.sh 输出读取）
BAD_NODE="${BAD_NODE:-}"          # NVLink CRC 高的节点名
HUNG_POD_LWS="${HUNG_POD_LWS:-}"  # 卡死的 LWS 实例名

mode="${1:-plan}"

emit() {
  local level="$1"; shift
  local rollback="$1"; shift
  local cmd="$*"
  if [ "$mode" = "plan" ]; then
    cat <<EOF
- level: ${level}
  command: ${cmd}
  rollback: ${rollback}
EOF
  elif [ "$mode" = "--apply-l2" ] && [ "$level" = "L2" ]; then
    echo "[apply L2] $cmd"
    eval "$cmd"
  fi
}

cat <<HEADER
plan:
  playbook: 02-nccl-hang
  generated_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
  actions:
HEADER

# ---- L1：抓证据（已在 nccl_diag.sh 完成，这里仅提示）----
emit L1 "(none)" "ls -la \$INCIDENT_DIR/evidence/nccl/ 2>/dev/null || echo '先跑 nccl_diag.sh'"

# ---- L2：注入 NCCL 看门狗 env（下次再卡能 60s 内 crash）----
emit L2 \
  "kubectl set env deploy/${VLLM_DEPLOYMENT} -n ${VLLM_NAMESPACE} NCCL_TIMEOUT- NCCL_BLOCKING_WAIT- TORCH_NCCL_ENABLE_MONITORING-" \
  "kubectl set env deploy/${VLLM_DEPLOYMENT} -n ${VLLM_NAMESPACE} NCCL_TIMEOUT=60 NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ENABLE_MONITORING=1"

# ---- L3：重启卡死的 LWS 实例（不是单 pod，必须整组）----
if [ -n "$HUNG_POD_LWS" ]; then
  emit L3 \
    "(无 — 重启完只能等就绪)" \
    "kubectl delete pods -n ${VLLM_NAMESPACE} -l leaderworkerset.sigs.k8s.io/group-key=${HUNG_POD_LWS}  # 影响：该组所有 leader+worker 同时重启"
else
  emit L3 \
    "(无)" \
    "kubectl rollout restart deploy/${VLLM_DEPLOYMENT} -n ${VLLM_NAMESPACE}  # 影响：所有 replica 滚动，期间整体容量 -1"
fi

# ---- L3：硬件嫌疑 → taint 节点 ----
if [ -n "$BAD_NODE" ]; then
  emit L3 \
    "kubectl taint nodes ${BAD_NODE} nccl-bad-:NoSchedule" \
    "kubectl taint nodes ${BAD_NODE} nccl-bad=:NoSchedule  # 影响：调度系统不再往该节点放新 pod；需要后续报修硬件"
fi

cat <<FOOTER

verification_query: |
  sum(rate(vllm:generation_tokens_total[1m]))
verification_threshold_min: 1.0
verification_must_also: |
  vllm:num_requests_running > 0
FOOTER
