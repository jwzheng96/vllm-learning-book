#!/usr/bin/env bash
# remediate_01.sh — Playbook 01：抢占级联 / KV 压力 的整改清单生成器
#
# 这个脚本本身不直接执行 L3 动作；它输出按 L1/L2/L3 分组的命令清单，
# agent（Claude）逐条读、L1/L2 直接 bash 跑、L3 用 AskUserQuestion 弹确认后再跑。
#
# 用法：
#   remediate_01.sh                       # 打印整改计划（dry-run）
#   remediate_01.sh --apply-l2            # 直接执行所有 L2 动作（agent 用）
#
# 环境：
#   VLLM_NAMESPACE, VLLM_DEPLOYMENT（默认 vllm），VLLM_LWS（LeaderWorkerSet 名）
#   MAX_NUM_SEQS_NEW（默认 32），MAX_NUM_BATCHED_TOKENS_NEW（默认 2048）
#   GATEWAY_DEPLOY（默认 vllm-gateway，可省略，没有则跳过 gateway 动作）

set -u
set -o pipefail

VLLM_NAMESPACE="${VLLM_NAMESPACE:-vllm}"
VLLM_DEPLOYMENT="${VLLM_DEPLOYMENT:-vllm}"
VLLM_LWS="${VLLM_LWS:-vllm}"
MAX_NUM_SEQS_NEW="${MAX_NUM_SEQS_NEW:-32}"
MAX_NUM_BATCHED_TOKENS_NEW="${MAX_NUM_BATCHED_TOKENS_NEW:-2048}"
GATEWAY_DEPLOY="${GATEWAY_DEPLOY:-}"

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
  playbook: 01-preempt-cascade
  generated_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
  actions:
HEADER

# ---- L1：只读基线 ----
emit L1 "(none)" "kubectl describe deploy/${VLLM_DEPLOYMENT} -n ${VLLM_NAMESPACE} | grep -E 'MAX_NUM_SEQS|GPU_MEMORY_UTILIZATION|KV_CACHE_DTYPE' || true"

# ---- L2：受控扰动 ----
emit L2 \
  "kubectl set env deploy/${VLLM_DEPLOYMENT} MAX_NUM_SEQS-" \
  "kubectl set env deploy/${VLLM_DEPLOYMENT} -n ${VLLM_NAMESPACE} MAX_NUM_SEQS=${MAX_NUM_SEQS_NEW}"

emit L2 \
  "kubectl set env deploy/${VLLM_DEPLOYMENT} MAX_NUM_BATCHED_TOKENS-" \
  "kubectl set env deploy/${VLLM_DEPLOYMENT} -n ${VLLM_NAMESPACE} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS_NEW}"

# 增加 replica（用 LWS scale 而不是 deployment scale，避免破坏 leader/worker 拓扑）
emit L2 \
  "kubectl scale lws/${VLLM_LWS} -n ${VLLM_NAMESPACE} --replicas=\$(kubectl get lws/${VLLM_LWS} -n ${VLLM_NAMESPACE} -o jsonpath='{.spec.replicas}')" \
  "kubectl scale lws/${VLLM_LWS} -n ${VLLM_NAMESPACE} --replicas=\$((\$(kubectl get lws/${VLLM_LWS} -n ${VLLM_NAMESPACE} -o jsonpath='{.spec.replicas}') + 2))"

# 网关 admission control（KV > 0.85 → 429）
if [ -n "$GATEWAY_DEPLOY" ]; then
  emit L2 \
    "kubectl set env deploy/${GATEWAY_DEPLOY} -n ${VLLM_NAMESPACE} ADMISSION_KV_THRESHOLD-" \
    "kubectl set env deploy/${GATEWAY_DEPLOY} -n ${VLLM_NAMESPACE} ADMISSION_KV_THRESHOLD=0.85"
fi

# ---- L3：高破坏性，需 AskUserQuestion 确认 ----
emit L3 \
  "kubectl rollout undo deploy/${VLLM_DEPLOYMENT} -n ${VLLM_NAMESPACE}" \
  "kubectl rollout restart deploy/${VLLM_DEPLOYMENT} -n ${VLLM_NAMESPACE}  # 影响：所有 replica 滚动重启，期间触发冷启动"

emit L3 \
  "(none — 缩回去会重新引发抢占)" \
  "kubectl scale lws/${VLLM_LWS} -n ${VLLM_NAMESPACE} --replicas=\$((\$(kubectl get lws/${VLLM_LWS} -n ${VLLM_NAMESPACE} -o jsonpath='{.spec.replicas}') + 4))  # 影响：更激进扩容 +4"

cat <<FOOTER

verification_query: |
  histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[2m])) by (le)) * 1000
verification_threshold_ms: ${TTFT_SLO_MS:-2000}
FOOTER
