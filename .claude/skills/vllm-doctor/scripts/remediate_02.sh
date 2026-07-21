#!/usr/bin/env bash
# remediate_02.sh — Playbook 02：NCCL Hang 的整改清单生成器
#
# 用法同 remediate_01.sh：只生成计划，不执行 mutation。
#
# 这个生成器不假设部署一定是 LeaderWorkerSet，也不硬编码 NCCL/PyTorch
# 环境变量。实际变量和值必须来自目标版本官方文档和 staging 故障演练。

set -u
set -o pipefail

VLLM_NAMESPACE="${VLLM_NAMESPACE:-vllm}"
VLLM_DEPLOYMENT="${VLLM_DEPLOYMENT:-vllm}"

emit_candidate() {
  local level="$1"; shift
  local action="$*"
  cat <<EOF
- level: ${level}
  candidate: ${action}
  command: "<render only after exact target/current state and explicit approval>"
  rollback: "<restore captured current state>"
  stop_condition: "<derive from service SLO and incident evidence>"
EOF
}

cat <<HEADER
plan:
  playbook: 02-nccl-hang
  generated_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
  actions:
HEADER

emit_candidate L1 "read evidence/nccl, deployment topology, rank mapping and current communication environment"
emit_candidate L2 "change only version-verified collective timeout/monitoring configuration after staging validation"
emit_candidate L3 "drain and rebuild the complete resolved parallel deployment unit"
emit_candidate L3 "cordon or taint an evidence-confirmed suspect node under the platform hardware procedure"

cat <<FOOTER

verification_query: |
  sum(rate(vllm:generation_tokens_total[1m]))
verification_threshold_min: "<required from same-workload healthy baseline>"
verification_must_also: |
  vllm:num_requests_running > 0
FOOTER
