#!/usr/bin/env bash
# remediate_01.sh — Playbook 01：抢占级联 / KV 压力 的整改清单生成器
#
# 这个脚本只输出按 L1/L2/L3 分组的候选命令清单，绝不执行 mutation。
# L2/L3 都必须由 agent 展示精确 target/current/rollback 后逐条征得用户批准。
#
# 用法：
#   remediate_01.sh                       # 打印整改计划（dry-run）
#
# 这个生成器故意不渲染可直接执行的 mutation，也不提供通用候选值。
# agent 必须先解析实际部署单元、读取 current state，再为一条动作生成命令并询问批准。

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
  playbook: 01-preempt-cascade
  generated_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
  actions:
HEADER

emit_candidate L1 "read ${VLLM_NAMESPACE}/${VLLM_DEPLOYMENT} config, rollout and capacity baseline"
emit_candidate L2 "change one admission, sequence-budget or token-budget variable using the measured experiment matrix"
emit_candidate L2 "scale the resolved deployment unit to the capacity-calculated replica target"
emit_candidate L2 "change gateway admission using the validated KV/queue/SLO policy"
emit_candidate L3 "drain and restart the complete resolved deployment unit"

cat <<FOOTER

verification_query: |
  histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[2m])) by (le)) * 1000
verification_threshold_ms: "<required from approved service SLO>"
FOOTER
