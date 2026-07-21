# Playbook 02: NCCL Hang

## Symptom Reconfirm
- **必须同时满足**：
  - `sum(rate(vllm:generation_tokens_total[1m])) ≈ 0`
  - `sum(vllm:num_requests_running) > 0`
  - 持续超过该服务已定义的无进展窗口
- **排除**：
  - `num_requests_running == 0` → 没人请求，不是 hang
  - `request_inference_time` 还在涨 → 是单请求慢，不是 NCCL 卡

## Triage Commands

```bash
# 全 worker py-spy dump + NVLink 错误 + IB 状态
bash "$CLAUDE_SKILL_DIR/scripts/nccl_diag.sh" \
  "$INCIDENT_DIR/evidence/nccl"

# 看节点 GPU 利用率 —— pmon 显示长时间 0% util 是 hang 的旁证
for pod in $(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} \
    -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} \
    -o name); do
  kubectl exec -n ${VLLM_NAMESPACE:-vllm} $pod -- nvidia-smi pmon -c 5 \
    >> "$INCIDENT_DIR/evidence/gpu-pmon.txt"
done
```

## Root Cause 判定

读 `evidence/nccl/summary.txt`：

| 现象 | 根因 |
| --- | --- |
| 全部 worker stack 同时停在 collective，step/token 无进展 | NCCL/collective hang 假设较强；继续核对 rank 与 transport log |
| 仅 1 个 worker 不在 collective，其他 rank 等待 | 该 rank 的 GPU、进程或输入路径可能是 straggler |
| 硬件错误 counter 与故障时间对齐且只集中于节点 | 硬件/链路问题假设较强，交由平台按硬件流程确认 |
| timeout/monitor 配置与批准基线不同 | 配置漂移；不能仅因某变量未设就断定根因 |
| mesh 变更/重启与故障时间对齐 | 网络路径是候选原因；必须用连接与路由证据确认 |

## Remediate

`scripts/remediate_02.sh` 只生成计划。每项动作展示当前状态、完整部署单元、blast radius、回滚与停止条件后逐条批准：

- **L2（需逐命令批准）**：只在目标 NCCL/PyTorch 版本官方语义和 staging 演练确认后，修改 timeout/monitoring；环境变量名和值都不得从本 playbook 硬编码。
- **L3（需逐命令批准）**：从路由池移除并重建**完整并行部署单元**；部署单元可能是单 Pod 多 GPU 或多 Pod group，先从实际拓扑解析。taint/cordon 节点是独立平台变更，另行批准。

## Verification

重新 warmup 后，覆盖批准的验证请求和观测窗口：
- generation token rate 与 step progress 恢复到事故前同类负载基线
- running 请求可以完成，所有 rank 无持续 collective wait

任一不满足且重启已做过 → `NEEDS_HUMAN`，把 `evidence/nccl/` 整包给硬件 / 平台团队。

## Long-term

- 将经目标版本验证的 timeout/monitoring 语义写进基线并做故障演练
- DCGM/平台持续监控硬件错误；阈值和归零/累积语义由硬件运维标准定义
- 根据实际 transport 和动态端口范围配置网络策略，不照抄固定端口
- 见 `reference/checklist-prelaunch.md` 第 5、9 条；NCCL 环境速查见 `reference/nccl-env.md`

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md + 07-incident-playbook.md case 2 -->
