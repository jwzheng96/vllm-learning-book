# Playbook 02: NCCL Hang

## Symptom Reconfirm
- **必须同时满足**：
  - `sum(rate(vllm:generation_tokens_total[1m])) ≈ 0`
  - `sum(vllm:num_requests_running) > 0`
  - 持续 ≥ 60s
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
| 全部 worker 都 `NCCL_WAIT_LIKELY` | 真 NCCL 集合通信卡死（rank 间消息丢失或拓扑断） |
| 仅 1 个 worker `NOT_IN_NCCL`，其他都 `NCCL_WAIT_LIKELY` | 那 1 个 worker 慢（OOM 边缘 / 单 GPU 错误 / 该节点 thermal throttle），其他在等它 |
| 多 pod NVLink CRC/Replay 错误 NON-ZERO | 硬件问题（NVLink 通道劣化），需换节点 |
| `nccl-env.txt` 里 `NCCL_TIMEOUT` 未设 | 上次卡的就是这次，看门狗缺失，整改先注入 env |
| 同一时刻 mesh sidecar 重启 | 服务网格代理把 NCCL 端口路由断了，参考 `06-reliability-and-failure-modes.md:105` |

## Remediate

按 `scripts/remediate_02.sh` 输出执行。关键动作：

- **L2（直接做）**：
  - 注入 `NCCL_TIMEOUT=60 NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ENABLE_MONITORING=1`
  - 让"下次再卡 60s 就 crash"，K8s 自动重启
- **L3（弹确认）**：
  - 重启整个 LWS group（**必须整组重启，不是单 pod**——NCCL group 跨 rank，单删一个会让剩下的继续等）
  - 如有 `BAD_NODE`，taint 该节点 + 报修硬件

## Verification

整改后 120s 起（留足重新初始化时间），连续 3 个采样点：
- `sum(rate(vllm:generation_tokens_total[1m])) > 1.0`
- `sum(vllm:num_requests_running) > 0`

任一不满足且重启已做过 → `NEEDS_HUMAN`，把 `evidence/nccl/` 整包给硬件 / 平台团队。

## Long-term

- `NCCL_TIMEOUT=60`、`NCCL_BLOCKING_WAIT=1` 写进基线配置（默认就有）
- DCGM 持续监控 NVLink CRC / Replay，CRC > 0 触发告警
- 网格 sidecar 不要拦截 NCCL 端口（一般是 6000-6100 + IB 端口）
- 见 `reference/checklist-prelaunch.md` 第 5、9 条；NCCL 环境速查见 `reference/nccl-env.md`

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md L79-111 + 07-incident-playbook.md case 2 -->
