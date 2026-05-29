# Playbook 06: 冷启动 / readiness 慢

## Symptom Reconfirm
- 新 pod 调度后超过 5 分钟还没 Ready，或 Ready 后 TTFT p99 仍异常高 ≥ 60 s
- 历史正常冷启 60-180s；本次明显偏长
- 排除：模型权重确实大（70B+）→ 5-10 分钟是正常的，不算故障

## Triage Commands

```bash
# 1. 看 pod 事件 + 容器启动日志
PEND_POD=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} \
  -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} \
  --field-selector=status.phase!=Running -o name | head -1)
if [ -n "$PEND_POD" ]; then
  kubectl describe -n ${VLLM_NAMESPACE:-vllm} $PEND_POD \
    > "$INCIDENT_DIR/evidence/pending-pod-describe.txt"
  kubectl logs -n ${VLLM_NAMESPACE:-vllm} $PEND_POD --tail=200 \
    > "$INCIDENT_DIR/evidence/pending-pod.log"
fi

# 2. torch.compile 缓存路径是否挂载且非空（缺失会重新编译 60-180s）
first_pod=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  'ls -la $VLLM_TORCH_COMPILE_CACHE_DIR 2>/dev/null | head' \
  > "$INCIDENT_DIR/evidence/compile-cache.txt"

# 3. 模型权重存放位置：本地盘 / NFS / OSS / HF
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  'env | grep -E "MODEL_PATH|HF_HOME|VLLM_TORCH_COMPILE_CACHE_DIR"' \
  > "$INCIDENT_DIR/evidence/model-paths.txt"
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| `pending-pod-describe.txt` 显示 `Insufficient nvidia.com/gpu` | 节点池容量不够，HPA 扩了但无节点接 |
| 日志卡在 `torch.compile` | 编译缓存没挂载 / 缓存路径为空，每次重编 |
| 日志卡在 `Downloading model` | 模型走的是 HF / S3 拉，第一次慢 |
| 日志卡在 `Loading safetensors` 且超长 | 权重在 NFS，I/O 慢 |
| Ready 但 TTFT 高 | CUDA Graph 还没 capture 完，正常的 warmup 期 |

## Remediate

- **L1**：抓证据
- **L2（直接做）**：
  - 给 deployment 加 `VLLM_TORCH_COMPILE_CACHE_DIR=/persistent/torch-cache`（PV/PVC 已挂时）
  - `readinessProbe.initialDelaySeconds` 调到 600s（70B 模型）
  - 模型存储改用本地 SSD 镜像（如果原是 NFS）
- **L3（弹确认）**：
  - 提前 warm 节点池：`kubectl scale lws/${VLLM_LWS:-vllm} --replicas=+N`（影响：成本）
  - 改用 baked-in image（模型权重打进镜像，第一次拉镜像慢但启动飞快）—— 影响：镜像变 30-50GB

## Verification

- 新建 pod 从 `Pending` 到 `Ready` 时间 < 历史 p95 × 1.5
- Ready 后 60s 内 TTFT p99 < `$TTFT_SLO_MS`

## Long-term

- KEDA `minReplicaCount >= 2`、`cooldownPeriod >= 300s`，避免 scale-to-zero
- 编译缓存放 NFS 或 PVC 共享，所有 pod 复用
- 起容量预留（节点池 over-provision 10-20%）
- 见 `reference/checklist-prelaunch.md` 第 1、2、3 条

<!-- source: ../../08-production-deployment/04-autoscaling-and-capacity.md L253-267 -->
