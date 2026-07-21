# Playbook 06: 冷启动 / readiness 慢

## Symptom Reconfirm
- 新实例从 scheduled 到 Ready，或 Ready 到满足放流条件的时长，显著越过同模型/硬件/缓存状态的历史基线
- 分别记录镜像拉取、调度、权重读取、compile/capture、warmup；不要用模型参数量推断固定“正常分钟数”

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

# 2. compile/cache 路径是否存在；非空不代表与当前版本兼容
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
| compile 阶段明显慢于同版本基线 | cache miss/不兼容或 compile workload 改变，需比对 digest 与日志 |
| 下载阶段慢 | 远端制品、网络或鉴权是候选原因 |
| 权重读取阶段慢 | 存储吞吐/并发是候选原因，需用 I/O 证据确认 |
| Ready 但 TTFT 高 | readiness 未覆盖完整 warmup，或真实流量形态未预热 |

## Remediate

- **L1（只读）**：抓证据
- **L2（需逐命令批准）**：根据分段证据调整 readiness、缓存挂载或制品分发；候选值来自目标模型启动分布，并验证缓存的版本隔离、权限和一致性。
- **L3（需逐命令批准）**：预热节点/实例池或改变镜像制品布局；先计算容量、镜像分发成本、安全扫描和回滚。

## Verification

- 新建 pod 从 `Pending` 到 `Ready` 时间 < 历史 p95 × 1.5
- Ready 后覆盖批准 warmup 与请求矩阵，TTFT 满足该服务 SLO

## Long-term

- 是否保留 warm replica、cooldown 与 scale-to-zero 由冷启动分布和成本目标共同决定
- 编译缓存只有在版本键、并发写和存储语义验证后才能共享
- 节点容量预留按扩容 SLO 与供应时间计算，不套固定比例
- 见 `reference/checklist-prelaunch.md` 第 1、2、3 条

<!-- source: ../../08-production-deployment/04-autoscaling-and-capacity.md -->
