# Playbook 03: GPU OOM / OOMKilled

## Symptom Reconfirm
- 任一满足：
  - Pod 状态 `OOMKilled`（exit code 137）
  - 日志里 `torch.cuda.OutOfMemoryError`
  - `DCGM_FI_DEV_FB_USED / total > 0.98` 且 `vllm:gpu_cache_usage_perc > 0.95`
- 排除：CPU OOM（看 `dmesg | grep -i oom`，目标进程是 Python 主进程不是 vllm worker → 走容器 memory limit）

## Triage Commands

```bash
# 1. 最近 OOM 事件
kubectl get events -n ${VLLM_NAMESPACE:-vllm} \
  --field-selector reason=OOMKilling \
  --sort-by='.lastTimestamp' \
  > "$INCIDENT_DIR/evidence/oom-events.txt"

# 2. 各 pod restart 次数（OOMKilled 会 restart）
kubectl get pods -n ${VLLM_NAMESPACE:-vllm} \
  -o custom-columns='NAME:.metadata.name,RESTART:.status.containerStatuses[*].restartCount,STATUS:.status.phase' \
  > "$INCIDENT_DIR/evidence/pod-restarts.txt"

# 3. GPU 显存历史趋势
curl -sS "$PROM_URL/api/v1/query_range?query=DCGM_FI_DEV_FB_USED&start=$(($(date +%s)-1800))&end=$(date +%s)&step=30" \
  > "$INCIDENT_DIR/evidence/fb-used-30m.json"

# 4. 当前 gpu_memory_utilization / max_num_seqs 配置
first_pod=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  "env | grep -E 'GPU_MEMORY_UTILIZATION|MAX_NUM_SEQS|KV_CACHE_DTYPE'" \
  > "$INCIDENT_DIR/evidence/vllm-env.txt"
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| 持续高位 `FB_used` 然后突刺到 100% | 激活内存峰值（长 prompt 进 prefill）撞天花板 |
| 仅启动后立刻 OOM | `--gpu-memory-utilization` 设太高（>0.92）+ allocator 碎片 |
| OOM 集中在某些 pod | 模型权重大小漂移（混了 FP16 / FP8 quantization 配错） |
| OOM 在切换 LoRA 时 | LoRA slot 没清干净，见 `08-lora-thrash` |

## Remediate

- **L2（直接做）**：
  - 降 `GPU_MEMORY_UTILIZATION` 0.95 → 0.85（留 10% 给峰值）
  - 改 `KV_CACHE_DTYPE=fp8`（如果硬件支持），KV 体积砍半
  - 调 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片
  - 降 `MAX_NUM_SEQS` 至 32 或更低
- **L3（弹确认）**：
  - 滚动重启加载新配置
  - 切到更高 SKU 的节点池（cost 影响）

## Verification

- 重启后 5 分钟内 `OOMKilled` 事件计数 = 0
- `DCGM_FI_DEV_FB_USED / total < 0.92` 持续 5 分钟
- `vllm:gpu_cache_usage_perc < 0.9`

## Long-term

- 模型上线前用 `--profile-num-tokens` 估激活峰值
- `gpu_memory_utilization` 留 ≥ 15% headroom
- DCGM 告警阈值 `FB_used > 0.92` 提前通知
- 见 `reference/checklist-prelaunch.md` 第 8 条

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md L54-76 -->
