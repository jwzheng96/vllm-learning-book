# Playbook 03: GPU OOM / OOMKilled

## Symptom Reconfirm
- 至少一项**明确 OOM 证据**：
  - Pod 状态 `OOMKilled`（exit code 137）
  - 日志里 `torch.cuda.OutOfMemoryError`
- 高 DCGM 显存占用或高 `vllm:kv_cache_usage_perc` 只能作为相关证据，不能单独确认 OOM。
- 用 Pod termination reason、container memory 指标和 runtime log 区分 GPU OOM、容器 CPU memory limit 与节点压力；不要默认读取宿主机 `dmesg`。

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
| OOM 与长 prompt/prefill 时间对齐 | 激活峰值是候选原因；用请求长度与 profiler 复核 |
| 仅启动后立刻 OOM | 权重/配置/可用显存不匹配；分别核对，不预设 allocator 碎片 |
| OOM 集中在某些 pod | 核对 GPU/MIG、模型 digest、参数和其他进程的差异 |
| OOM 与 LoRA load 时间对齐 | adapter 内存预算是候选原因，见 `08-lora-thrash` |

## Remediate

- **L2（需逐命令批准）**：根据复现实验一次只改一个变量，例如降低 admission/seq budget、收紧请求长度，或在已验证质量与 backend 支持后测试 KV dtype。候选值来自目标模型的容量测试。
- **L3（需逐命令批准）**：drain 后滚动重启或切换节点池；先确认稳定池容量、cold-start 与回滚命令。

## Verification

- 覆盖事故请求长度和并发的观测窗口内没有新 OOM 证据
- DCGM 显存与 `vllm:kv_cache_usage_perc` 回到该模型池压测得到的安全区间
- TTFT/TPOT/质量与吞吐没有因处置越过批准 gate

## Long-term

- 上线前用真实长度/并发矩阵测量激活与 KV 峰值
- `gpu_memory_utilization` 与 admission headroom 由故障点反推，并记录模型/硬件适用范围
- DCGM 告警阈值来自本节点池基线，并和 OOM/重启证据联合判断
- 见 `reference/checklist-prelaunch.md` 第 8 条

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md -->
