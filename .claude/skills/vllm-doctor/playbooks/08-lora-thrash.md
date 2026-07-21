# Playbook 08: LoRA 适配器抖动 (LoRA Adapter Thrash)

## Symptom Reconfirm
- 多租户 LoRA 部署下：
  - 经审计的控制面事件显示同一 adapter 反复 load/unload
  - TTFT 尖刺与 adapter 切换时间对齐
  - `vllm:lora_requests_info` 显示 waiting/running adapter 压力；数据并行场景先注意源码警告，该 gauge 可能误导
- 排除：单 LoRA 单租户场景不会有这个问题

## Triage Commands

```bash
# 1. 当前源码只提供 LoRA request info gauge，不提供通用 loading latency
# histogram。加载事件/耗时从实际 adapter 控制面或网关审计指标读取。
curl -sS -G --data-urlencode 'query=vllm:lora_requests_info' \
  "$PROM_URL/api/v1/query" > "$INCIDENT_DIR/evidence/lora-info.json"

# 2. 当前 max_loras / max_lora_rank 配置
first_pod=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  'env | grep -E "MAX_LORAS|MAX_LORA_RANK|MAX_CPU_LORAS"' \
  > "$INCIDENT_DIR/evidence/lora-env.txt"

# 3. 活跃 adapter 数从经脱敏的路由/控制面指标读取，不通过生产日志
# 猜测；若没有该信号，根因标记 unconfirmed。
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| waiting adapter 与控制面 load/unload 同时反复 | slot/admission 压力假设较强 |
| 加载耗时与存储延迟时间对齐 | 制品存储是候选原因；用 I/O 数据确认 |
| 实际活跃 adapter 超过已测 GPU/CPU 容量 | 容量不足；需核对配置与 adapter rank/size |

## Remediate

- **L1（只读）**：抓证据
- **L2（需逐命令批准）**：按 adapter rank/size 和目标硬件容量实验调整 GPU/CPU slot，或更改制品缓存；先验证 OOM、质量、启动和一致性。
- **L3（需逐命令批准）**：
  - 改路由策略：相同 LoRA 的请求 sticky 到同一 pod（降低切换频率）—— 影响：负载不均
  - 给热门 LoRA 起独立 pod 池 —— 影响：成本

## Verification

- 覆盖代表 adapter 工作集后，重复 load/unload 事件回到批准基线
- TTFT、OOM、质量和负载均衡全部满足 gate

## Long-term

- 多租户 LoRA 上线前用 adapter rank/size 与并发工作集做容量实验，再设 slot 与 admission
- LoRA 仓库走 PV/PVC 本地缓存
- LoRA 大小标准化（不同 rank 混着用会让显存预算更难算）
- 见 `09-advanced-features/04-lora-serving.md` 详解

<!-- source: ../../09-advanced-features/04-lora-serving.md -->
