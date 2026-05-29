# Playbook 08: LoRA 适配器抖动 (LoRA Adapter Thrash)

## Symptom Reconfirm
- 多租户 LoRA 部署下：
  - 同一 LoRA 反复 load / unload，`vllm:lora_loading_seconds` 高频出现
  - TTFT 在切换 LoRA 时尖刺（基础模型推理快，但 LoRA 切换慢）
  - LoRA slot 命中率（如果有指标）持续走低
- 排除：单 LoRA 单租户场景不会有这个问题

## Triage Commands

```bash
# 1. LoRA 加载次数 / 加载耗时
curl -sS "$PROM_URL/api/v1/query?query=sum(rate(vllm:lora_loading_seconds_count[5m]))" \
  > "$INCIDENT_DIR/evidence/lora-load-rate.json"
curl -sS "$PROM_URL/api/v1/query?query=histogram_quantile(0.99,sum(rate(vllm:lora_loading_seconds_bucket[5m]))by(le))" \
  > "$INCIDENT_DIR/evidence/lora-load-p99.json"

# 2. 当前 max_loras / max_lora_rank 配置
first_pod=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  'env | grep -E "MAX_LORAS|MAX_LORA_RANK|MAX_CPU_LORAS"' \
  > "$INCIDENT_DIR/evidence/lora-env.txt"

# 3. 流量里出现的 LoRA 数（用于和 max_loras 对比）
kubectl logs -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} \
  --tail=2000 | grep -oE 'lora_request.*name=[^ ]+' | sort -u | wc -l \
  > "$INCIDENT_DIR/evidence/distinct-loras-2k.txt"
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| 活跃 LoRA 数 > `MAX_LORAS` 而且远大于 | GPU slot 不够，LRU 一直在替换 |
| LoRA 加载耗时 p99 > 2 s | LoRA 文件存远端（NFS/OSS），加载慢 |
| 单 pod 上同时 max_loras + max_cpu_loras 都满 | CPU 缓存也满了，必须从存储拉 |

## Remediate

- **L1**：抓证据
- **L2（直接做）**：
  - 调大 `MAX_LORAS`（典型 4 → 8 或 16，取决于显存）
  - 调大 `MAX_CPU_LORAS`（CPU 侧缓存，命中减少远端拉取）
  - 把 LoRA 仓库镜像到本地 SSD
- **L3（弹确认）**：
  - 改路由策略：相同 LoRA 的请求 sticky 到同一 pod（降低切换频率）—— 影响：负载不均
  - 给热门 LoRA 起独立 pod 池 —— 影响：成本

## Verification

- LoRA 加载速率 5 分钟内降 50%+
- 切换 LoRA 时 TTFT 尖刺消失（p99 回归基线 ± 20%）

## Long-term

- 多租户 LoRA 上线前估算并发活跃 LoRA 数，设 `MAX_LORAS` 至少 = p95 并发活跃数
- LoRA 仓库走 PV/PVC 本地缓存
- LoRA 大小标准化（不同 rank 混着用会让显存预算更难算）
- 见 `09-advanced-features/04-lora-serving.md` 详解

<!-- source: ../../09-advanced-features/04-lora-serving.md L90-140 -->
