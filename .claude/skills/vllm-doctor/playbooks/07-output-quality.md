# Playbook 07: 输出质量异常 (Output Quality Drop)

## Symptom Reconfirm
- 任一满足：
  - `format_compliance_rate < 0.9`（要求 JSON / function call 等结构化输出的场景）
  - 用户反馈 thumbs-down 比例突增
  - EOS token 命中率骤降（请求被 max_tokens 截断率上升）
  - 输出含非 ASCII / 控制字符 / 重复词组
- 必须排除：客户端 prompt 变了（先看请求样本，对比基线 prompt 模板）

## Triage Commands

```bash
# 1. 拿 30 条最近请求样本（如果开了 OTel + 日志归档）
# （下面是占位，按你们的日志栈调整）
kubectl logs -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} \
  --tail=500 | grep -E 'prompt|completion' | head -30 \
  > "$INCIDENT_DIR/evidence/sample-completions.txt"

# 2. 最近一次模型 / quantization config 变更
kubectl rollout history deploy/${VLLM_DEPLOYMENT:-vllm} -n ${VLLM_NAMESPACE:-vllm} \
  > "$INCIDENT_DIR/evidence/vllm-rollout.txt"

# 3. 当前 quantization / dtype 配置
first_pod=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  'env | grep -E "QUANTIZATION|KV_CACHE_DTYPE|DTYPE"' \
  > "$INCIDENT_DIR/evidence/dtype-env.txt"

# 4. 权重 hash（确认没装到错的模型）
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  'find $MODEL_PATH -name "*.safetensors" -exec sha256sum {} \; 2>/dev/null | head -3' \
  > "$INCIDENT_DIR/evidence/weight-hashes.txt"
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| 输出全是乱码 / 非 ASCII | 权重加载错（混了 FP8 校准集不匹配的版本），或 tokenizer 错配 |
| 输出截断且 EOS 不出 | sampling 参数错（`stop` 没生效 / `max_tokens` 太小） |
| 仅 JSON 类输出格式错 | guided decoding / outlines / xgrammar 配错 |
| 输出复读 | repetition_penalty / no_repeat_ngram_size 没设，或模型受指令污染 |
| 最近改了 quantization | FP8 / AWQ 校准漂移，需重新校准 |

## Remediate

- **L1**：抓 30 条样本，跑离线 eval（PPL / 简单格式合规率）
- **L2（直接做）**：
  - sampling 参数回归基线（`temperature`、`top_p`、`stop`）
  - 结构化输出场景启用 `guided_json` 或 `xgrammar`
- **L3（弹确认）**：
  - 回滚到上一个 image（影响：可能丢失新能力，但能立刻恢复质量）
  - 切换 quantization（FP8 → FP16）—— 影响：吞吐降 20-30%

## Verification

- 用 50 条标准 prompt 跑 spot check，`format_compliance_rate ≥ 0.95`
- 用户 thumbs-down 比例 3 小时窗口回归基线

## Long-term

- 上线前必做：50 条 golden prompt 的 PPL/eval baseline
- 每次 quantization 变更必须重新校准并对比 PPL
- canary 部署阶段加自动质量门：`format_compliance < 0.95` 自动 rollback
- 见 `reference/checklist-prelaunch.md` 第 13 条

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md L27-28 + 07-incident-playbook.md case 7 -->
