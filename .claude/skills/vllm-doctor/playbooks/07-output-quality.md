# Playbook 07: 输出质量异常 (Output Quality Drop)

## Symptom Reconfirm
- 任一满足：
  - 部署自定义的格式合规率越过预先批准的质量 gate
  - 用户反馈率相对同类时段基线突增
  - EOS token 命中率骤降（请求被 max_tokens 截断率上升）
  - 不符合业务语言/字符策略、控制字符或异常重复显著上升
- 必须先核对请求模板、模型/tokenizer/template revision 和 sampling 参数；非 ASCII 本身不是质量故障。

## Triage Commands

```bash
# 1. 使用版本化、无用户敏感数据的 golden fixture 离线复现。
# 不从生产日志抓 prompt/output；若组织另有批准的数据流程，按其审计执行。

# 2. 最近一次模型 / quantization config 变更
kubectl rollout history deploy/${VLLM_DEPLOYMENT:-vllm} -n ${VLLM_NAMESPACE:-vllm} \
  > "$INCIDENT_DIR/evidence/vllm-rollout.txt"

# 3. 当前 quantization / dtype 配置
first_pod=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  'env | grep -E "QUANTIZATION|KV_CACHE_DTYPE|DTYPE"' \
  > "$INCIDENT_DIR/evidence/dtype-env.txt"

# 4. 从发布 manifest 读取 image/model/tokenizer/template digest；不要在服务
# Pod 内遍历权重文件，也不要把局部文件 hash 当作完整 provenance。
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| golden 输出乱码或违反语言策略 | model/tokenizer/template 错配是候选原因；按 manifest 二分 |
| 输出截断且 EOS 不出 | sampling 参数错（`stop` 没生效 / `max_tokens` 太小） |
| 仅 JSON 类输出格式错 | guided decoding / outlines / xgrammar 配错 |
| 输出复读 | 请求参数、模板、模型或 backend 都可能导致，不能直接归因于某个 penalty |
| 最近改了 quantization/backend | 变更与退化相关；用稳定版本 A/B 和任务级质量确认因果 |

## Remediate

- **L1（只读）**：用批准的 golden fixture 跑离线 eval，并比较 release manifest
- **L2（需逐命令批准）**：把 sampling/template/structured-output 配置切回已验证基线；每次只改一个变量并提供回滚
- **L3（需逐命令批准）**：
  - 回滚到上一个 image（影响：可能丢失新能力，但能立刻恢复质量）
  - 切换 quantization/backend——影响和容量变化必须从目标模型压测给出

## Verification

- 运行版本化 golden/质量集，全部预定义 gate 通过
- 覆盖业务反馈窗口后，线上质量信号回到批准基线

## Long-term

- 上线前维护覆盖业务切片的版本化、无敏感数据 golden/eval baseline
- 每次 quantization 变更必须重新校准并对比 PPL
- canary 质量 gate 失败时停止放量；rollback 仍是需要显式批准的集群变更
- 见 `reference/checklist-prelaunch.md` 第 13 条

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md + 07-incident-playbook.md case 7 -->
