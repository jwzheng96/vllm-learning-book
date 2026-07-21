# Playbook 05: Prefix Cache 命中率塌方 (Cache Hit Regression)

## Symptom Reconfirm
- 用 `prefix_cache_hits_total / prefix_cache_queries_total` 计算的 token 命中率显著低于该 workload 历史基线
- TTFT 与 prefill work 同步变差；TPOT 可能较稳定，但必须用数据确认
- 时间点和某次发布 / 路由策略改动重合

## Triage Commands

```bash
# 1. token 口径命中率趋势 + 同期发布事件
curl -sS -G \
  --data-urlencode 'query=sum(rate(vllm:prefix_cache_hits_total[5m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])), 1)' \
  --data-urlencode "start=$(($(date +%s)-3600))" --data-urlencode "end=$(date +%s)" --data-urlencode 'step=60' \
  "$PROM_URL/api/v1/query_range" \
  > "$INCIDENT_DIR/evidence/hit-rate-1h.json"

# 2. 网关 / 路由器近期变更
kubectl rollout history deploy/vllm-gateway -n ${VLLM_NAMESPACE:-vllm} \
  > "$INCIDENT_DIR/evidence/gateway-rollout.txt"
kubectl rollout history deploy/${VLLM_DEPLOYMENT:-vllm} -n ${VLLM_NAMESPACE:-vllm} \
  > "$INCIDENT_DIR/evidence/vllm-rollout.txt"

# 3. 记录部署 manifest 中的 model/tokenizer/template revision；不要只看
# transformers 包版本，也不要在生产 Pod 临时运行任意 Python。
first_pod=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  "env | grep -E 'MODEL_REVISION|TOKENIZER_REVISION|CHAT_TEMPLATE'" \
  > "$INCIDENT_DIR/evidence/model-input-revisions.txt"

# 4. 请求是否还分配到同一个 pod（sticky session 失效会让 cache 完全错位）
curl -sS "$PROM_URL/api/v1/query?query=sum(rate(vllm:request_success_total[5m]))by(pod)" \
  > "$INCIDENT_DIR/evidence/req-by-pod.json"
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| Gateway 最近回滚到 round-robin 路由 | smart router 配置丢了，sticky session 失效 |
| model/tokenizer/template revision 改变 | cache key 输入改变；用 manifest 与 golden tokenization 验证 |
| 请求分布与稳定版本显著不同 | 路由变化是候选原因；需和 prefix 分布联合验证 |
| 单 pod 重启后该 pod 命中率暂降 | 可能是 cache 冷启动；按业务窗口观察重建曲线 |

## Remediate

- **L1（只读）**：抓证据
- **L2（需逐命令批准）**：只在 A/B 证明确有收益后修改 gateway 路由；先展示当前策略、候选策略、受影响租户和回滚。
- **L3（需逐命令批准）**：
  - 回滚 gateway 到上一个 revision（`kubectl rollout undo deploy/vllm-gateway`）
  - 影响：所有路由配置回到 N-1，包含其他可能的改动
  - 回滚 vllm 镜像（如果 tokenizer 版本就是从这次镜像变的）—— 影响：全员重启 + 可能丢失新模型能力

## Verification

- 覆盖典型 prefix 重用周期后，token 命中率回到该 workload 的批准基线区间
- TTFT/prefill work 回到 SLO，且负载均衡和错误预算没有退化

## Long-term

- Gateway 路由策略走 canary，毕业条件由覆盖请求数、基线区间与错误预算决定
- tokenizer/template 升级使用无敏感信息的批准 fixture 做 golden 与 warmup，不回放未经授权的真实请求
- 命中率告警使用按 workload 建立的动态或版本化基线
- 见 `reference/checklist-prelaunch.md` 第 12 条

<!-- source: ../../08-production-deployment/07-incident-playbook.md case 3 -->
