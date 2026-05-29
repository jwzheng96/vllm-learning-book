# Playbook 05: Prefix Cache 命中率塌方 (Cache Hit Regression)

## Symptom Reconfirm
- `vllm:gpu_prefix_cache_hit_rate` 从历史基线（一般 60-90%）跌到 < 50%
- TTFT p99 翻倍，TPOT 影响小（命中是 prefill 加速，decode 没受影响）
- 时间点和某次发布 / 路由策略改动重合

## Triage Commands

```bash
# 1. 命中率 1 小时趋势 + 同期发布事件
curl -sS "$PROM_URL/api/v1/query_range?query=avg(vllm:gpu_prefix_cache_hit_rate)&start=$(($(date +%s)-3600))&end=$(date +%s)&step=60" \
  > "$INCIDENT_DIR/evidence/hit-rate-1h.json"

# 2. 网关 / 路由器近期变更
kubectl rollout history deploy/vllm-gateway -n ${VLLM_NAMESPACE:-vllm} \
  > "$INCIDENT_DIR/evidence/gateway-rollout.txt"
kubectl rollout history deploy/${VLLM_DEPLOYMENT:-vllm} -n ${VLLM_NAMESPACE:-vllm} \
  > "$INCIDENT_DIR/evidence/vllm-rollout.txt"

# 3. tokenizer 版本检查（tokenizer 一变，旧 cache 全部 miss）
first_pod=$(kubectl get pods -n ${VLLM_NAMESPACE:-vllm} -l ${VLLM_SERVICE_LABEL:-app.kubernetes.io/name=vllm} -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${VLLM_NAMESPACE:-vllm} $first_pod -- bash -lc \
  "python3 -c 'import transformers; print(transformers.__version__)'" \
  > "$INCIDENT_DIR/evidence/transformers-version.txt"

# 4. 请求是否还分配到同一个 pod（sticky session 失效会让 cache 完全错位）
curl -sS "$PROM_URL/api/v1/query?query=sum(rate(vllm:request_success_total[5m]))by(pod)" \
  > "$INCIDENT_DIR/evidence/req-by-pod.json"
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| Gateway 最近回滚到 round-robin 路由 | smart router 配置丢了，sticky session 失效 |
| tokenizer 升级（transformers 大版本变化） | 切词不同，prefix hash 全变 |
| 请求按 pod 分布从 90/10 变 50/50 | 路由权重错配 |
| 单 pod 重启（cache 被清） | 一次性现象，等待 cache 重建即可，多观察 5 分钟 |

## Remediate

- **L1**：抓证据
- **L2（直接做）**：
  - 检查 gateway 当前路由策略，若是 round-robin 改回 prefix-aware
  - 确认 `sticky_session=true`
- **L3（弹确认）**：
  - 回滚 gateway 到上一个 revision（`kubectl rollout undo deploy/vllm-gateway`）
  - 影响：所有路由配置回到 N-1，包含其他可能的改动
  - 回滚 vllm 镜像（如果 tokenizer 版本就是从这次镜像变的）—— 影响：全员重启 + 可能丢失新模型能力

## Verification

- 10 分钟内 `vllm:gpu_prefix_cache_hit_rate` 回升至基线 ± 10%
- TTFT p99 回到塌方前水平

## Long-term

- Gateway 路由策略变更必须走 canary：先 5% 流量观察 cache 命中率
- tokenizer 升级强制做 cache 暖启动（先回放 N 万个真实请求）
- 监控加 `prefix_cache_hit_rate < baseline*0.7` 告警
- 见 `reference/checklist-prelaunch.md` 第 12 条

<!-- source: ../../08-production-deployment/07-incident-playbook.md case 3 -->
