# Playbook 01: KV 抢占级联 (Preempt Cascade)

## Symptom Reconfirm
- **必须同时满足**：
  - `vllm:gpu_cache_usage_perc >= 0.9` 持续 ≥ 60s
  - `rate(vllm:num_preemptions_total[5m]) >= 0.5/s`
  - TTFT p99 和 TPOT 同时变差
- **排除**：
  - 如果同时出现 OOMKilled / exit code 137 → 走 `03-gpu-oom`
  - 如果 throughput == 0 → 走 `02-nccl-hang`

## Triage Commands

```bash
# 1. 拉过去 10 分钟的 KV / 抢占速率 / TTFT 趋势
bash "$CLAUDE_SKILL_DIR/scripts/kv_pressure_diag.sh" \
  "$INCIDENT_DIR/evidence/kv"

# 2. 当前实例数 vs HPA / KEDA 目标
kubectl get lws/${VLLM_LWS:-vllm} -n ${VLLM_NAMESPACE:-vllm} \
  -o jsonpath='{.spec.replicas}' > "$INCIDENT_DIR/evidence/replicas.txt"

# 3. 是否有长尾请求霸占 batch（一个 >5min 的请求会把整个 batch KV 钉满）
# longest_running_seconds 由 kv_pressure_diag 输出
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| `kv ≥ 0.9` + `preempt > 0.5/s` + 流量正常 | 容量不足 / `max_num_seqs` 设大了 |
| `kv ≥ 0.95` + `longest_running > 300s` | 长尾请求堵 batch |
| `kv ≥ 0.9` + 突然出现，QPS 同步翻倍 | 客户端重试雪崩（先做 04，再回这里） |
| `kv` 周期性 0.9↔0.6 抖动 + replica 频繁变化 | HPA cooldown 太短，flapping |

## Remediate

按 `scripts/remediate_01.sh` 输出的 L1/L2/L3 顺序执行。关键动作：

- **L2（直接做）**：
  - `MAX_NUM_SEQS` 从 64 → 32（牺牲吞吐换稳定）
  - `MAX_NUM_BATCHED_TOKENS` 从 4096 → 2048
  - LWS 扩容 +2 replica
  - Gateway 加 admission control：`kv_cache_usage > 0.85` 时返回 429
- **L3（弹确认）**：
  - 滚动重启（`kubectl rollout restart`）—— 影响：全员经历一次冷启动
  - 更激进扩容 +4 replica —— 影响：成本

## Verification

整改后 60s 起，连续 3 个采样点必须全部满足：
- `vllm:gpu_cache_usage_perc < 0.85`
- `rate(vllm:num_preemptions_total[5m]) < 0.1/s`
- TTFT p99 < `$TTFT_SLO_MS`（默认 2000ms）

## Long-term

- 启用 chunked prefill：`--enable-chunked-prefill --max-num-batched-tokens 2048-4096`
- KEDA 触发器加 `gpu_cache_usage_perc > 0.8` 作为扩容信号（不只是 queue）
- 长上下文请求隔离到独立 pod 池（避免长尾堵主池）
- 见 `reference/checklist-prelaunch.md` 第 4、6 条

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md L114-146 + 07-incident-playbook.md case 1 -->
