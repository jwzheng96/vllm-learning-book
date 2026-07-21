# Playbook 01: KV 抢占级联 (Preempt Cascade)

## Symptom Reconfirm
- **必须同时满足**：
  - `vllm:kv_cache_usage_perc` 越过该模型池压测得到的 `$KV_HIGH`
  - `rate(vllm:num_preemptions_total[5m])` 相对健康基线上升
  - queue、TTFT 或 TPOT 同时变差
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

# 3. 从网关请求长度分布确认是否有超长请求；vLLM 核心指标不提供
# “最长在途请求”这个 gauge，不能从 histogram 伪造单请求证据。
```

## Root Cause 判定

| 现象 | 根因 |
| --- | --- |
| KV 越过本池安全水位 + preempt 上升 + 流量正常 | KV 容量不足；`max_num_seqs` 只是待验证假设之一 |
| 长度分布突增 + KV/preempt 同步恶化 | 超长请求带来的容量压力 |
| KV 突然升高，入口尝试率同步异常上升 | 客户端重试雪崩（先做 04，再回这里） |
| KV 周期振荡 + replica 频繁变化 | autoscaler flapping；需对齐扩缩事件验证 |

## Remediate

`scripts/remediate_01.sh` 只输出候选计划。以下每个变更都必须展示唯一 target、当前值、候选值、回滚和停止条件，并逐条取得批准：

- **L2（需逐命令批准）**：按压测矩阵降低 admission / `max_num_seqs` 的一个档位；或按已测单副本容量计算扩容数；网关在本池安全水位和队列条件下拒绝新请求。
- **L3（需逐命令批准）**：滚动重启或切换节点池。先 drain，说明 cold-start、成本和可回滚容量。

## Verification

经过一个完整业务观测窗口后，连续采样必须全部满足：
- `vllm:kv_cache_usage_perc` 回到本池已验证安全区间
- preemption rate 回到事故前健康基线
- queue、TTFT 与 TPOT 满足该服务已批准的 SLO

## Long-term

- 对 chunked prefill、token budget 与 seq budget 做成组 A/B 压测，不照抄固定值
- autoscaling 同时观察 `vllm:kv_cache_usage_perc`、queue 和稳定窗口，阈值来自本池容量实验
- 长上下文请求隔离到独立 pod 池（避免长尾堵主池）
- 见 `reference/checklist-prelaunch.md` 第 4、6 条

<!-- source: ../../08-production-deployment/06-reliability-and-failure-modes.md + 07-incident-playbook.md case 1 -->
