---
name: vllm-doctor
description: Use when a running vLLM service has latency, queue, KV-cache, worker-stall, OOM, retry, prefix-cache, cold-start, output-quality, or LoRA instability symptoms.
---

# vLLM Production Stability Doctor

把 vllm-learning notebook 里的 incident playbook 编成 fail-closed 的证据收集与整改建议流程。

**Core safety rule: Every cluster mutation requires explicit user approval.** Read-only external evidence collection may run automatically. Local artifacts may only be written under the newly disclosed `$INCIDENT_DIR`; changes to any other filesystem path, Kubernetes, gateway, model, node, process, or external state are mutations regardless of L1/L2/L3 label.

## 何时使用

- 线上 vLLM Pod 出现 TTFT/TPOT 抖动、5xx 飙升、卡 worker、OOMKilled、prefix cache 命中率塌方
- 客户反馈输出质量或业务字符策略异常
- 滚动升级、扩容后行为退化
- 例行健康巡检

## 调用前置条件

线上分类必须从本服务的 SLO 与容量实验显式提供阈值；缺失时 fail closed。下列值是占位符，不是默认建议：

```bash
export VLLM_NAMESPACE=vllm                                  # k8s namespace
export VLLM_SERVICE_LABEL=app.kubernetes.io/name=vllm       # pod selector
export PROM_URL=http://prometheus.monitoring:9090           # Prometheus 入口
export KUBECONFIG=$HOME/.kube/config                        # 默认值，可省
export TTFT_SLO_MS="REPLACE_WITH_APPROVED_SLO_MS"
export QUEUE_HIGH="REPLACE_WITH_VALIDATED_QUEUE_WATERLINE"
export KV_HIGH="REPLACE_WITH_VALIDATED_KV_WATERLINE"
export PREEMPT_HIGH_PER_SEC="REPLACE_WITH_VALIDATED_PREEMPT_RATE"
export PREFIX_CACHE_DROP_FROM="REPLACE_WITH_VERSIONED_WORKLOAD_BASELINE"
export RUNNING_LOW="REPLACE_WITH_VALIDATED_LOW_RUNNING_BOUNDARY"
# 可选：跑离线 dry-run
export VLLM_DOCTOR_FIXTURE=/path/to/golden3.json
```

工作产物默认写到 `./vllm-doctor-incident-$(date +%Y%m%d-%H%M%S)/`，下文统称 `$INCIDENT_DIR`。

---

## 工作流（agent 拿到这份后逐阶段执行）

### Phase 0  环境探测

```bash
INCIDENT_DIR="./vllm-doctor-incident-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$INCIDENT_DIR"
bash "$CLAUDE_SKILL_DIR/scripts/connect_probe.sh" > "$INCIDENT_DIR/connect.json"
```

- 任一探测项失败（kubectl/prom/gpu）→ 把对应的 actionable 错误信息原样给用户，**等他修好再继续**。不要静默跳过。
- 探测全 ok → 进 Phase 1。

### Phase 1  Golden 3 拉取

```bash
bash "$CLAUDE_SKILL_DIR/scripts/golden3.sh" > "$INCIDENT_DIR/golden3.json"
```

输出 schema（以下数字只是离线 fixture）：

```json
{
  "ts": "2026-05-29T12:00:00Z",
  "ttft_p99_ms": 9000,
  "queue": 80,
  "kv_usage": 0.95,
  "throughput": 100.0,
  "running": 50,
  "prefix_cache_hit_rate": 0.82,
  "preempt_rate_per_sec": 0.6,
  "request_failed_rate": 0.0,
  "format_compliance_rate": 1.0
}
```

如果 `$VLLM_DOCTOR_FIXTURE` 设置了，`golden3.sh` 会直接拷贝 fixture 当输出（test mode）。

### Phase 2  决策树路由

```bash
python3 "$CLAUDE_SKILL_DIR/scripts/triage.py" \
  < "$INCIDENT_DIR/golden3.json" > "$INCIDENT_DIR/triage.json"
```

输出形如 `{"playbook": "02-nccl-hang", "confidence": 0.95, "reason": "throughput=0 AND running=8 >0"}`。

`confidence < 0.5` → 提示用户当前症状不明确，建议先人工核对再触发本 skill。

### Phase 3  深度诊断（命中的 playbook）

读取 `playbooks/<playbook>.md`，按其 "Triage Commands" 节执行。所有命令输出写到 `$INCIDENT_DIR/evidence/`。

把 playbook 的 if/then 表当作假设生成器。只有相互独立的 runtime evidence 支持时才记录 root cause；否则写 `unconfirmed` 和下一条只读检查。

### Phase 4  整改建议与逐条授权

每个 playbook 的 "Remediate" 节按 L1/L2/L3 分级：

- **L1（只读）**：拉取经过脱敏的 metrics、describe、logs、stack 与配置基线，可直接做
- **L2（有状态变更）**：改 env、replica、gateway policy 或 rollout 参数，必须逐条征得显式批准
- **L3（破坏性变更）**：delete/restart/taint/rollback/scale-to-zero，必须逐条征得显式批准，并说明 blast radius 与恢复门槛

整改脚本只生成计划，绝不执行 mutation。对每条 L2/L3，先解析唯一 target、读取 current state、写 command/rollback/stop condition，再询问用户；批准只覆盖该条已展示命令。用户跳过、目标变化或 current state 漂移时重新生成计划。

每条 action 用一致格式落 log：

```
<timestamp>  L2  <exact approved command>
  rollback: <restore captured current state>
```

调用 `scripts/remediate_<id>.sh`（如存在）只生成候选计划；没有专用脚本时也只从 playbook 生成建议，不直接执行。

### Phase 5  恢复验证

```bash
: "${VERIFY_INTERVAL_SECONDS:?set from the playbook observation window}"
for i in 1 2 3; do
  sleep "$VERIFY_INTERVAL_SECONDS"
  bash "$CLAUDE_SKILL_DIR/scripts/golden3.sh" > "$INCIDENT_DIR/verify-$i.json"
done
python3 "$CLAUDE_SKILL_DIR/scripts/triage.py" --verify \
  "$INCIDENT_DIR/verify-1.json" \
  "$INCIDENT_DIR/verify-2.json" \
  "$INCIDENT_DIR/verify-3.json" > "$INCIDENT_DIR/verify.json"
```

`triage.py --verify` 只会返回 `NO_ACTIVE_ROUTE`、`NOT_RESOLVED` 或 `INSUFFICIENT_EVIDENCE`，不会单独宣告事故恢复。只有三个采样点和 playbook 的全部 Verification gate 都通过，报告才可写 `RESOLVED`。否则：
- 命中其他 playbook → 链式进入下一条
- 命中相同 playbook 且整改已用尽 → `status: NEEDS_HUMAN`，把证据包路径告诉用户

### Phase 6  输出报告

写 `$INCIDENT_DIR/report.md`，结构：

```markdown
# Incident Report <timestamp>

## 触发症状
（金标三指标的 before 表，引用 golden3.json）

## 命中 playbook
（id + 信心 + 理由）

## 证据
（evidence/ 下关键文件清单）

## 执行的整改
（actions.log 内容渲染成表格）

## 恢复结果
RESOLVED / NOT_RESOLVED / INSUFFICIENT_EVIDENCE / NEEDS_HUMAN
（verify-1/2/3 表）

## 长期改进建议
（指向 reference/checklist-prelaunch.md 对应条目）
```

最后把 `report.md` 的路径直接打印给用户。

---

## 关键约束

- **任何 mutation 都先问**：不得因为“低风险”“可回滚”“事故紧急”或标成 L2 而跳过批准。
- **不要并行跑 L2/L3 整改**：每次批准并执行一条，完成恢复验证后再建议下一条。
- **每次只走一条 playbook**：决策树命中多个 → 取 confidence 最高，其余写入 `triage.json.alternatives` 供报告引用。
- **批准不能批量**：L2/L3 都一条一条问，不把不同 target/action 打包。
- **不要覆盖证据**：`$INCIDENT_DIR` 每次新建，命名含时间戳，不复用。
- **不要碰 vllm 子模块**：本 skill 只读子模块（如果需要参考源码行号），从不修改。

---

## 离线 dry-run（测 skill 本身的连接逻辑）

```bash
# 1. 校验 frontmatter 没坏
python3 -c "import yaml; print(yaml.safe_load(open('SKILL.md').read().split('---')[1]))"

# 2. 喂 mock 数据测决策树分支
echo '{"ttft_p99_ms":9000,"queue":80,"kv_usage":0.95,"throughput":100,"running":50,"prefix_cache_hit_rate":0.8,"preempt_rate_per_sec":0.6,"request_failed_rate":0,"format_compliance_rate":1}' \
  | VLLM_DOCTOR_USE_EXAMPLE_THRESHOLDS=1 python3 scripts/triage.py
# 期望：playbook=01-preempt-cascade

echo '{"ttft_p99_ms":500,"queue":0,"kv_usage":0.5,"throughput":0,"running":8,"prefix_cache_hit_rate":0.8,"preempt_rate_per_sec":0,"request_failed_rate":0,"format_compliance_rate":1}' \
  | VLLM_DOCTOR_USE_EXAMPLE_THRESHOLDS=1 python3 scripts/triage.py
# 期望：playbook=02-nccl-hang

# 3. shellcheck
shellcheck scripts/*.sh
```

---

## 参考材料（深读用）

skill 自包含，但需要更细的背景时读：

- `playbooks/` 下每份对应一个合成 incident 场景
- `reference/promql-cheatsheet.md` — 全部用到的 PromQL
- `reference/nccl-env.md` — NCCL_* 环境变量影响
- `reference/checklist-prelaunch.md` — 防患于未然的 15 条上线前检查
