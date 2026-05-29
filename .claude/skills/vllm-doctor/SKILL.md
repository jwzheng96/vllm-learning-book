---
name: vllm-doctor
description: 诊断并自动整改运行中的 vLLM 集群常见稳定性问题（KV 抢占、NCCL hang、OOM、重试雪崩、prefix cache 塌方、冷启动、输出乱码、LoRA 抖动）。按 Golden 3 指标自动路由 playbook，分级执行整改（L1/L2 直接做，L3 弹确认）。
allowed-tools: [Bash, Read, Write, AskUserQuestion]
---

# vLLM Production Stability Doctor

把 vllm-learning notebook 里散落的 incident playbook 编成 agent 可以自动跑的诊断+整改流程。

## 何时使用

- 线上 vLLM Pod 出现 TTFT/TPOT 抖动、5xx 飙升、卡 worker、OOMKilled、prefix cache 命中率塌方
- 客户反馈输出质量异常或非 ASCII
- 滚动升级、扩容后行为退化
- 例行健康巡检

## 调用前置条件

shell 里必须 export 以下变量（不全也能跑，会在 Phase 0 提示）：

```bash
export VLLM_NAMESPACE=vllm                                  # k8s namespace
export VLLM_SERVICE_LABEL=app.kubernetes.io/name=vllm       # pod selector
export PROM_URL=http://prometheus.monitoring:9090           # Prometheus 入口
export KUBECONFIG=$HOME/.kube/config                        # 默认值，可省
export TTFT_SLO_MS=2000                                     # 决策树阈值，可省
export QUEUE_HIGH=50
export KV_HIGH=0.9
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

输出 schema：

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

判定 root cause：playbook 文件里 "Root Cause 判定" 节给的是 if/then 表，按表落定。

### Phase 4  整改执行（三级授权）

每个 playbook 的 "Remediate" 节按 L1/L2/L3 分级：

- **L1（只读/旁路）**：拉 dump、抓日志、记录基线 → 直接做
- **L2（受控扰动）**：改 env、调 `max_num_seqs`、加 replica、改 gateway rate limit → 直接做，但执行前先把命令和**回滚命令**写进 `$INCIDENT_DIR/actions.log`
- **L3（高破坏性）**：`kubectl delete pod`、`taint node`、`kubectl set image` 回滚、`kubectl scale --replicas=0` → 用 **AskUserQuestion** 弹一条确认，options = ["执行", "跳过"]，附 30 字内的 blast radius 说明

每条 action 用一致格式落 log：

```
2026-05-29T12:05:30Z  L2  kubectl set env deploy/vllm MAX_NUM_SEQS=32
  rollback: kubectl set env deploy/vllm MAX_NUM_SEQS-
```

调用 playbook 提供的 `scripts/remediate_<id>.sh`（如存在），它内部已经按级别分好块；没有专用脚本的 playbook 则按其 markdown 里给出的命令逐条执行。

### Phase 5  恢复验证

```bash
for i in 1 2 3; do
  sleep 60
  bash "$CLAUDE_SKILL_DIR/scripts/golden3.sh" > "$INCIDENT_DIR/verify-$i.json"
done
python3 "$CLAUDE_SKILL_DIR/scripts/triage.py" --verify \
  "$INCIDENT_DIR/verify-1.json" \
  "$INCIDENT_DIR/verify-2.json" \
  "$INCIDENT_DIR/verify-3.json" > "$INCIDENT_DIR/verify.json"
```

三个采样点都满足该 playbook 的 "Verification" 表达式 → `status: RESOLVED`。否则：
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
RESOLVED / NEEDS_HUMAN
（verify-1/2/3 表）

## 长期改进建议
（指向 reference/checklist-prelaunch.md 对应条目）
```

最后把 `report.md` 的路径直接打印给用户。

---

## 关键约束

- **不要并行跑 L2/L3 整改**：vLLM rollout 期间再改 env 会触发二次重启，先做完再观察。
- **每次只走一条 playbook**：决策树命中多个 → 取 confidence 最高，其余写入 `triage.json.alternatives` 供报告引用。
- **AskUserQuestion 不能批量**：L3 整改需要一条一条问，不要打包成多选。
- **不要覆盖证据**：`$INCIDENT_DIR` 每次新建，命名含时间戳，不复用。
- **不要碰 vllm 子模块**：本 skill 只读子模块（如果需要参考源码行号），从不修改。

---

## 离线 dry-run（测 skill 本身的连接逻辑）

```bash
# 1. 校验 frontmatter 没坏
python3 -c "import yaml; print(yaml.safe_load(open('SKILL.md').read().split('---')[1]))"

# 2. 喂 mock 数据测决策树分支
echo '{"ttft_p99_ms":9000,"queue":80,"kv_usage":0.95,"throughput":100,"running":50,"prefix_cache_hit_rate":0.8,"preempt_rate_per_sec":0.6,"request_failed_rate":0,"format_compliance_rate":1}' \
  | python3 scripts/triage.py
# 期望：playbook=01-preempt-cascade

echo '{"ttft_p99_ms":500,"queue":0,"kv_usage":0.5,"throughput":0,"running":8,"prefix_cache_hit_rate":0.8,"preempt_rate_per_sec":0,"request_failed_rate":0,"format_compliance_rate":1}' \
  | python3 scripts/triage.py
# 期望：playbook=02-nccl-hang

# 3. shellcheck
shellcheck scripts/*.sh
```

---

## 参考材料（深读用）

skill 自包含，但需要更细的背景时读：

- `playbooks/` 下每份对应的 incident 原案例
- `reference/promql-cheatsheet.md` — 全部用到的 PromQL
- `reference/nccl-env.md` — NCCL_* 环境变量影响
- `reference/checklist-prelaunch.md` — 防患于未然的 15 条上线前检查
