# vLLM 事故复盘模板

> 事故可以是隔离环境注入或 tabletop；必须明确 `exercise` / `production`，不能把桌面推演写成真实故障。

## 1. Header

| 字段 | 值 |
| --- | --- |
| incident ID / type | `inc-YYYYMMDD-sequence` / `exercise` |
| severity | `not-assigned` |
| start / detect / mitigate / recover UTC | `not-recorded` |
| commander / responders / reviewer | `not-recorded` |
| source/image/model/config revision | `not-recorded` |
| affected tenants/routes/regions | `not-recorded` |
| status | `draft` |

## 2. Impact

| Metric | Baseline | Incident peak | Duration | Evidence |
| --- | ---: | ---: | ---: | --- |
| failed requests | `not-measured` | `not-measured` | `not-measured s` | `not-recorded` |
| TTFT / TPOT / E2E p99 | `not-measured ms` | `not-measured ms` | `not-measured s` | `not-recorded` |
| waiting / preemption rate | `not-measured` | `not-measured` | `not-measured s` | `not-recorded` |
| tenants/users affected | `not-measured` | `not-measured` | `not-measured s` | `not-recorded` |
| data/security impact | `not-assessed` | `not-assessed` | `not-applicable` | `not-recorded` |

User-visible impact statement: `not-recorded`。

## 3. Detection

- First signal/alert: `not-recorded`
- Alert query + threshold + for window: `not-recorded`
- Why this signal represented user impact: `not-recorded`
- Missed/late signals: `not-assessed`
- Deploy/config/model annotations present: `not-assessed`

## 4. Timeline

| UTC | Actor/system | Observation (fact) | Action | Evidence link |
| --- | --- | --- | --- | --- |
| `not-recorded` | `not-recorded` | `not-recorded` | `not-recorded` | `artifacts/not-recorded` |

事实与假设分开；不要在早期时间线里反向填入后来才知道的根因。

## 5. Triage evidence

| Layer | Evidence checked | Finding | Ruled in/out |
| --- | --- | --- | --- |
| client/gateway/network | `not-recorded` | `not-recorded` | `not-decided` |
| API/frontend/tokenizer | `not-recorded` | `not-recorded` | `not-decided` |
| scheduler queue/KV | `not-recorded` | `not-recorded` | `not-decided` |
| model runner/GPU | `not-recorded` | `not-recorded` | `not-decided` |
| distributed/NCCL | `not-recorded` | `not-recorded` | `not-decided` |
| storage/model dependency | `not-recorded` | `not-recorded` | `not-decided` |

Relevant current metrics: `not-recorded`。Logs/traces/profiles redacted: `not-assessed`。

## 6. Root cause and contributing factors

- Direct technical cause: `not-proven`
- Mechanism/evidence chain: `not-recorded`
- Trigger: `not-recorded`
- Contributing technical factors: `not-recorded`
- Contributing process/organizational factors: `not-recorded`
- Why existing controls did not prevent/contain it: `not-recorded`
- Alternative hypotheses rejected + evidence: `not-recorded`

使用“5 Whys”只能帮助提问，不能替代可验证的 causal chain。

## 7. Mitigation and recovery

| Action | Expected effect | Actual effect | Risk/blast radius | Rollback | Evidence |
| --- | --- | --- | --- | --- | --- |
| `not-recorded` | `not-recorded` | `not-measured` | `not-assessed` | `not-recorded` | `not-recorded` |

- Admission/drain behavior: `not-recorded`
- Requests lost/retried/duplicated: `not-assessed`
- Secret/data exposure during debugging: `not-assessed`
- Recovery verification: health + golden request + SLO + error + queue = `not-run`
- Time metrics: MTTD `not-calculated`, MTTA `not-calculated`, MTTR `not-calculated`

```text
MTTD = detect_time - incident_start
MTTA = first_safe_action_time - detect_time
MTTR = verified_recovery_time - incident_start
```

## 8. What went well / poorly / lucky

- Went well: `not-recorded`
- Went poorly: `not-recorded`
- Lucky/near miss: `not-recorded`
- Unsafe action avoided: `not-recorded`

不记录个人归责；记录系统、界面、权限、文档和决策条件。

## 9. Corrective actions

| ID | Action | Type | Owner | Due UTC | Acceptance evidence | Priority | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CA-1` | `not-decided` | prevent/detect/mitigate/recover | `unassigned` | `not-set` | `not-defined` | `not-ranked` | `open` |

每项必须可验证；“加强监控”“优化性能”不是可验收 action。

## 10. Regression / drill

- Minimal reproducer or tabletop script: `not-recorded`
- Bounded injection and stop condition: `not-recorded`
- Regression test/golden request: `not-created`
- Next drill UTC + owner: `not-scheduled`
- Runbook/dashboard/alert updated links: `not-updated`

## 11. Communication and audit

- Stakeholder updates and UTC: `not-recorded`
- Customer communication required/sent: `not-assessed`
- Security/privacy/legal escalation: `not-assessed`
- Artifact manifest + retention class: `not-recorded`
- Review sign-off: `not-recorded`

## Acceptance rubric

| 项 | Pass 条件 | 状态 |
| --- | --- | --- |
| Impact | 用户影响有单位、窗口和证据 | `not-evaluated` |
| Causality | 根因由跨层证据链支持，列出被排除假设 | `not-evaluated` |
| Safety | mitigation blast radius、rollback、data/secret 审查完整 | `not-evaluated` |
| Recovery | health/golden/SLO/error/queue 全部验证 | `not-evaluated` |
| Learning | actions 有 owner、due、acceptance evidence、复演 | `not-evaluated` |
