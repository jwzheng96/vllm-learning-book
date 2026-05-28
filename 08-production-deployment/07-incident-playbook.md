# 07. 真实故障 Runbook：8 个面试可以讲的案例

> **谁该读这一篇？** 一线 on-call SRE、面试推理岗位想"讲事故"的工程师、postmortem 文化的推动者。
>
> **前置阅读：** [`05-slo-and-observability.md`](./05-slo-and-observability.md)、[`06-reliability-and-failure-modes.md`](./06-reliability-and-failure-modes.md)
>
> **耗时：** 约 30 分钟
>
> **学完能：**
> 1. 用"症状 → 排查 → 根因 → 修复 → 长期改进"的结构复盘任意 LLM 故障
> 2. 在面试里把至少 4 个真实 case（NCCL hang / preempt cascade / cache 跌 / retry storm 等）讲清楚
> 3. 写出可机械执行的 runbook 与决策树
> 4. 区分紧急止血动作与长期改进项

"你处理过最复杂的生产故障是什么？" —— 这是中后期面试必问题。本节把 LLM 推理常见 8 种故障写成可读的 runbook，每个都按 **症状 → 排查 → 根因 → 修复 → 长期改进** 的结构。你能把这些讲清楚，面试官会觉得你"真做过事"。

---

## 故障 1：TTFT p99 突增 10 倍

### 症状
- Grafana 报警：TTFT p99 从 200ms 飙到 3s
- 用户投诉"卡半天才出第一个字"
- TPOT 一切正常

### 排查（5 分钟内做这些）
```
# 1. 先看队列
PromQL: sum(vllm:num_requests_waiting) → 持续 > 10？YES
# 2. 再看 KV
PromQL: avg(vllm:gpu_cache_usage_perc) → > 0.9？YES
# 3. 是否有 preempt
PromQL: rate(vllm:num_preemptions_total[5m]) → > 0？YES
# 4. 流量层面
PromQL: sum(rate(vllm:request_success_total[1m])) → 比昨天同时段高 2×
```

### 根因
流量突增，但 HPA cooldown 还没释放，KV 满了，请求开始排队。

### 修复（紧急）
1. 手动扩 Pod 2 个（绕过 HPA）
2. Gateway 临时降级：拒绝 max_tokens > 2048 的请求
3. 看 TTFT 回落

### 长期改进
- HPA 阈值降低：`num_requests_waiting > 3` 而不是 10
- Predictive scaling：基于 hour-of-day 提前扩
- Warm pool：常驻 20% 冗余
- 客户端 SDK 调小默认 max_tokens

### 面试讲法
> "我们碰到 TTFT 突增。我先看 vLLM 的金信号：发现 queue depth、KV usage、preempt 三件套全红。说明 KV 压力大、调度饥饿。临时手动扩容 + Gateway 限流稳住，长期改 HPA 阈值 + warm pool。"

---

## 故障 2：NCCL hang，整组 Pod 停摆

### 症状
- `vllm:num_requests_running` 一直 > 0 但 token throughput = 0
- Pod health check 通过、liveness 通过
- `kubectl logs` 没新输出

### 排查
```
kubectl exec pod -- nvidia-smi          # GPU 完全空闲
kubectl exec pod -- py-spy dump --pid 1 # Python 栈卡在 ncclAllReduce
kubectl exec pod -- cat /var/log/nccl-*.log
nvidia-smi nvlink -e                    # NVLink CRC error 计数上涨
```

### 根因
NVLink 某条链路有间歇错误，NCCL 一次 AllReduce 卡死，整组 hang。

### 修复（紧急）
- LeaderWorkerSet 重启整组 Pod
- 节点打 taint，让其他 Pod 不调度过来
- 通知运维换 NVLink 线/查硬件

### 长期改进
- `NCCL_BLOCKING_WAIT=1`、`NCCL_TIMEOUT=60`：超时变 crash 而不是 hang
- Liveness probe 检查 "最近 30s 是否有成功 step"
- DCGM 监控 NVLink CRC，预警硬件问题
- 节点定期 nccl-tests 自检

### 面试讲法
> "诊断 hang vs crash 的差别是关键。我们靠 throughput=0 而 running>0 的金信号识别。py-spy 看到栈卡在 NCCL，nvidia-smi 看 NVLink 错误。立刻重启 + 隔离节点。长期上 NCCL 超时配置 + DCGM 监控。"

---

## 故障 3：Prefix cache 命中率从 80% 跌到 5%

### 症状
- TTFT 整体上涨 50%
- 后端 GPU util 高了 30%
- `vllm:gpu_prefix_cache_hit_rate` 大跌

### 排查
- 最近变更：tokenizer 升级了吗？路由策略改了吗？
- Trace 一个用户 session：发现两轮对话路由到不同 Pod
- 检查 router：从 cache-aware 改回了 round-robin

### 根因
有人改 router config，把 routing policy 误设成 random。

### 修复
回滚 router config。

### 长期改进
- 重要配置变更走 review + 金丝雀
- 路由策略 metric exposed，发布前自动比对
- 配置变更触发 SLO check pipeline

### 面试讲法
> "Cache hit 突跌通常三件事之一：模型升级（tokenizer 变）、路由变（session 不 sticky）、流量模式变。我们靠最近变更回溯发现路由配置错误。"

---

## 故障 4：上游 retry storm 把服务打挂

### 症状
- 一个 Pod OOM 重启 30s
- 但其他 Pod 也开始 5xx
- 全集群崩溃

### 排查
- 看 traffic：QPS 1 分钟内涨了 5×
- 看 client log：SDK 默认无限 retry
- 一个 Pod 挂了，请求全打到剩下 Pod → 它们也挂

### 根因
客户端 SDK retry 无 budget，故障期间流量呈倍数增长。

### 修复（紧急）
- Gateway 强制限流（先 5×，逐步放）
- 通知客户端方禁用 retry
- 等流量平稳后慢慢扩

### 长期改进
- Gateway 默认有 ratelimit（按 client / API key）
- 客户端 SDK 强制 jittered exp backoff + retry budget
- Pod 优雅 drain，不要 hard kill 触发 retry
- 跨 AZ 部署，单 AZ 挂不会全崩

### 面试讲法
> "重启风暴是经典 cascade。流量曲线是关键证据。修复优先稳定（限流），再补 SDK 行为规范。长期上多 AZ + sane retry budget。"

---

## 故障 5：模型升级后输出乱码

### 症状
- 新模型上 10% 流量 30 分钟
- 用户投诉"输出是乱七八糟的 token"
- error rate 不高（HTTP 200，body 是乱字符）

### 排查
- 抽样几个 response：确实 garbage（重复字符、控制字符）
- 用同样 prompt 在 staging 试旧模型：正常
- 看新模型加载：FP8 量化，没换 calibration set

### 根因
新模型权重格式不兼容当前 vLLM 版本的 FP8 量化逻辑。

### 修复（紧急）
- 自动回滚（灰度规则触发：thumbs-down rate > 5%）
- 灰度发布暂停
- 老模型恢复 100% 流量

### 长期改进
- 灰度规则加上"format 合规率"和"质量代理"
- Pre-production benchmark 强制要做
- Calibration set 与 prod workload 对齐

### 面试讲法
> "模型质量回归是 LLM 特有的故障类型。HTTP 200 不代表没事。我们靠 thumbs-down rate 和 format 合规率作为代理指标，配合自动回滚。"

---

## 故障 6：单卡的 ECC 错误导致输出"偶尔"错

### 症状
- 1% 的请求输出明显错误（但不是乱码）
- 没有 OOM、没有 crash
- 间歇出现，不规律

### 排查
```
nvidia-smi -q -d ECC | grep "Volatile"  # ECC error 计数非零
dmesg | grep -i "xid"                   # 看到 Xid 错误
```

### 根因
一张 H100 的 HBM 有零星 ECC 错误，导致 weight 偶尔 bit flip。

### 修复（紧急）
- 隔离该节点
- Pod 重新调度到其他节点

### 长期改进
- DCGM 监控 ECC 错误（`DCGM_FI_DEV_ECC_DBE_VOL_TOTAL`）
- 节点健康检查自动 cordon 有 ECC 历史的卡
- 重要任务跑前 nccl-tests + memcheck

### 面试讲法
> "ECC 错导致输出错是最'诡异'的故障之一——不 crash、不 OOM，但答案是错的。靠 DCGM 监控提前发现，否则只能靠业务投诉。"

---

## 故障 7：CUDA Graph capture 失败，服务起不来

### 症状
- 部署新版本，Pod 启动后崩溃
- log："RuntimeError: CUDA error during graph capture"

### 排查
- vLLM 版本：新升级到 v0.x.x，CUDA Graph 与新 attention backend 兼容性问题
- 试 `--enforce-eager`：能起，但跑慢 30%

### 根因
新版本的某个 op 不支持 CUDA Graph，但配置默认开启。

### 修复（紧急）
- 临时 `--enforce-eager`，先把服务起来
- 或回滚 vLLM 版本

### 长期改进
- 升级 vLLM 必须先在 staging 跑通完整 benchmark
- CUDA Graph capture 错误 fail-fast 报警
- 框架升级走金丝雀

### 面试讲法
> "框架升级踩坑很常见。`--enforce-eager` 是 LLM 推理的 'safe mode'，临时禁掉 CUDA Graph 优化保持可用。"

---

## 故障 8：长上下文请求耗光 KV，把短请求挤死

### 症状
- 短 chat 请求 TTFT 飙
- 看 batch：里面有几个 100k token 长上下文请求
- KV usage 一直 > 95%

### 排查
- 流量类型 mix：发现 RAG team 上了新功能，prompt 5-100k token
- 当前部署没有 isolation：长 / 短请求混在同一 Pod 池

### 根因
长上下文请求占用 KV 远多于短请求，导致大量短请求被踢/排队。

### 修复
1. 紧急：Gateway 根据 prompt 长度路由：> 16k 走专门 Pod 池
2. 那个池关 chunked prefill 默认，调大 max_num_batched_tokens
3. 短请求池保持原配置

### 长期改进
- 永久建立两个 Pod 池：short context (8k) + long context (100k+)
- Smart router 自动识别 prompt 长度路由
- Gateway 在路由前 tokenize 预估长度

### 面试讲法
> "长上下文请求是个流量隔离问题。我们建立两个独立 Pod 池，Gateway 按 prompt 长度路由。这是 vLLM/LLM 部署里很重要的'workload partitioning'思想。"

---

## 写好 runbook 的 5 条原则

每个 runbook 应该有：

1. **明确的 symptom**：用户 / 监控看到什么
2. **可机械执行的检查命令**：复制粘贴就能跑
3. **二分诊断**：每一步缩小怀疑范围
4. **紧急 vs 长期分离**：先止血再治本
5. **跟代码 / 配置位置关联**：vLLM 源码、Pod yaml、Helm values

例子骨架：

```
## TTFT 飙升 Runbook

Symptom: Grafana alert "TTFT_P99_HIGH" fires

Triage (< 5 min):
  - queue depth (PromQL)
  - kv usage (PromQL)
  - preempt rate (PromQL)

Decision tree:
  - queue > 10 + kv > 0.9 + preempt > 0 → 容量不够 (see scaling-emergency.md)
  - queue == 0 + cache hit 跌 → 路由问题 (see routing-debug.md)
  - 不稳定（间歇）→ 单 Pod 异常 (see pod-debug.md)

Emergency actions:
  1. kubectl scale lws/<name> --replicas=+2
  2. 临时降级 (--max-tokens 上限 2048)
  3. 通知 on-call

Root cause analysis: follow standard postmortem template
```

---

## 面试常见追问

**Q: 你最难处理的 LLM 故障是哪个？**
A: 挑 NCCL hang 或 ECC 这种"症状诡异、调试难"的——故事性强。

**Q: 怎么减少故障 MTTR（平均恢复时间）？**
A: ①完整 runbook ②自动告警 + 自动诊断脚本 ③on-call 培训 ④chaos 演练熟悉故障模式 ⑤关键操作 self-service（一键扩容、一键回滚）。

**Q: 怎么避免重复故障？**
A: 每次 postmortem 必须出 action items：①监控覆盖（如果当时没告警，加告警）；②自动化（如果当时手动操作，写脚本）；③演练（下次纳入 chaos 计划）。

**Q: postmortem 怎么写？**
A: 标准结构：summary → timeline → impact → root cause → contributing factors → action items → lessons learned。**最重要的是 blameless**——不点名指责，只看流程改进。

---

## 小结

- 8 个高频故障覆盖 LLM 推理的所有典型失效维度：容量/通信/路由/重试/质量/硬件/框架/隔离。
- 任何故障都按 "Symptom → Triage (<5min) → Root cause → Emergency fix → Long-term" 五段写。
- 紧急动作只追求"止血"，长期改进必须落地为监控、配置或自动化代码改动。
- LLM 特有故障（NCCL hang、ECC bit flip、模型质量回归）单靠传统 5xx/latency 告警发现不了，必须额外加代理指标。
- Blameless postmortem 是把每次故障变成系统改进的关键文化。

## 自检

1. 复述"NCCL hang"故障的 5 段结构，并写出第一时间要敲的 3 条命令。
2. TTFT 飙升时，金信号三件套是哪三个？分别用哪条 PromQL 看？
3. 给一个 8k chat + 100k RAG 混部场景，画出 Gateway 按 prompt 长度路由的伪代码。
4. 写出一份 blameless postmortem 的章节顺序，并说明 "action items" 至少应该覆盖哪 3 类。

## 下一步

- 横向延展：[`09-advanced-features/`](../09-advanced-features/01-sampling-and-logits.md) 把高级特性也纳入 runbook 视角（如 LoRA 加载失败、structured output 校验失败）
- 想看源码：`vllm/v1/engine/core.py`（engine 主循环，理解 hang 怎么发生）、`vllm/v1/core/sched/scheduler.py`（preempt 路径）
- 想动手：[`07-hands-on/04-profiling-and-debugging.md`](../07-hands-on/04-profiling-and-debugging.md) 用 py-spy / nsys 复现卡死与抖动

---

## 总章节小结

整个 `08-production-deployment/` 7 章覆盖：

1. 参考架构（llm-d / AIBrix / vLLM Production Stack）
2. 智能路由（cache-aware / load-aware / LoRA-aware）
3. Gateway / Service Mesh（Istio + ExtProc + EPP）
4. 自动扩缩与容量
5. SLO 与可观测性
6. 稳定性与失效模式
7. 真实故障 runbook（本节）

这套加在前面的 vLLM 内部原理之上，构成了"原理 + 工程"的完整图。
面试官问到任何一个层面，都能展开讲，且能讲到代码级 / 工程级深度。

---

## Sources

- [Production-Grade LLM Inference at Scale with KServe, llm-d, and vLLM](https://llm-d.ai/blog/production-grade-llm-inference-at-scale-kserve-llm-d-vllm)
- [LLM Observability: A Complete Guide to Monitoring Production Deployments](https://inference.net/content/llm-observability-monitoring-production-deployments/)
- [The P99 Problem: Designing LLM Inference for Real Users](https://agentnativedev.medium.com/the-p99-problem-designing-llm-inference-for-real-users-11deb35bb8d4)
- [Service Mesh Debugging: When Istio Breaks Your Inference Pipeline](https://www.kubenatives.com/p/service-mesh-debugging-when-istio)
