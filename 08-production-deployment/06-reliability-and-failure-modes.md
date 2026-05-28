# 06. 稳定性：LLM 推理的失效模式与防护

> **谁该读这一篇？** 负责稳定性、on-call、容灾演练的 SRE / 平台工程师。
>
> **前置阅读：** [`02-architecture.md`](../01-overview/02-architecture.md)、[`05-slo-and-observability.md`](./05-slo-and-observability.md)、[`02-continuous-batching.md`](../02-core-concepts/02-continuous-batching.md)（理解 preempt 行为）
>
> **耗时：** 约 30 分钟
>
> **学完能：**
> 1. 把 LLM 推理失效分到 6 大类（资源/通信/调度/模型/框架/上下游）并对每类给出至少 1 种防御
> 2. 检测并自愈 NCCL hang、GPU OOM、preempt cascade、retry storm
> 3. 设计灰度发布 + 自动回滚阈值
> 4. 写出"上线前 checklist"和典型 chaos 演练剧本

LLM 推理服务的"挂掉"姿势比普通微服务多得多——GPU OOM、NCCL hang、KV cascade、CUDA Graph 异常、长尾尾巴拖死一整批请求……本节系统梳理失效模式，给出对应的工程对策。

---

## 1. LLM 特有的失效模式分类

```mermaid
flowchart LR
    Root["LLM 推理失效模式"]
    R1["1. 资源类<br/>· GPU OOM（KV 爆 / activation 高峰）<br/>· HBM 带宽饱和<br/>· NVLink/IB 链路抖动"]
    R2["2. 通信类<br/>· NCCL hang（一卡停响应，整组卡死）<br/>· RDMA 异常 / GPU-Direct 失效<br/>· Mesh sidecar 拦了 NCCL"]
    R3["3. 调度类<br/>· Preempt cascade（KV 不够 → 频繁踢）<br/>· Queue 雪崩（流量突增）<br/>· 长尾长请求拖死 batch"]
    R4["4. 模型类<br/>· 模型权重损坏 / 版本不一致<br/>· 量化精度异常（输出乱码）<br/>· LoRA 加载失败"]
    R5["5. 框架类<br/>· CUDA Graph capture 失败<br/>· torch.compile 编译失败<br/>· Tokenizer 不一致"]
    R6["6. 上下游<br/>· Gateway / Smart Router 故障<br/>· 客户端 retry storm<br/>· 上游限流误伤"]
    Root --> R1
    Root --> R2
    Root --> R3
    Root --> R4
    Root --> R5
    Root --> R6

    classDef hw    fill:#fee2e2,stroke:#b91c1c,color:#1a1f29;
    classDef comm  fill:#fef3c7,stroke:#b45309,color:#1a1f29;
    classDef sched fill:#eff5ff,stroke:#2563eb,color:#1a1f29;
    classDef model fill:#dcfce7,stroke:#15803d,color:#1a1f29;
    classDef fw    fill:#f7f8fa,stroke:#5b6573,color:#1a1f29;
    class R1 hw;
    class R2 comm;
    class R3 sched;
    class R4 model;
    class R5 fw;
    class R6 hw;
```

下面挑高频且容易被忽视的几个详谈。

---

## 2. GPU OOM：最常见也最隐蔽

### 2.1 OOM 不一定是 KV 满

vLLM 启动时 profile run 已经预留好 KV cache。运行时 OOM 通常是：

1. **激活的临时高峰**：特定 batch + seq_len 组合下 activation 比 profile 时高
2. **多模态 encoder 高峰**：vision encoder 处理多张大图
3. **CUDA Graph workspace** 占用：录制时分配
4. **碎片化**：长时间运行后 PyTorch caching allocator 内部碎片
5. **第三方进程**：同卡有别的容器抢显存（不该但发生过）

### 2.2 防御
- `gpu_memory_utilization` 别上 0.95，留 5-10% 缓冲
- 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片
- DCGM 监控 `DCGM_FI_DEV_FB_USED`，临界值告警
- Pod 设 `nvidia.com/gpu.memory` resource，K8s 调度感知

### 2.3 OOM 时怎么办
- vLLM 通常会先抛 `torch.cuda.OutOfMemoryError`
- 配合 `restartPolicy: Always`，自动重启
- 但**多次 OOM** 意味着配置问题（KV 算大了），需告警而非依赖自愈

---

## 3. NCCL Hang：分布式推理的噩梦

### 3.1 症状
TP=8 部署，某个时刻所有 8 个 Pod 全部"看似正常运行"但没有任何 GPU 活动。

- `vllm:num_requests_running` 不变
- Pod 没崩、健康检查通过
- 但是新请求全部超时

### 3.2 根因
一个 Pod 内的 NCCL 集合通信（AllReduce）卡住了：

- 某张卡卡了一次 kernel
- 其他卡在等它，整组通信 hang

### 3.3 检测
```bash
# 在 Pod 里看
nvidia-smi pmon -i 0   # 一段时间无活动
NCCL_DEBUG=INFO  # 启动时开
```

NCCL 提供 watchdog：`NCCL_BLOCKING_WAIT=1` + 超时机制能让 hang 转 crash。

### 3.4 防御
1. **NCCL watchdog 超时**：`NCCL_TIMEOUT=60`（秒），过期 abort 整组
2. **Pod 整组重启**：LeaderWorkerSet 检测到任一 Pod 崩，整组重启
3. **Liveness probe**：探测点不要光是 `/health`，要包含"上一次成功 step 在 N 秒内"
4. **告警**：`vllm:num_requests_running > 0 但 token throughput == 0` 持续 1 分钟

### 3.5 排查
NCCL log + `py-spy dump` Python 栈 + `nvidia-smi nvlink -gt e`（看 NVLink 错误）。

---

## 4. Preempt Cascade：调度雪崩

### 4.1 情景
KV 接近满 → 新请求来 → preempt 一个 running → 那个 running 被重新调度 → KV 又满 → preempt 另一个 → 反复…
表现：`num_preemptions_total` 像火山喷发，TTFT/TPOT 全崩。

### 4.2 防御

**入场准入控制**（admission control）：

- KV usage > 阈值（如 0.85）时拒绝新请求或排队
- 而不是接收后再 preempt

**优先级队列**：

- 已 running 的请求 priority 高于刚进的
- 一个请求被 preempt 一次后下次更高优先级（避免反复牺牲）

**Long context 隔离**：

- 100k+ token 请求路由到专门 Pod
- 不和 chat 请求竞争同一 KV pool

**配置调优**：

- `--max-num-seqs` 适当下降，避免过度并发
- `--max-num-batched-tokens` 防长 prefill 抢 KV

### 4.3 vLLM Scheduler 内置的保护
- 同一请求被 preempt 一次后会 appendleft 到 waiting，下次最优先恢复
- preempt 优先选 waiting 时间最短的（年轻请求）

---

## 5. 长尾长请求：拖死整个 batch

### 5.1 问题
一个请求 `max_tokens=999999` 加 `temperature=0.0`，可能停不下来生成几百万 token。

- 占着 KV 不放
- 让 batch 平均处理时间变长
- 影响其他用户的 TPOT

### 5.2 防御

**Gateway 强制 max_tokens 上限**：

- 普通用户 `max_tokens ≤ 4096`
- 长文档生成业务专用 endpoint

**Repetition detection**：

- vLLM 的 stop_token、stop_string
- 或自定义：n-gram 重复检测，触发后强制 EOS

**Server-side timeout**：

- `--max-model-time-per-request 30s`（伪 API，按实际版本）
- 超时强制结束

**计费导向**：

- 按 token 计费天然约束滥用

---

## 6. 客户端 Retry Storm

### 6.1 场景
LLM 服务一时抖动 → 客户端 SDK 自动重试 → 流量翻倍 → 服务彻底崩 → 恶性循环。

### 6.2 防御

**Server 端：限流 + 区分 503/429**

- 503：服务真有问题，客户端不该立刻重试
- 429：你超 quota，客户端不该 retry 这个请求

**Client SDK：指数退避 + jitter**

- 第一次 1s，然后 2s, 4s, 8s, + random jitter
- 重试 budget：总重试次数 / 时间窗内有上限

**Server 端：负载丢卒保车**

- 流量极高时主动返回 429 给低优先级客户端
- 保关键客户

---

## 7. Failure Mode 分类与应对（速查表）

| Failure                 | 检测                                    | 自愈                  | 人工动作            |
| ----------------------- | ------------------------------------- | ------------------- | --------------- |
| Pod OOM                 | exit code, `node_oom_kills`           | K8s restart         | 检 KV 配置          |
| GPU OOM                 | torch error                           | Pod restart         | 降 KV / 关 CG     |
| NCCL hang               | throughput 0 + running > 0            | Pod 整组重启            | 看 NCCL log     |
| Preempt cascade         | preempt rate 持续高                      | admission control 自动放慢 | 调 max_num_seqs |
| Long-tail request       | 单 req 跑超 5min                          | 强制 EOS                | 限 max_tokens   |
| Retry storm             | 请求量异常                                  | 503 + ratelimit      | client SDK 改   |
| Tokenizer drift         | cache hit rate 突跌                     | n/a                 | 检模型版本           |
| CUDA Graph 异常          | 启动失败 / 输出乱                            | enforce-eager       | 调小 capture size|
| 模型权重损坏              | 输出全 garbage                          | replica fallback   | 重新下载            |
| Pod scrape timeout      | Prom up=0                            | restart            | 检 metrics 端点    |

---

## 8. 灰度发布与回滚

模型版本变更是高风险操作。生产惯例：

### 8.1 灰度策略

```mermaid
flowchart LR
    S0["旧版本<br/>100%"]
    S1["Canary 5%<br/>观察 30 min"]
    S2["10%<br/>观察 1-4 h"]
    S3["30%<br/>观察 1-4 h"]
    S4["50%<br/>观察 1-4 h"]
    S5["新版本 100%"]
    Rollback["自动回滚<br/>error rate > 0.5% ·<br/>TTFT p99 涨 > 20% ·<br/>thumbs-down rate 升"]

    S0 --> S1 --> S2 --> S3 --> S4 --> S5
    S1 -.->|超阈值| Rollback
    S2 -.-> Rollback
    S3 -.-> Rollback
    S4 -.-> Rollback
    Rollback --> S0

    classDef stage fill:#eff5ff,stroke:#2563eb,color:#1a1f29;
    classDef bad   fill:#fee2e2,stroke:#b91c1c,color:#1a1f29;
    class S0,S1,S2,S3,S4,S5 stage;
    class Rollback bad;
```

### 8.2 蓝绿
适合"完全不同模型"切换：

- 新模型 cluster 起来跑 staging
- Gateway 一键切流
- 老 cluster 留 24h 准备回滚

### 8.3 Shadow Traffic
新模型不返回给用户，但收一份请求 → 对比新旧输出质量。
适合：评估，不适合：性能 SLO 验证（shadow 一般无 SLO 流量）。

---

## 9. Chaos Engineering：主动制造故障

定期演练，避免"没演过的故障真发生时手足无措"：

| 演练                  | 期望系统反应                       |
| ------------------- | ---------------------------- |
| 干掉一个 vLLM Pod      | LB 切流到其他 Pod，无 5xx          |
| 节点 NIC 断 30s       | Pod 检测到，自动重启              |
| GPU 故障注入          | Pod 重启，请求重路由               |
| Gateway 抖动 5s      | SDK 自动重试，无客户端报错            |
| 注入 KV 高水位         | preempt 增加但 SLO 不破          |
| 上游限流（gateway 故障）  | Pod 不雪崩，等待恢复               |

工具：Litmus、ChaosMesh、Gremlin、k8s-fault-injector。

---

## 10. 高可用部署原则

| 原则                         | 含义                              |
| -------------------------- | ------------------------------- |
| 跨可用区分布                  | Pod 跨 AZ 分布，单 AZ 故障不影响整体        |
| 多 region                  | 大区域故障切流（DNS + global LB）          |
| 主备 Gateway                | Gateway 自己也要 HA                   |
| Multi-tenant 隔离           | 一个租户的 retry storm 不影响其他             |
| 关键路径无 SPOF              | 任何单点死了都有 fallback                 |
| 演练频次                     | 每月一次完整故障演练，每周一次小演练               |

---

## 11. 一份"上线前"checklist

```
□ HPA / KEDA 配置 + cooldown 足够
□ preStop hook + terminationGracePeriod ≥ 600s
□ readinessProbe 在 model load 完成才 ready
□ NCCL watchdog 超时配置
□ DCGM monitoring（GPU 健康）
□ OOM auto-restart
□ Smart router 有降级到 round-robin
□ Gateway 限流（RPS + token + concurrent）
□ 强制 max_tokens 上限
□ 客户端 SDK 用指数退避 + jitter
□ 灰度发布 + 自动回滚阈值
□ Chaos 演练通过
□ 跨 AZ / 跨 region 分布
□ runbook 写好（详见 07）
□ on-call 培训
```

---

## 12. 面试常见追问

**Q: 一个 vLLM Pod 突然不响应，怎么排查？**
A: ①看 throughput（0 + running>0 = NCCL hang）；②看 GPU util；③`py-spy dump` Python 栈；④检查 mesh sidecar；⑤强制重启 Pod 看是否恢复。

**Q: KV cascade 怎么处理？**
A: 入场准入控制（KV > 阈值拒绝新进）；优先级队列（保 running）；长 context 隔离；调小 max_num_seqs。Long term：扩容或量化。

**Q: 重试在 LLM 下要注意什么？**
A: ①SSE 第一帧后不能重试；②不要重试同一 Pod（让 router 重新选）；③重试 budget；④区分 5xx 类型（503 vs 429 处理不同）。

**Q: 怎么演练 GPU 故障？**
A: ①`nvidia-smi -i N -p 0` kill 进程；②xid 错误注入；③拔卡（极端，物理机）；④用 GPU operator 的 fault injector。生产前最好都演练。

**Q: 模型质量 regression 怎么发现？**
A: ①shadow 流量对比新旧输出；②用户反馈 thumbs；③离线 eval（金标集）；④EOS rate、格式合规率等代理指标。多层防护。

---

## 小结

- 失效模式分 6 类：资源、通信、调度、模型、框架、上下游；逐类设防御比"加 retry"有效得多。
- NCCL hang 是分布式推理的头号噩梦，必须用 `NCCL_TIMEOUT` + LWS 整组重启 + "running>0 但 throughput=0" 告警三层防护。
- Preempt cascade 用 admission control + 优先级队列 + 长上下文隔离来斩断。
- 重试在 LLM 下要分清 503/429、SSE 第一帧前后、retry budget，否则 retry storm 会自己把服务打死。
- 灰度发布、自动回滚、chaos 演练是把"未知未知"变成"已知"的常规手段。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. TP=8 Pod 组：8 个 Pod CPU 5%、GPU util 0%、`num_requests_running > 0` —— 第一反应？检测路径？**

**第一反应**：**这是 NCCL hang！** 请求"在跑"（scheduler 觉得有 running），但 GPU 没动（util 0）、CPU 也没动（5% 是基础轮询）—— 典型死锁特征。

**检测路径**：

```bash
# Step 1: 看请求是不是真在 running
curl http://pod:8000/metrics | grep -E "num_requests_(running|waiting)"
# 如果 running > 0 但 waiting 也涨 → 卡死无产出

# Step 2: py-spy 看 Python 栈
for pod in $(kubectl get pods -l app=vllm -o name); do
    kubectl exec $pod -- py-spy dump --pid 1
done | tee /tmp/dumps.txt
grep -A5 "all_reduce\|c10d" /tmp/dumps.txt
# 典型表现：所有 worker 都在 c10d.work.wait()

# Step 3: 看 NCCL 调用一致性
NCCL_DEBUG=INFO ; 重启服务看 log
# Step 4: 检查物理链路
kubectl exec <pod> -- nvidia-smi nvlink -e
# CRC 错误 / link down → 物理硬件问题
```

**典型根因**：

1. **不同 rank 看到不一致的 batch shape**（scheduler bug，scheduler 没同步广播）
2. **某 rank 慢了几百 ms**（GC pause、CPU 抢占等）触发其他 rank wait 超时
3. **NVLink 物理故障**
4. **NCCL 版本与驱动不匹配**（升级 CUDA driver 但忘了升级 NCCL）

**应急**：直接 `kubectl delete pod -l app=vllm` 整组重启（LWS 会自动重建）。

---

**2. KV 使用率长期 > 0.9, preempt rate 每分钟数十，3 个独立缓解手段。**

**缓解手段（独立 + 立即生效）**：

| 手段 | 操作 | 收益 | 副作用 |
| --- | --- | --- | --- |
| **1. 扩 pod 数** | HPA 触发 / 手动 `kubectl scale` | 横向分流，每 pod 负载降 | 成本上升；需几十秒 warm |
| **2. 降 `max_num_seqs`** | rolling update `--max-num-seqs 32`（原 64）| 单 pod 并发减半 → KV 占用减半 → preempt 消失 | 单 pod 吞吐降 |
| **3. 切 FP8 KV** | rolling update `--kv-cache-dtype fp8` | 单 token KV 减半 → 同显存装 2× 请求 | 精度损失 < 1% |
| 4. admission control 拒新请求 | gateway 返 429 给客户端 | 立即降压（保护现有请求）| 部分用户被拒 |
| 5. 临时切到长 chunked prefill | `--max-num-batched-tokens 2048` | step 时长稳定，减少 KV 峰值瞬时占用 | TTFT 略升 |

**实战顺序**：

1. **立即**：admission control 拒新请求（保护当前用户）
2. **5 分钟内**：扩 pod（HPA 自动 / 手动）
3. **若 HPA 太慢**：rolling update 降 max_num_seqs 或切 FP8
4. **长期**：复盘容量规划，提高 utilization 目标 或 拉长高峰扩容窗口

→ 关键是**多手段并行**，单一手段都有滞后或副作用。

---

**3. 灰度 5% 阶段，定 3 个自动回滚阈值？覆盖什么风险？**

| 阈值 | 数值 | 覆盖风险 |
| --- | --- | --- |
| **TTFT p99 相对基线退化 > 30%** | `histogram_quantile(0.99, new) / histogram_quantile(0.99, baseline) > 1.3` | 性能回归（新版本 scheduler bug、kernel 选择失误、模型 load 异常）|
| **错误率 > 1%** | `rate(request_success{finished_reason="abort"}) / rate(request_success) > 0.01` | 正确性问题（模型输出乱码、grammar 失效、attention bug） |
| **OOM / corrupted_requests > 0** | `increase(corrupted_requests_total[5m]) > 0` OR pod OOMKilled count > 0 | 完全无法跑（显存配置错、新代码路径 bug） |

**第 4 个常用**：
| **KV cache hit rate 退化 > 50%** | `cache_hit_rate_new < cache_hit_rate_baseline × 0.5` | prefix caching 失效（hash 算法改了、cache key 不兼容） |

**实施**：

- 用 Argo Rollouts / Flagger 等 progressive delivery 工具
- 每个阈值连续命中 N 分钟（比如 3 分钟）才触发，避免毛刺
- 触发后自动 abort rollout + 通知 oncall

**5% → 25% → 50% → 100% 灰度**：每个 stage 持续 30 分钟观察，阈值不变。

---

**4. Chaos 演练：模拟"单 Pod NCCL hang"，期望自愈时间 + 用户侧表现？**

**演练设计**：

```bash
# Step 1: 选一个生产 pod 注入故障
target_pod=$(kubectl get pods -l app=vllm -o name | head -1)

# Step 2: 注入 NCCL hang
# 方法 A: kill 一个 worker 进程 → 其他 worker AllReduce wait
kubectl exec $target_pod -- pkill -STOP worker_0
# 方法 B: 用 chaos-mesh 注入网络分区
# 方法 C: 用 sigstop 暂停所有 NCCL 端口

# Step 3: 观察自愈
watch kubectl get pods -l app=vllm
```

**预期时间线 + 用户侧表现**：

| 时间 | 系统行为 | 用户侧 |
| --- | --- | --- |
| T=0 | 注入 hang | 部分请求开始挂在那个 pod |
| T=10s | 该 pod liveness probe 失败（vLLM 没回 /health）| TTFT 飙、SSE 帧间隔大 |
| T=15-30s | K8s kill 整组 pod (LWS 触发) | 该 pod 上的请求 5xx 返回；其他 pod 正常 |
| T=30-90s | LWS 重建 pod 组、NCCL 重新初始化 | 部分用户 retry，落到其他 pod，恢复正常 |
| T=90-120s | 新 pod ready，加入服务池 | 全集群恢复 |

**目标 SLA**：

- **MTTR < 2 分钟**
- **影响用户比例 < 1/N（N 为 pod 总数），retry 后 100% 成功**
- **整体 SLO 不受影响**（按 5 分钟滑动窗口）

**演练通过标准**：

- LWS 自动重建（不需人工 `kubectl delete pod`）
- 客户端 retry 后请求成功（gateway 摘除挂掉 pod 的速度足够快）
- 监控告警在 T=10s 内触发（不需 oncall 来检查）

**演练失败的常见 root cause**：

- liveness probe 太宽松（如 60s 才检测 hang）→ 改成 10s
- LWS 没配（用了 Deployment）→ 改 CRD
- gateway 健康检查间隔太久（5min）→ 改 10s
- client retry policy 没配 → 教育业务方

→ 演练目的不是"看看会怎样"，是**验证你设计的恢复机制确实工作**。

## 下一步

- 下一节：[`07-incident-playbook.md`](./07-incident-playbook.md)（把这些理论转成可执行 runbook）
- 想看源码：`vllm/v1/core/sched/`（preempt 与调度）、`vllm/distributed/`（NCCL/通信）
- 想动手：[`07-hands-on/04-profiling-and-debugging.md`](../07-hands-on/04-profiling-and-debugging.md) 主动制造 OOM/preempt 验证防护

