# 容量计算与故障诊断练习：公式、单位、证据

> **谁该读这一篇？** 会讲概念但容易在白板计算、单位换算或线上排障追问中失分的候选人和在岗工程师。
>
> **前置阅读：** [`01-common-questions.md`](./01-common-questions.md)、[`02-system-design.md`](./02-system-design.md)、[`04-autoscaling-and-capacity.md`](../08-production-deployment/04-autoscaling-and-capacity.md)
>
> **耗时：** 约 60 分钟
>
> **难度：** 高级
>
> **当前性说明：** 本章按 vLLM `b23bd73f540175f9e117eaee5029cd7d8df63964` 静态复核；所有数值题都是明确假设下的教学演算，不是当前 SHA 的硬件 benchmark。
>
> **学完能：**
> 1. 计算 weight、KV、request、batch、world size、goodput 与 replica reserve
> 2. 每一步保留单位、假设、中间值和 sanity check
> 3. 对八类常见故障先列证据，再做单变量缓解与回滚

## 0. 白板计算协议

先写四行，再按计算器：

1. **Asked**：题目要的是理论下界、每 rank、每 replica，还是集群总量？
2. **Assumptions**：模型 config、dtype、并行、长度分布、SLO、reserve。
3. **Formula + units**：每个乘数都写单位，确认能约掉。
4. **Sanity check**：输入翻倍时结果应怎样变化？与物理上限是否同量级？

常用换算：`1 KiB=1024 byte`，`1 MiB=1024 KiB`，`1 GiB=1024 MiB`；厂商常用十进制 GB，GPU / OS 工具显示口径可能不同，答题时先声明。

---

## 1. 权重内存：理论下界不等于可部署

### 题目

一个实际加载 `13×10^9 params` 的 dense 模型以 BF16 (`2 byte/param`) 加载；假设 checkpoint / padding / scale 之外的**权重相关**额外开销为理论权重的 3%。TP=2，所有权重均匀分片。求每 rank 权重相关内存。它能否证明两张 16 GiB GPU 可部署？

### 解答

**假设：** 这里只算 weights；不包含 CUDA context、activation、workspace、KV、graph 与 allocator reserve。

理论权重：

$$13\times10^9\ param\times2\ byte/param=26\times10^9\ byte$$

换成 GiB：

$$26\times10^9\ byte\div2^{30}\ byte/GiB\approx24.21\ GiB$$

加 3%：

$$24.21\ GiB\times1.03\approx24.94\ GiB$$

TP=2 每 rank：

$$24.94\ GiB\div2\ rank\approx12.47\ GiB/rank$$

**答案：** 每 rank 约 `12.47 GiB` 权重相关内存。不能据此证明 16 GiB 可部署，因为只剩约 `3.53 GiB` 给所有 non-weight 与 KV，且“均匀分片”本身也可能不成立。

**sanity check：** TP 从 2 改 1，均匀分片假设下每 rank 应翻倍到约 24.94 GiB。

**面试追问：** 哪些参数可能每 rank 复制？INT4 为什么不能简单按 `0.5 byte/param` 当物理占用？

---

## 2. KV bytes/token：用 KV heads，不用 query heads

### 题目

一个 decoder 有 `L=32 layer`、`H_kv=8 KV heads`、`D_head=128 element/head`，KV dtype 为 BF16 (`2 byte/element`)。求逻辑 KV bytes/token。

### 解答

公式：

$$B_{KV/token}=2(K,V)\times L\times H_{kv}\times D_{head}\times b_{dtype}$$

代入：

$$2\times32\ layer\times8\ head/layer\times128\ element/head\times2\ byte/element$$

$$=131{,}072\ byte/token=128\ KiB/token$$

**答案：** `128 KiB/token`。

**sanity check：** KV dtype 改为 1 byte 时理论值减半；KV heads 从 8 到 4 时也减半。若使用 query heads 32，结果会错误放大 4 倍。

**边界：** TP / PP 后每 rank 容量、MLA、hybrid cache 与 padding 要看实际 `KVCacheSpec`，不能只除 world size。

---

## 3. 单请求 KV：prompt 与已生成 token 都占容量

### 题目

沿用 `128 KiB/token`。请求有 `1,500 input token`，最多生成 `500 output token`。求最大逻辑 KV；若平均只生成 120 token，平均逻辑 KV 是多少？

### 解答

最大 tokens：

$$1{,}500+500=2{,}000\ token$$

最大 KV：

$$2{,}000\ token\times128\ KiB/token=256{,}000\ KiB=250\ MiB$$

平均 tokens：

$$1{,}500+120=1{,}620\ token$$

平均 KV：

$$1{,}620\times128\ KiB=207{,}360\ KiB=202.5\ MiB$$

**答案：** 上限 `250 MiB/request`，给定输出均值下约 `202.5 MiB/request`。

**sanity check：** 输出上限增加 500 token，应增加 `500×128 KiB=62.5 MiB`，而不是重算权重。

**面试追问：** admission 应用 max 还是分位数？prefix sharing 会改变物理占用但为什么不能作为安全上限？

---

## 4. Batch / 并发容量：先做上界，再交给 profile

### 题目

一个 replica 启动后报告可供 KV 使用的实际容量为 `30 GiB`。保留 10% KV 容量给长度长尾 / block 尾部 / 运行波动；按上一题最大 `250 MiB/request`，求只受 KV 限制的保守并发上界。`max_num_seqs` 应直接设为这个值吗？

### 解答

可用 KV：

$$30\ GiB\times1024\ MiB/GiB\times(1-0.10)=27{,}648\ MiB$$

请求数：

$$\left\lfloor27{,}648\ MiB\div250\ MiB/request\right\rfloor=110\ request$$

**答案：** KV-only 保守上界约 110 active requests。

不能直接把 `max_num_seqs=110` 当生产结论，因为：

- 实际长度是分布，page allocation / block size 会离散化；
- scheduler token budget、activation / batch workspace 与 model runner 也有限制；
- SLO 可能在 KV 满之前就因 batch 太大失败；
- multimodal encoder cache、LoRA slots 或 speculative state 还占资源。

**sanity check：** 若 bytes/token 或每请求 tokens 翻倍，上界应近似减半。

**验证：** 从较低并发阶梯压测，只有在 p99、错误率、preemption 和显存余量同时通过时提高。

---

## 5. TP / PP / DP / EP world size：先画坐标

### 题目

两节点，每节点 8 GPU，共 16 ranks。单模型副本配置 TP=4、PP=2。可以放多少个 DP replicas？若 MoE 配置 EP=4，总 rank 是否变成 `4×2×2×4=64`？

### 解答

单副本 ranks：

$$R_{model}=TP\times PP=4\times2=8\ rank/replica$$

可放 DP：

$$DP=16\ rank\div8\ rank/replica=2\ replica$$

常见坐标可写 `(dp, pp, tp)`，共 `2×2×4=16`。

**EP=4 通常不是额外乘数。** 它在既有模型 ranks 上创建 expert-parallel group / 映射；要读启动模式与 `parallel_state.py`，确认 EP group 是否复用 TP / DP 维度。盲乘得到 64 与物理 16 ranks 矛盾。

**sanity check：** 每个 rank 必须恰好映射到一张 GPU；把所有 replica 坐标列出应正好 16 行且无重复。

**面试追问：** 如果要求每个 TP group 不跨节点，如何放置？PP stage 不均衡怎样从 trace 看出来？

---

## 6. Throughput 与 goodput：不要乘独立边际概率

### 题目

压测发送 120 request/s。观测到 100 request/s 返回成功，其中 96 request/s 同时满足 TTFT 与 TPOT deadline；golden sampling 显示这 96 中有 94 request/s 通过质量校验。求 transport throughput、SLO goodput、quality-adjusted goodput。

### 解答

- transport throughput：`100 successful request/s`。
- SLO goodput：`96 request/s`。
- quality-adjusted goodput：`94 request/s`。

相对输入的比例：

$$100/120=83.3\%$$

$$96/120=80.0\%$$

$$94/120=78.3\%$$

**关键：** 这里使用同一请求的 joint outcome。不能把“TTFT 90% 达标 × TPOT 95% 达标”相乘，除非证明独立；实际二者常因 queue / load 相关。

**sanity check：** `quality goodput ≤ SLO goodput ≤ successful throughput ≤ offered load`。

**面试追问：** output length 不同如何用 good output tokens/s补充？超时后仍完成的请求算什么成本？

---

## 7. Replica count、目标利用率与 N+1

### 题目

峰值 `λ_peak=350 request/s`。单 replica 在目标长度分布下，**满足全部 SLO**的 sustainable goodput `μ_good=55 request/s`。为吸收 burst，目标利用率 `ρ_target=0.75`；要求任一 replica 故障仍承载峰值，并在 rollout 时额外保留一个 canary replica。求总副本。

### 解答

steady replicas：

$$R_{steady}=\left\lceil\frac{350\ request/s}{55\ request/s/replica\times0.75}\right\rceil$$

$$=\lceil8.48\rceil=9\ replicas$$

加一个 failure reserve 和一个 rollout reserve：

$$R_{total}=9+1+1=11\ replicas$$

**答案：** 初始规划 11 replicas。

**N+1 sanity check：** 故障一台且 canary 不接生产流量时，9 个 steady replicas 仍在；峰值每台负载 `350/9=38.9 request/s`，低于 `55×0.75=41.25 request/s` 的目标。

**边界：** 若 canary 也占用一个原本的 steady replica，或 AZ 故障一次损失多台，reserve 公式必须重算。

---

## 8. KV transfer 下界：判断 disaggregation 是否可能成立

### 题目

P/D 拆分需要为一个请求传 `4 GiB` KV。测得有效单向带宽 `80 Gbit/s`，先忽略 queue / setup / serialization。求传输时间下界。若聚合部署相对拆分只多 250 ms compute interference，这个方案是否有胜算？

### 解答

先统一单位：

$$4\ GiB\times8\ bit/byte=32\ Gibit$$

若按题目近似把 `80 Gbit/s` 当 `80×10^9 bit/s`，严格换算后约：

$$T\approx\frac{4\times2^{30}\ byte\times8\ bit/byte}{80\times10^9\ bit/s}\approx0.429\ s$$

即理论下界约 `429 ms`，还未加 queue / setup。它已经大于只节省的 250 ms，因此按当前假设**没有端到端 latency 胜算**。

**sanity check：** 带宽翻倍，时间下界减半；传输数据翻倍，时间翻倍。

**面试追问：** 为什么实际有效带宽低于链路标称？如果 P/D 的价值是独立扩缩而非单请求 latency，应怎样重新评估？

---

## 9. 证据优先的故障诊断八题

统一答题顺序：`确认症状 → 校验观测 → 分层假设 → 只读取证 → 最小缓解 → 复测 → 回滚 / 根修`。下列 signal 名是类别；实际 Prometheus 名必须从当前 SHA `/metrics` inventory 选择。

### 场景 A：TTFT p99 上升，TPOT 正常

**先要证据：** offered load 与 input length buckets、waiting queue / queue time、tokenizer / renderer CPU、prefill tokens / step、cold model / prefix hit、gateway time。

**假设排序：** 长 prompt 或排队 > frontend CPU > cache locality collapse > cold / compile > 网络。

**最小动作：** 对超长输入做 admission / 独立 pool；扩 frontend 只在 CPU 证据成立时做。

**失败 / 回滚：** 降 token budget若让 queue 更长且 TTFT 更差，立即回滚；不能因 GPU util 低直接扩 GPU。

### 场景 B：TPOT p99 上升，TTFT 正常

**先要证据：** running / batch work、output length、KV usage / preemption、decode backend / graph fallback、collective、GPU memory bandwidth 与 power / thermal。

**假设排序：** decode batch / context 变大 > preemption / thrash > kernel fallback > NCCL / hardware。

**最小动作：** canary 降 active sequences 或 step token work，验证 TPOT 与 goodput共同变化。

**失败 / 回滚：** TPOT 恢复但 goodput 跌破容量门禁，不是完成修复；应扩容或分池。

### 场景 C：启动或运行 OOM

**先要证据：** 每 rank启动 profile、weights、non-KV peak、KV page count、graph capture、LoRA / encoder cache、OOM stack 与同节点其他进程。

**区分：** 启动 OOM、capture OOM、首个大请求 OOM、长期碎片 / 泄漏不是一类。

**最小动作：** 恢复已验证 memory / graph / concurrency config；先 admission，后优化。

**失败 / 回滚：** 禁止无界降低 `gpu_memory_utilization` 后反复重启；要保留原 profile 与可复现输入。

### 场景 D：Preemption 持续增长

**先要证据：** KV usage、running / waiting、context / output length、max sequences、prefix hit tokens、abort / retry。

**假设排序：** active working set 超 KV > 长度分布变化 > cache locality 降 > admission / retry 放大。

**最小动作：** 限 active sequences / output tokens，隔离长 context；观察重复 prefill 与 TTFT。

**失败 / 回滚：** 只扩 `max_num_seqs` 往往加剧 thrash；发现 preemption 增长即撤销。

### 场景 E：NCCL hang 或多 rank timeout

**先要证据：** 第一个异常 rank、所有 rank stack / logs、rank map、GPU / NIC health、topology、collective type / size、版本和最近变更。

**假设排序：** 单 rank提前异常 > group / rank 配错 > link / NIC > collective mismatch > driver / library。

**最小动作：** drain 整个 model group / node failure domain，不把单 rank重启后塞回旧 group。

**失败 / 回滚：** 禁止只延长 timeout掩盖 collective 不一致；恢复必须做多 rank soak。

### 场景 F：Prefix-cache hit rate 突降

**先要证据：** hit **tokens** 与 query tokens、chat template / tokenizer / model revision、cache salt、LoRA / multimodal mix、router、prompt hash sample。

**假设排序：** workload / template 变化 > routing locality > capacity eviction > identity / salt改变 > metric 口径。

**最小动作：** 对比升级前后 tokenized prefix与路由，不先调大 cache。

**失败 / 回滚：** 若输出一致性异常，优先禁用 cache / 回滚版本，而不是追求 hit rate。

### 场景 G：GPU 低利用、frontend CPU 满

**先要证据：** tokenizer / renderer / JSON / media profile、API event-loop lag、engine queue、request size、API replicas 与 DP mapping。

**假设排序：** tokenization / chat rendering > media IO / processor > serialization / logprobs response > Python contention。

**最小动作：** 按 profile 扩 / 隔离 frontend，限制巨大请求；确保 API replica 与 engine topology支持该扩展。

**失败 / 回滚：** 不能在未知瓶颈时开启更多 API processes；processor cache / runtime LoRA 与多 API server 有约束。

### 场景 H：Retry storm 与 queue 不收敛

**先要证据：** original vs retry attempts、429 / 5xx / timeout、retry delay / budget、waiting age、client / gateway layers、cancellation传播。

**假设排序：** 无界同步 retry > timeout小于实际 service time > downstream partial outage > admission缺失。

**最小动作：** fail closed admission、retry budget、jitter、circuit breaker；停止重复工作比盲目扩容更快。

**失败 / 回滚：** queue下降但成功 goodput也归零说明 shedding过度；按优先级恢复并监测系统是否收敛。

---

## 10. 自己出题与评分

给自己换一组模型 config / dtype /长度，重做题 1–8。每题 5 分：

| 维度 | 1 分标准 |
| --- | --- |
| Assumption | 所有未给输入显式写出 |
| Formula | 公式与对象匹配 |
| Units | 每步单位可约分 |
| Intermediate | 中间值可复核 |
| Sanity | 有趋势 / 物理上限检查 |

故障题每题再按“证据、假设排序、最小动作、复测、回滚”各 1 分。任何一步引用不存在的 metric 或把估算称作硬件实测，该题最高 2 分。

> **生产取舍：** 容量表给的是决策边界，不是调度器的行为模拟。最终值必须由目标硬件 benchmark、failure test 与质量 gate闭环。

> **硬件验证状态：** 未执行当前 SHA GPU run；本章无硬件实测结果。

## 小结

- 权重决定能否放置，KV 决定 context / active capacity，service demand 决定 SLO 下 goodput。
- `TP×PP×DP` 先映射物理 ranks；EP 是否增加维度要看 group 定义。
- replica 公式必须用 SLO goodput，并加入 failure / rollout reserve。
- 排障先建立 joint evidence，单变量缓解后同时复测 SLO、quality 与 goodput。

## 自检

1. 为什么 `13B × 2 bytes` 不是 13B 模型的部署显存？
2. 为什么不能把 TTFT 达标率与 TPOT 达标率直接相乘？
3. 一个 AZ 同时损失三副本时，题 7 的 N+1 要怎样改？
4. Retry storm 中“扩容”和“先停止重复工作”分别何时有效？

## 下一步

- [`04-mock-interview-and-rubric.md`](./04-mock-interview-and-rubric.md)：把计算题放进完整五轮面试
- [`08-production-capstone.md`](../07-hands-on/08-production-capstone.md)：生成真实 benchmark / incident evidence

## Source trail

- `vllm/config/{cache,scheduler,parallel}.py`
- `vllm/v1/kv_cache_interface.py`、`vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/distributed/parallel_state.py`
- `vllm/entrypoints/`、`vllm/v1/engine/`
