# vLLM 工程面试：30 道三层回答

> **谁该读这一篇？** 已读过核心概念、源码走读与生产章节，准备把“知道名词”升级为“能解释机制、边界和验证方法”的工程师。
>
> **前置阅读：** [`PagedAttention`](../02-core-concepts/01-paged-attention.md)、[`入口与主循环`](../03-code-walkthrough/01-entry-points.md)、[`分布式并行`](../05-distributed/01-tp-pp-ep.md)、[`生产部署架构`](../08-production-deployment/01-deployment-architectures.md)
>
> **耗时：** 约 70 分钟；建议每题先口述，再看参考答案。
>
> **难度：** 中高级
>
> **当前性说明：** 本章按 vLLM `b23bd73f540175f9e117eaee5029cd7d8df63964` 静态复核；没有在当前 SHA 上执行 GPU benchmark。性能与默认值问题必须回答“目标版本 / 模型 / 硬件 / workload 下测得”，不能背固定倍数。
>
> **学完能：**
>
> 1. 每题先给 30 秒结论，再用 3 分钟讲机制与源码入口
> 2. 主动补充生产取舍、验证信号、失败边界与回滚条件
> 3. 面对追问时区分稳定契约、当前实现与待测假设

## 回答协议

每题都按五格回答：

- **30 秒结论**：先回答“是什么 / 为什么”。
- **3 分钟机制与源码**：讲数据流、状态与源码入口，不背行号。
- **生产取舍**：说明收益、成本和适用条件。
- **验证、失败与回滚**：给证据；什么现象会否决方案；怎么安全退回。
- **追问**：至少准备两个能继续展开的问题。

面试官真正区分的是：候选人能否把结论限定在证据范围内，而不是能否报出最多的 feature 名。

---

## A. 引擎与请求生命周期

### Q1. vLLM 解决什么问题？

**30 秒结论：** vLLM 是面向 LLM inference / serving 的引擎。它把请求调度、KV cache、模型执行、采样 / pooling、并行与 OpenAI-compatible serving 组合起来，目标是在给定质量与延迟 SLO 下提高 goodput；Paged KV 与迭代级调度是核心机制，但不是性能承诺。

**3 分钟机制与源码：** 用户入口在 `vllm/entrypoints/`，V1 engine loop 在 `vllm/v1/engine/`，调度在 `vllm/v1/core/sched/scheduler.py`，worker / runner 在 `vllm/v1/worker/`。一次请求跨 frontend、EngineCore、worker；每 step 重新形成 batch，执行模型并把结果回传。生成请求走 sampler，pooling runner 走 pooler。

**生产取舍：** vLLM 提供通用能力，具体模型、量化、attention backend 和平台组合仍有支持矩阵。选择引擎要比较质量、SLO、运维与升级成本，不能引用脱离环境的吞吐倍数。

**验证、失败与回滚：** 固定模型 revision、tokenizer、dtype、prompt / output 分布和并发，比较 p50 / p99 TTFT、TPOT、goodput、显存与错误率。若质量或尾延迟不达门禁，回到已验证 image / SHA / config。

**追问：** V1 的进程边界是什么？为什么 tokens/s 不是唯一选型指标？

### Q2. 一个生成请求如何走完全链路？

**30 秒结论：** frontend 校验、渲染并 tokenize，请求进入 EngineCore waiting；Scheduler 分配 token budget 与 KV block；worker forward + sample；EngineCore 更新状态，直到 stop / EOS / length / abort，再释放资源并流式返回。

**3 分钟机制与源码：** 入口请求转换位于 `vllm/entrypoints/openai/`；`vllm/v1/request.py` 保存请求状态；`Scheduler.schedule()` 产出 `SchedulerOutput`；GPU runner 维护 persistent batch，执行 model 与 sampler；engine core 根据输出更新 token、finish reason 和队列。取消、超时与客户端断开也必须传播到资源释放路径。

**生产取舍：** frontend CPU、engine queue、GPU step、网络流式任一段都可能决定尾延迟；只看 GPU utilization 会误诊。

**验证、失败与回滚：** 用 request ID 关联 gateway、frontend、engine 与 worker 证据，分解 queue、tokenization、prefill、decode、serialization。失败注入客户端取消与 worker 异常，确认请求不泄漏 KV block。

**追问：** streaming 首字节与 TTFT 有何区别？abort 如何传到 Scheduler？

### Q3. Continuous batching 与 static batching 有什么差别？

**30 秒结论：** static batch 等一组序列共同完成；continuous batching 在迭代边界重新调度，完成请求退出、等待请求进入，因此能处理长度异质性并提高资源利用率。

**3 分钟机制与源码：** `vllm/v1/core/sched/scheduler.py` 每 step 在 token budget、KV capacity、优先级与模型约束下选择请求；runner 的 persistent batch 复用相邻 step 中仍运行的请求。chunked prefill 还允许长 prompt 分段，与 decode 请求共享 step。

**生产取舍：** 更大 batch 可能提高吞吐，却放大单 step 时间与 TPOT；混合长 prefill 还会影响公平性。连续调度不是“GPU 永远 100%”。

**验证、失败与回滚：** 扫描并发和长度混合，记录 batch tokens、queue time、TTFT / TPOT p99 与 preemption。若 p99 越线，回退 token budget或拆分 workload pool。

**追问：** 为什么 iteration-level 不等于任意 token 可随时插入 kernel？persistent batch 优化依赖什么流量特征？

### Q4. V1 的 Scheduler 每步决定什么？

**30 秒结论：** 它决定哪些请求本步计算多少 token、需要哪些 KV block、哪些 encoder / structured-output / speculative 工作可调度，以及谁需要被推迟或抢占。

**3 分钟机制与源码：** 主入口是 `vllm/v1/core/sched/scheduler.py`。约束来自 `max_num_batched_tokens`、`max_num_seqs`、KV cache manager、encoder budget、请求状态与调度策略；输出包含新 / cached 请求、本步 token 数、block IDs、LoRA / structured-output 等元数据。

**生产取舍：** 调度策略在 throughput、fairness、priority 与尾延迟之间取舍。配置值是容量上限，不是 SLO 保证。

**验证、失败与回滚：** 构造短 decode、长 prefill、priority 与 KV 压力混合流量，检查是否饥饿、是否频繁 preempt。策略改变先 canary；发现低优先级饿死或高优先级无界抢占时回滚。

**追问：** token budget 与 KV capacity 为什么是两种约束？waiting 队首不可调度时是否应该跳过？

---

## B. KV cache、分页与前缀复用

### Q5. Paged KV cache 解决什么？

**30 秒结论：** 它把请求的逻辑 token 位置映射到固定大小的物理 KV block，让物理块无需连续，并支持按需分配、释放和共享前缀；主要解决连续大块预留造成的碎片与刚性容量问题。

**3 分钟机制与源码：** `vllm/v1/core/block_pool.py` 管物理 block，`kv_cache_manager.py` 管请求映射，attention backend 读取 block table 间接寻址。最后一个 block 仍可能有内部碎片；block metadata、table 和 kernel 也有成本。

**生产取舍：** block 小降低尾块浪费，却增加 metadata / table 长度；block 大相反。默认值与可用集合受平台 / backend 约束，不能把某个数字当普适最优。

**验证、失败与回滚：** 扫描真实长度分布与 block size，比较可容纳 tokens、尾块浪费、kernel latency 与 p99；启动校验失败或性能退化就恢复已验证值。

**追问：** 逻辑 block 与物理 block 如何解耦？为什么分页仍有内部碎片？

### Q6. KV cache 每 token 多少字节？

**30 秒结论：** 对常见 decoder attention，一个 token 的近似 KV 字节为 `2 × L × H_kv × D_head × bytes(dtype)`；TP 后每 rank 通常只持有本 rank 的 KV heads，但 MLA、跨层共享、混合 cache spec 会改变公式。

**3 分钟机制与源码：** `2` 是 K 与 V；`L` 为缓存层数；`H_kv` 是 KV head 数而非 query head 数。实际 cache spec 位于 `vllm/v1/kv_cache_interface.py`，启动 profile 与 cache config 决定物理页面数。GQA / MQA 通过减少 KV heads 降低容量。

**生产取舍：** FP8 KV 可能省显存，但需要 backend / scale / 精度验证；增加 context 或并发是同一 KV 预算的竞争。

**验证、失败与回滚：** 用模型 config 算理论 bytes/token，再用启动日志的 page size / page count 与 GPU 增量校验；误差大时检查 TP 分片、hybrid layer、padding 与 allocator reserve。

**追问：** 为什么 hidden size 不能直接代替 KV heads 计算？PP 会怎样改变每 rank KV？

### Q7. vLLM 如何决定 KV cache 容量？

**30 秒结论：** 启动时先加载模型并 profile 非 KV 峰值，再在配置允许的 GPU memory budget 中给 KV cache 分配页面；显式 KV 字节配置、offload 与不同平台会改变路径。

**3 分钟机制与源码：** 配置在 `vllm/config/cache.py`，worker 的 memory profiling 与 `vllm/v1/kv_cache_interface.py` 共同生成 cache config。不能简单写成“总显存乘 utilization 减权重”，因为 graph、activation、非 Torch 内存、并行 rank 与实现 reserve 都参与。

**生产取舍：** 把 utilization 推高可增加 KV，却缩小瞬时峰值余量；跳过 profile / 固定 cache 便于复现，但操作者承担 OOM 风险。

**验证、失败与回滚：** 保存每 rank 启动 profile、page count、空闲余量，跑最大允许输入与混合流量 soak。出现启动失败或运行 OOM，先恢复旧预算并降低 admission，而不是盲目重试。

**追问：** 为什么 profile 通过仍可能运行 OOM？CUDA Graph capture 对预算有什么影响？

### Q8. Prefix caching 的 hash 为什么要链式？

**30 秒结论：** 当前 block hash 包含前一 block hash、当前完整 block token 与 extra keys；链式结构让相同 token block 在不同前缀下产生不同 key，避免把语义不同的 KV 错误复用。

**3 分钟机制与源码：** 入口在 `vllm/v1/core/kv_cache_utils.py`。extra keys 覆盖多模态 identifier / offset、LoRA identity 与 cache salt 等影响 KV 的上下文。只缓存完整 block，匹配的是最长可复用前缀，不是任意子串。

**生产取舍：** 更强隔离减少错误命中，但 salt / 路由不一致会降低 hit rate；命中率高也不等于节省了足够 GPU 时间。

**验证、失败与回滚：** 用同前缀、单 token 差异、不同 LoRA / 图片 / salt 做正反例；同时看 hit tokens、TTFT 与输出一致性。疑似错误命中时立即禁用 prefix caching 并隔离版本。

**追问：** 为什么只 hash token IDs 不够？tokenizer / chat template 升级如何影响 cache？

### Q9. ref count、free queue 与驱逐是什么关系？

**30 秒结论：** 被请求引用的 block 不能作为新分配复用；引用归零后进入可回收集合，但若仍保留 hash metadata，未来可在被真正复用前命中 prefix cache。

**3 分钟机制与源码：** `vllm/v1/core/block_pool.py` 维护 block、free list 与 hash map；cache manager 在请求完成、abort 或 preemption 时释放引用。具体共享 / copy-on-write 能力必须以当前 V1 请求路径为准，不能把旧 beam-search 实现直接套来。

**生产取舍：** 延迟驱逐提高命中机会，却占用可观察的 cache working set；高 churn 会让 hash 命中与实际保留时间下降。

**验证、失败与回滚：** 反复请求共享前缀并注入 cache pressure，观察命中、驱逐与 block 使用是否守恒；abort 后 block 不归还属于资源泄漏故障。

**追问：** ref count 为零为何不等于立刻清空数据？并发命中如何避免驱逐竞态？

---

## C. 调度、抢占与输出

### Q10. KV 不够时为什么会 preempt？

**30 秒结论：** Scheduler 若无法为本步请求分配足够 KV，就必须释放部分运行请求的 block，让系统继续前进；被抢占请求未来重算或按实现支持的方式恢复。

**3 分钟机制与源码：** 抢占逻辑在 `vllm/v1/core/sched/scheduler.py` 与 cache manager 交界。当前策略、优先级与重算细节要读锁定 SHA，不能假设所有平台都有 host swap。preemption 会增加重复 prefill、TTFT 与 goodput 损失。

**生产取舍：** 提高 `max_num_seqs` 可增加表面并发，却可能造成 thrash；降低并发、缩短上下文、增加 KV 或扩副本更稳。

**验证、失败与回滚：** 监测当前版本注册的 preemption counter、KV usage、waiting / running 与 TTFT。持续抢占时先 admission / 并发回滚，不能只扩 retry。

**追问：** 为什么 preemption counter 增长可能导致 retry storm？priority preemption 如何避免低优先级饿死？

### Q11. Chunked prefill 解决什么？

**30 秒结论：** 它把长 prompt 的 prefill 拆成多 step，在 token budget 内与其他请求交错，降低单个长 prompt 独占一步造成的 decode 抖动。

**3 分钟机制与源码：** Scheduler 给请求安排本步 `num_scheduled_tokens`；attention 读取已有 KV 与当前 chunk。相关配置在 `vllm/config/scheduler.py`，多模态还有 `disable_chunked_mm_input` 等边界。

**生产取舍：** chunk 小改善公平性，却增加 step / launch / 调度次数；chunk 大提高效率但可能抬高 TPOT p99。

**验证、失败与回滚：** 用双峰 prompt 长度同时测 TTFT、TPOT 与吞吐，逐步扫描 token budget；出现质量差异或特定模型 / 多模态不支持时回到默认并隔离长请求池。

**追问：** chunked prefill 与 prefix cache 如何相互作用？为什么长上下文 attention 仍然昂贵？

### Q12. seed 能保证跨部署 bitwise deterministic 吗？

**30 秒结论：** 不能。seed 控制 per-request generator 的随机序列，但 backend、batch 组成、并行规约、浮点精度、driver / kernel 与模型 revision 都可能改变 logits 或采样路径。

**3 分钟机制与源码：** `vllm/sampling_params.py` 区分 greedy、random 与 random-seeded；`vllm/v1/sample/` 选择 sampling backend。带 generator 还可能让 top-k/top-p 走不同 fallback。greedy 也可能在近似相等 logits 上受数值差异影响。

**生产取舍：** 强复现通常牺牲 batching、backend 自由度或性能。生产更应定义输出 / quality 容差，而非承诺跨硬件逐 token 相同。

**验证、失败与回滚：** 固定完整环境与流量顺序重复运行，分别测同进程、重启、跨卡；把确定性等级写进 API 契约。达不到时撤销 bitwise 承诺。

**追问：** temperature=0 是否足够？为什么 per-request generator 可能改变 backend？

### Q13. Sampling 的处理顺序为什么重要？

**30 秒结论：** raw logprobs、allowed / bad words、自定义 processors、penalties、temperature、min-p、top-k/top-p 与 gather 的先后会改变分布和 API 语义，不能随意交换。

**3 分钟机制与源码：** 当前顺序在 `vllm/v1/sample/sampler.py`：先按 mode 保存 raw 值并转 FP32，再应用约束、非 argmax-invariant processor 与 penalties；随机路径应用 temperature、argmax-invariant processor、min-p / top-k / top-p，最后 gather。具体 backend 见 `ops/topk_topp_sampler.py`。

**生产取舍：** logprobs 与异质 sampling 参数增加 full-vocab 计算、返回体与 batch 分支；custom processor 还有限制。

**验证、失败与回滚：** 用可手算小 vocab 单测顺序，检查 sampled token、rank 和 raw / processed logprobs；升级后契约变化则版本化响应或回滚。

**追问：** raw 与 processed logprobs 有何区别？为什么 top-N 返回仍可能需要 full-vocab softmax？

---

## D. Attention、runner 与编译

### Q14. Attention backend 如何选择？

**30 秒结论：** 由平台、attention type、dtype、head size、cache layout、模型功能与显式配置共同决定；不是“所有 NVIDIA 都固定某一个 backend”。

**3 分钟机制与源码：** registry 位于 `vllm/v1/attention/backends/registry.py`，backend 实现位于同目录及 MLA 子树。启动日志与 config 是实际选择证据；某 backend 支持 decode 不代表支持 encoder-only、sliding window、quantized KV 或所有 graph 模式。

**生产取舍：** 最快 backend 可能限制模型功能或硬件；显式强制提高可复现性，却可能在升级后启动失败。

**验证、失败与回滚：** 保存 backend 名、版本与启动理由，跑 correctness + shape / length matrix 与 profile。出现 unsupported 或数值回归时回到 auto / 已验证 backend。

**追问：** backend 与 attention layer 有什么区别？为什么同模型换 KV dtype 可能换路径？

### Q15. Paged attention kernel 做了什么？

**30 秒结论：** 它通过 block table 找到请求逻辑 KV 位置对应的物理页面，在不要求物理连续的前提下完成 QK、softmax 与 V 聚合。

**3 分钟机制与源码：** vLLM 自有 kernel 在 `csrc/attention/`，平台 backend 也可提供 paged 实现。kernel 要处理不同 sequence length、head mapping、partition / reduce、cache dtype 与 block table。现代部署实际走哪条路径由 backend selection 决定。

**生产取舍：** 间接寻址换取内存弹性；长 context、page table locality 与 split 策略决定效率。不能用“分页更省显存”推导“总是更快”。

**验证、失败与回滚：** 用 contiguous reference 做数值对比，扫描 context / batch / head / block size；性能或正确性异常时切已验证 backend，而非现场改 kernel。

**追问：** 为什么 decode 常是 memory-bound？block table 何时会成为负担？

### Q16. GQA / MQA 为什么能省 KV？

**30 秒结论：** 多个 query heads 共享更少的 KV heads，KV cache 按 `num_kv_heads` 存，因此容量按共享比例下降；计算时 backend 把 query head 映射到对应 KV head。

**3 分钟机制与源码：** 模型 config 给出 query 与 KV head 数，attention layer 与 backend 处理映射。TP 后 KV heads 的分片还受总 head 数与复制规则影响，容量应按每 rank 实际配置算。

**生产取舍：** 这是模型架构属性，不是 serving flag；换 GQA 模型涉及质量与 checkpoint，而非无损开关。

**验证、失败与回滚：** 对照 config、每 rank cache spec 与 bytes/token；发现 KV head 复制时修正容量模型。

**追问：** MQA 是 GQA 的什么特例？TP size 大于 KV heads 时怎么办？

### Q17. CUDA Graph 与 torch.compile 分别优化什么？

**30 秒结论：** CUDA Graph 主要减少重复 kernel launch / CPU 提交开销；`torch.compile` 捕获并优化可编译图、融合或重排算子。两者可组合，但有动态 shape、自定义 op、内存地址与 warmup 成本。

**3 分钟机制与源码：** 配置在 `vllm/config/compilation.py`，V1 graph dispatch 与 capture 位于 worker / compilation 相关模块。graph mode、capture sizes 与 fallback 都是版本相关实现。

**生产取舍：** 更多 capture size 可能改善命中，却增加启动时间和显存；强动态图功能可能回 eager。compile cache 还影响冷启动与镜像可移植性。

**验证、失败与回滚：** 分冷 / 热启动记录 compile、capture、graph replay 命中、显存和 p99；失败时用已验证 compilation config 或 eager 做诊断，不把 eager 当默认性能结论。

**追问：** 为什么输入地址稳定很重要？动态 LoRA / custom op 如何影响 graph？

---

## E. 分布式与模型架构

### Q18. Tensor Parallel 如何切分与通信？

**30 秒结论：** TP 把层内权重 / heads / hidden 维分到多个 rank，局部 matmul 后用 collective 合并。它让单模型跨卡放置，也引入每层通信与同步。

**3 分钟机制与源码：** 核心状态与 process groups 在 `vllm/distributed/parallel_state.py`；column-parallel 与 row-parallel layers 位于 `vllm/model_executor/layers/`。具体 collective 次数受 fused layer、sequence parallel、MoE 与模型结构影响，不能固定说“每层恰好两次”。

**生产取舍：** TP 降低每 rank 权重 / KV，可能降低单请求 latency；跨慢链路扩 TP 会让通信吞噬收益。优先在高速互联域内 TP。

**验证、失败与回滚：** 比较 TP=1/2/4 的 tokens/s、TTFT、collective trace、link bandwidth 与每 rank 显存；跨节点性能倒退则改为较小 TP + replica DP。

**追问：** row / column parallel 为什么配对？TP 与 DP 的扩容目标有什么不同？

### Q19. Pipeline Parallel 的收益和 bubble 是什么？

**30 秒结论：** PP 把层段放到不同 stage，边界传 activation，主要用于模型放置或降低层内 collective 范围；stage 不均衡与 microbatch 不足会产生 bubble。

**3 分钟机制与源码：** PP rank / group 仍由 `parallel_state.py` 管理，模型实现按 layer range 装载。推理只有 forward，但请求 / microbatch 调度、首尾 stage 输出与不均衡仍需处理。

**生产取舍：** PP 边界通信通常小于大 TP collective，却增加串行 stage latency 与调度复杂度。TP×PP 必须满足 world size。

**验证、失败与回滚：** 测每 stage 时间、bubble、边界通信和 OOM margin；发现一个 stage 长期成为瓶颈时重切层或退回 TP。

**追问：** 为什么层数均分不一定负载均衡？PP 对 TTFT 与 throughput 的影响为何不同？

### Q20. Expert Parallel 与 EPLB 解决什么？

**30 秒结论：** EP 把 MoE experts 分布到 ranks，token 根据 router 结果跨 rank dispatch；EPLB 通过统计负载并重排 / 复制 experts，减少热点导致的 straggler。

**3 分钟机制与源码：** EP / EPLB 位于 `vllm/distributed/` 和 `vllm/distributed/eplb/`；模型 fused MoE layer 与 collective backend共同决定数据流。世界大小、expert 数、冗余与 top-k 必须一致。

**生产取舍：** EP 降低每 rank expert 权重，却引入 all-to-all 与负载不均；EPLB 有观测窗口和重平衡成本。

**验证、失败与回滚：** 记录每 expert token、rank busy time、all-to-all 和 p99；重平衡导致抖动或收益不足时关闭 EPLB / 恢复固定映射。

**追问：** 为什么平均 token 数相同仍可能有 straggler？TP 与 EP 可以怎样组合？

### Q21. TP / PP / DP / EP world size 怎么算？

**30 秒结论：** 先画 process-group 维度，常见总进程数近似 `TP × PP × DP`；EP 通常在既有 ranks 上定义 expert group，是否形成额外乘数取决于部署模式，不能盲目把四者相乘。

**3 分钟机制与源码：** 看 `parallel_state.py` 创建的 group 与启动参数，再核对每 rank coordinates。DP 是模型副本 / engine 并行，TP / PP 是单副本内部；EP 复用或重组 ranks。

**生产取舍：** world size 正确只说明拓扑可构造，不说明链路、NUMA、容错域或性能合理。

**验证、失败与回滚：** 启动时导出 rank-to-host/GPU/group 表，检查每张卡唯一归属与 group size；NCCL hang 前先验证拓扑和环境一致性。

**追问：** 两节点各 8 卡、TP=4、PP=2 时能有几个 DP replica？EP=8 应如何解释？

### Q22. Disaggregated prefill / decode 何时值得？

**30 秒结论：** 它把偏计算的 prefill 与偏 KV 读取的 decode 放到独立 worker pool，并通过 connector 转移 KV；只有资源独立优化收益超过 KV 传输、排队和故障复杂度时才值得。

**3 分钟机制与源码：** connector 接口与传输状态在 V1 KV connector 相关模块；production stack / 外部系统还负责路由与拓扑。必须把 transfer bytes、带宽、首包等待、backpressure 与版本一致性纳入状态机。

**生产取舍：** 可独立扩缩 P/D、隔离干扰；代价是网络、额外副本、跨池排队和更大故障域。

**验证、失败与回滚：** 比较聚合与拆分的端到端 TTFT / TPOT / goodput，注入 connector timeout / partial failure。无法安全 fallback 时不应直接上线。

**追问：** KV transfer 时间怎样估算？prefill / decode replica 比例如何动态调整？

---

## F. 高级能力与质量

### Q23. 量化怎么选？

**30 秒结论：** 先确认目标模型、checkpoint format、GPU architecture 与 vLLM backend 支持，再在质量门禁下比较权重、activation 和 KV dtype；不存在跨模型通用的“最佳 INT4 / FP8”。

**3 分钟机制与源码：** quantization registry 与实现位于 `vllm/model_executor/layers/quantization/`；load format、linear / MoE layer 和 kernel共同决定是否支持。量化节省的理论字节还要扣除 scale、zero point、padding 与未量化层。

**生产取舍：** 更低位宽省显存 / 带宽，却可能降低质量、增加 dequant 或限制 backend；KV 量化还影响长上下文。

**验证、失败与回滚：** 用 golden set、perplexity / task metric 与服务 SLO对照 BF16 / FP16 baseline；不兼容应启动失败，质量越线则回滚不可变 model artifact。

**追问：** weight-only 与 W8A8 的瓶颈不同在哪里？为什么“模型能加载”不足以验收？

### Q24. Speculative decoding 为什么可能加速？

**30 秒结论：** draft 一次提出多个 token，target 在一次或较少 forward 中验证；接受率高且 draft 成本低时，每次 target step 产出更多被接受 token，同时 rejection sampling 保持 target 分布。

**3 分钟机制与源码：** V1 实现在 `vllm/v1/spec_decode/` 与 sampler rejection path。收益由 proposal length、接受率、target verification、draft overhead、batch 与 sampling 参数决定；某些参数组合会被拒绝。

**生产取舍：** 低接受率会增加工作反而变慢；额外模型 / weights / state 增加显存与运维。

**验证、失败与回滚：** 记录 accepted tokens、draft / target time、端到端 output tokens/s 与质量等价性；接受率或 goodput 低于门禁即关闭 speculative config。

**追问：** 为什么接受率不是唯一指标？greedy 与 stochastic rejection 有何区别？

### Q25. Structured output 能保证什么，不能保证什么？

**30 秒结论：** 它在采样时 mask 当前 grammar 状态不允许的 token，强化语法约束；不保证字段语义、业务授权、工具安全或所有 backend 对完整 JSON Schema 等价支持。

**3 分钟机制与源码：** `vllm/v1/structured_output/` 编译 request grammar、生成 bitmask、推进 / rollback 状态。默认 backend 是 `auto`，当前有版本相关 fallback；显式 backend 验证失败不会静默换后端。

**生产取舍：** `auto` 易用但升级可能换 backend；显式配置可复现但要维护兼容矩阵。复杂 schema 增加 CPU compile 与 TTFT。

**验证、失败与回滚：** 做 backend × schema × tokenizer 正反例，保留原始 token 与 compile error；语义仍由服务端校验。升级异常可固定旧 backend / 旧 image。

**追问：** reasoning parser 何时应用 grammar？为什么结构合法仍不能直接执行工具？

### Q26. Multi-LoRA serving 的 CPU cache 与 GPU slot 是什么？

**30 秒结论：** adapter 先注册在容量受限的 CPU cache，活跃 adapter 写入 `max_loras` GPU slots；Punica mapping 让 batch token 指向不同 slot。`max_cpu_loras` 与 `max_loras` 是不同容量。

**3 分钟机制与源码：** `vllm/lora/model_manager.py`、`worker_manager.py` 与 `punica_wrapper/` 负责加载、激活、LRU 与增量 matmul。静态 `--lora-modules` 与运行时加载控制面要分开；后者源码只定位为本地开发能力。

**生产取舍：** 更多 slot 减少换入，却挤压 KV；更多 CPU cache 降低存储读取，却占 host memory。路径 / resolver 是安全边界。

**验证、失败与回滚：** 扫描热度与 rank，记录注册 / 激活 / 淘汰、TTFT、显存和版本 hash；不兼容或 thrash 时路由到静态版本化 pool。

**追问：** 单 batch 超过 `max_loras` 与时间上的 LRU thrash 有何不同？为什么 prefix hash 要包含 LoRA identity？

---

## G. 生产诊断与变更

### Q27. 启动慢如何分解？

**30 秒结论：** 把下载 / 权重读取、反序列化、模型初始化、memory profile、compile、CUDA Graph capture、distributed rendezvous 与 readiness 分开计时；“启动慢”不是一个根因。

**3 分钟机制与源码：** loader、worker 初始化、compilation config 和 graph capture 分属不同模块。多 rank 还要看最慢 rank与 shared filesystem。模型 cache 命中、compile cache 与 driver 都影响冷启动。

**生产取舍：** eager / 减少 capture 可缩短冷启动，却可能损失稳态性能；预烘焙 artifact 提高速度但增加供应链与兼容管理。

**验证、失败与回滚：** 保存阶段日志、每 rank 时间与 readiness；在同节点区分 cold / warm。启动优化不得跳过 correctness / health gate，失败回旧 image。

**追问：** 为什么 readiness 不能只看进程存在？compile cache 能否跨 GPU architecture 复用？

### Q28. 如何做公平 benchmark？

**30 秒结论：** 先定义 workload 和 SLO，再固定模型 / tokenizer / quality / sampling，分别报告请求与 token 长度分布、并发、p50 / p99、goodput、错误率和资源；不只报峰值 tokens/s。

**3 分钟机制与源码：** vLLM CLI benchmark 工具和 `07-hands-on/` 给出流程。服务 benchmark 要区分 input / output tokens、TTFT、TPOT / ITL、open / closed loop、warmup 与超时；比较不同引擎还要确保同一渲染和 stop 行为。

**生产取舍：** 峰值吞吐可能靠违反 p99 SLO 获得；synthetic 数据可复现但不能代表缓存与长度相关性。

**验证、失败与回滚：** 保存 exact command、dataset hash、environment 与 raw results；结果不可复现或 quality 不等价就作废，不进入容量承诺。

**追问：** open-loop 与 closed-loop 有何偏差？goodput 怎样把 SLO 纳入吞吐？

### Q29. TPOT p99 突升怎么排查？

**30 秒结论：** 先确认指标与时间窗，再按 workload 变化、batch / step、KV pressure / preemption、backend / compile fallback、collective / hardware、CPU / 网络分层定位；不要先调参数。

**3 分钟机制与源码：** 对照当前版本 `/metrics` inventory，关联 waiting / running、KV usage、preemption、input / output 长度、request latency buckets、GPU / link 与 engine logs。TTFT 与 TPOT 要分开；平均 GPU util 不能证明 kernel 健康。

**生产取舍：** 降 token budget / 并发通常能缓解尾延迟却损失吞吐；扩副本可能因冷启动与路由失衡短期恶化。

**验证、失败与回滚：** 先做只读取证和单变量 canary；缓解后要用同流量重放验证。若证据不足，不执行集群级 mutation；使用 [`vllm-doctor`](../08-production-deployment/09-vllm-doctor-skill.md) 的 fail-closed 流程。

**追问：** preemption 为零还能是什么？为什么 retry 会放大排队？

### Q30. 如何升级 vLLM 而不把“能启动”当成功？

**30 秒结论：** 固定 vLLM SHA、模型 / tokenizer revision、quantization、GPU / driver / CUDA / PyTorch、backend 与 API config，先做 source impact、golden correctness、性能与 failure gate，再 shadow / canary / drain；回滚单位是不可变兼容矩阵。

**3 分钟机制与源码：** 文档侧用 `tools.source_sync` 验证源码锚点；服务侧按 [`12-upgrades-rollbacks-and-compatibility.md`](../08-production-deployment/12-upgrades-rollbacks-and-compatibility.md) 建 change record。缓存、LoRA、chat template、metrics 与 client behavior 都可能是兼容面。

**生产取舍：** 快速跟 main 获得功能，也扩大回归面；长期冻结降低变更风险，却积累安全与兼容债务。

**验证、失败与回滚：** 预先定义 rejection criteria、drain 与 rollback，不在事故中临时决定。任何 quality、API、p99、错误率或恢复演练越线立即停止 rollout。

**追问：** 为什么旧 Pod 的 cache 不应直接给新版本复用？哪些 change 可只重跑局部矩阵？

---

## 口述实验与失败证据

随机抽 5 题，每题录音 4 分钟：前 30 秒不准讲实现细节，后 3 分钟必须包含一个源码入口、一个 production tradeoff、一个验证信号和一个 rollback 条件。复盘时记录：

| 字段 | 合格证据 |
| --- | --- |
| 结论 | 直接回答题目，无绝对性能承诺 |
| 机制 | 状态 / 数据流正确，源码路径存在 |
| 取舍 | 同时说收益与代价，并给适用条件 |
| 验证 | 指标来自当前 inventory，或说明需先核准 |
| 失败 / 回滚 | 有明确否决条件，不用“再观察”代替 |

失败证据包括：说出不存在的 flag / metric / file、把估算称作实测、被追问假设时无法给单位、只讲 mitigation 不讲验证与回滚。把失败题映射回相应技术章节，而不是继续背答案。

> **硬件验证状态：** 当前 SHA 未执行 GPU 口述题配套实验；涉及性能的问题只给测量方法，不给硬编码结果。

## 小结

- 30 秒回答负责结论，3 分钟回答负责机制与证据。
- 稳定契约、当前实现、环境默认值与待测假设必须分开。
- 每个优化答案都要带 quality / SLO gate，每个生产答案都要带 failure / rollback。
- 最强的回答不是数字最多，而是能说明数字来自哪里、在什么条件下失效。

## 自检

1. 随机抽 10 题，是否每题都能在 30 秒内直接回答？
2. 是否至少有 8 题能画出跨组件数据流，而非只报文件名？
3. 是否能把任意性能追问改写为“固定哪些变量、测哪些指标、什么条件否决”？
4. 是否能指出本章至少五个不能跨版本承诺的实现细节？

## 下一步

- [`02-system-design.md`](./02-system-design.md)：把单点知识组织成需求优先的完整架构
- [`03-capacity-and-troubleshooting-drills.md`](./03-capacity-and-troubleshooting-drills.md)：练公式、单位与证据树
- [`04-mock-interview-and-rubric.md`](./04-mock-interview-and-rubric.md)：按五轮 rubric 做完整模拟

## Source trail

- `vllm/v1/engine/core.py`、`vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`、`block_pool.py`、`kv_cache_utils.py`
- `vllm/v1/sample/`、`vllm/v1/structured_output/`、`vllm/v1/spec_decode/`
- `vllm/v1/attention/backends/`、`csrc/attention/`
- `vllm/distributed/parallel_state.py`、`vllm/distributed/eplb/`
- `vllm/model_executor/layers/quantization/`、`vllm/lora/`
