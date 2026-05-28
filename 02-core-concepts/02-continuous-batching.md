# 02. Continuous Batching（连续批处理）

> **谁该读这一篇？** 想搞清"为什么 vLLM 比 HF Transformers 吞吐高一个量级"且能在面试时不靠 buzzword 讲清的同学；想自己实现一套 iteration-level 调度器的工程师。
>
> **前置阅读：** [`01-paged-attention.md`](01-paged-attention.md)（PagedAttention 让 KV 不必连续，是 continuous batching 能跑起来的物理前提）；[`01-overview/00-prerequisites.md`](../01-overview/00-prerequisites.md) §8 batching。
>
> **耗时：** 约 14 分钟。
>
> **学完能：**
> 1. 在白板画出 No / Static / Continuous 三种 batching 的时间线，并说出 GPU 利用率差异。
> 2. 用"token budget" 概念解释 `max_num_batched_tokens` 如何同时影响吞吐和延迟。
> 3. 区分 vLLM 的 token-level 调度与 TGI 的 sequence-level 调度。
> 4. 解释 recompute vs swap 抢占策略的取舍，以及 V1 默认 recompute 的理由。

这是 vLLM 性能上的第二个杀器，与 PagedAttention 平级重要。它解决"GPU 在等慢请求"的问题。

---

## 1. 三种 batching 直观对比

### 1.1 No Batching（朴素逐请求）
```
GPU: [A.....][B.....][C.....][D.....]
请求一个接一个跑。GPU 长期空转。
```

### 1.2 Static Batching（HF Transformers 默认）
```
T=0    T=1     T=2          ...     T=10
[ABCD][ABCD][AB-D] ← C 完了，但槽位还占着  ...   [---D]
                                                   ↑
                                          只剩 D 在跑，浪费 75%
```
所有请求**同时进、同时出**。最慢的那个决定整组的延迟。

### 1.3 Continuous / Iteration-level Batching（vLLM）
```
T=0  T=1  T=2  T=3  T=4  T=5  T=6 ...
[ABCD][ABCD][ABED][AFED][AFED][AFGD][AHGD]
            ↑     ↑          ↑          ↑
         C 出 E 进  B 出 F 进  E 出 G 进  D 出 H 进
```
**每一步**（每生成一个 token）都检查：

- 谁完成了 → 立刻退出，释放 KV
- 等待队列有谁 → 立刻加入

GPU 几乎从不空转。

---

## 2. 为什么"每一步都调度"是可行的？

直觉上你可能觉得：每一步都重新组 batch、重建 tensor，开销不大吗？

vLLM 的回答是：

1. **PagedAttention 让 KV 不必连续**。请求加入/退出 batch 不需要搬数据。
2. **InputBatch 持久化**（V1 优化）：每步只更新 diff，不重建。
3. **Token-level**而非 sequence-level 调度：一次调度的单位是"算多少 token"，不是"跑哪些请求到底"，所以新请求中途插入完全可以。

---

## 3. Token Budget 是核心

V1 调度器的关键参数：`max_num_batched_tokens`（默认 8192 或类似）。

每一步，所有请求**加起来**最多算这么多 token：

```
running 请求 A (decode): 1 token
running 请求 B (decode): 1 token
running 请求 C (prefill, prompt=2048): 2048 token
waiting 请求 D (prefill, prompt=512):  512 token
                                      ────
                                      2562 token  < 8192 ✓
```

如果超了 budget：

- 长 prefill 会被 **chunk**（拆成多步跑，详见 `05-chunked-prefill.md`）
- 新请求暂时不让进

这样设计的好处：**每一步的工作量上限可预测**，单 step 延迟稳定。

---

## 4. Static vs Continuous 数学直觉

设有 N 个请求，长度服从某分布。Static batching：

```
total_time = N × max(L_i)   # 所有请求等最长的那个
```

Continuous batching（忽略调度开销）：

```
total_time = sum(L_i) / batch_capacity   # 紧凑利用
```

当长度方差大（实际场景几乎总是），continuous 的加速比可以达到 5-24×。

---

## 5. 调度优先级与公平性

Continuous batching 不是"先来后到"，vLLM 支持多种调度策略：

| 策略       | 行为                       | 适用场景         |
| -------- | ------------------------ | ------------ |
| FCFS（默认）  | 先到先服务                    | 通用、公平         |
| Priority | 高优先级请求优先调度，可抢占低优先级       | 多租户、SLO 服务  |

实际看 `vllm/v1/core/sched/scheduler.py` 里的 `_schedule_running` 和 `_schedule_waiting` 函数。

---

## 6. 完成检测：怎么知道一个请求该退出？

在每一步的 forward 后，Scheduler 检查每个请求：

1. **采样到了 EOS token** → FINISHED
2. **达到 max_tokens** → FINISHED
3. **stop_token / stop_string 匹配** → FINISHED

完成的请求：

- 从 running 队列移除
- 它的所有 KV block ref_cnt -= 1
- 通知 API server 流式返回（带 finish_reason）

代码：`vllm/v1/engine/llm_engine.py` 的输出处理 + `Scheduler.update_from_output()`。

---

## 7. 抢占（Preemption）：当 KV 不够时

continuous batching 的隐藏假设是"KV cache 总是够用"。但请求数多 + 序列长时会爆。

vLLM 的策略：

### 7.1 Recompute（默认，推荐）
- 直接 free 低优先级请求的所有 block
- 把它的 status 设回 WAITING
- 它的已生成 token 保留在 CPU 端
- 重新调度时**重新 prefill**（KV 重新算）

代价：算力浪费
好处：实现简单、不占额外显存

### 7.2 Swap（CPU 卸载）
- 把 block 从 GPU 拷到 CPU 内存
- 恢复时拷回来

代价：PCIe 拷贝慢、占 CPU 内存
好处：算力不浪费

V1 默认 recompute，因为：

1. PCIe 反而经常比重算还慢
2. 代码更简单
3. 配合 prefix caching，重算时前面的部分还能命中 cache

代码：`Scheduler._preempt()`。

---

## 8. 代码定位

| 行为                | 文件 : 函数                                                       |
| ----------------- | ------------------------------------------------------------- |
| 主调度入口             | `vllm/v1/core/sched/scheduler.py : Scheduler.schedule()`       |
| 运行中请求继续           | `Scheduler._schedule_running()`                                |
| 等待中请求加入           | `Scheduler._schedule_waiting()`                                |
| 抢占                | `Scheduler._preempt()` 或 `_preempt_lowest_priority_request()` |
| 输出处理              | `Scheduler.update_from_output()`                               |
| 输入 batch 增量更新      | `vllm/v1/worker/gpu_input_batch.py`                            |

---

## 9. 面试常见追问

**Q: continuous batching 一定比 static 快吗？**
A: 不一定。如果所有请求长度高度一致（比如 benchmark 的固定长度），两者性能接近。但生产场景长度方差大，continuous 必赢。

**Q: 调度本身不是有开销吗？为什么不阻塞 GPU？**
A: ①调度是纯 CPU 操作，跟 GPU 前向 overlap（V1 的 AsyncScheduler 显式做这件事）。②InputBatch 增量更新，不重建。③单步调度通常 < 1ms，GPU forward 几十 ms，比例可忽略。

**Q: 怎么平衡 throughput 和 latency？**
A: 关键旋钮是 `max_num_batched_tokens`：

- 大 → 大 batch，throughput 高，但 step 时长长，latency 抖动大
- 小 → 反之
生产场景一般 4096-8192，配合 chunked prefill 限制单步 prefill 量。

**Q: TGI 也声称支持 continuous batching，差别在哪？**
A: TGI 是 sequence-level（请求级），仍然按"prefill 阶段→decode 阶段"。vLLM 是 token-level，prefill 和 decode 混跑，且 chunked prefill 默认开启，更细。

---

## 小结

- Continuous batching = **iteration-level 动态批处理**：每生成 1 个 token 就重新组 batch，完成的请求立刻退出、等待的立刻进入，GPU 几乎不空转。
- 能跑起来的三个前提：① PagedAttention 让 KV 加入/退出不搬数据；② InputBatch 持久化只更新 diff；③ token-level 调度让"算多少 token"成为唯一调度单位。
- **`max_num_batched_tokens` 是核心旋钮**：大 → 吞吐高 / 延迟抖动大；小 → 反之；生产经验值 4096-8192。
- 抢占有两种：**recompute**（释放 KV，重新 prefill）与 **swap**（拷到 CPU）；V1 默认 recompute——因为 PCIe 慢、prefix cache 兜底、实现简单。
- vLLM 的 token-level 与 TGI 的 sequence-level 是本质差异——后者仍按"prefill 阶段 → decode 阶段"分组。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. No / Static / Continuous 三种 batching 的 gantt + GPU 空闲位置。**

```
No batching（每请求独立 forward）：
卡: [A 全程][闲][闲][B 全程][闲]...
空闲：请求间切换 + 单请求 decode 时 GPU 算力闲（每步只算 1 token）

Static batching（一批一起进出）：
卡: [批1: A短B中C长D中 一起跑][批2: 等批1全做完才进]
空闲：A 早做完后槽位被 C 锁住，等到 C 结束才能换下一批
      (典型："长尾尾"现象，GPU 利用率 < 30%)

Continuous batching（每 step 重组）：
卡: [step1: A B C D][step2: A 退出 + E 加入: B C D E][step3: B 退出 + F 加入]...
空闲：几乎没有；瓶颈变成 KV 容量与算力
      (GPU 利用率 80%+)
```

→ **核心区别**：static 的批边界是固定的，continuous 的边界是 token 级。

---

**2. `max_num_batched_tokens=8192`，16 个 decode + 1 个 prompt=12000 的 prefill，怎么排？**

V1 默认混合策略（[`05-chunked-prefill.md`](05-chunked-prefill.md) §5）：

1. 先给 16 个 decode 各 1 token → 占 budget 16
2. 剩余 budget = 8192 - 16 = 8176
3. 给那个 prefill 请求**切第一个 chunk = 8176 token**（不是 12000）
4. 一步内 forward 总 token = 16 + 8176 = 8192，正好打满

**接下来的 step**：

- step 2：16 个 decode（已经又生成 1 token）+ prefill 剩余 12000-8176=3824 token
- 16 decode + 3824 prefill chunk = 3840 token，仍在 8192 之内 → 这一步把 prefill 跑完
- prefill 完了立刻 decode 第一个新 token

**两个 step 跑完 12000 prefill + 32 个 decode token**，每步时长接近一致（约 8000 token forward），TPOT 平稳。如果不切，那 12000 token 一次 forward 会让其他 16 个用户等几百 ms。

---

**3. token budget 在哪几行被扣？**

源码：`vllm/v1/core/sched/scheduler.py`（行号会随版本变）

```python
def schedule(self):
    token_budget = self.scheduler_config.max_num_batched_tokens

    # ① 先调度 running（已在 batch 内的请求，多数 decode 1 token）
    self._schedule_running(token_budget)
    # 内部：for req in self.running:
    #         num_tokens = min(req.remaining_tokens, token_budget)
    #         token_budget -= num_tokens                ← 扣减 #1
    #         num_scheduled_tokens[req.id] = num_tokens

    # ② 再调度 waiting（新请求 / preempted）
    self._schedule_waiting(token_budget)
    # 内部：while waiting and token_budget > 0:
    #         req = waiting.pop_request()
    #         num_blocks = ...
    #         new_blocks = kv_cache_manager.allocate_slots(req, ...)
    #         if new_blocks is None:
    #             # KV 不够，触发 preempt（见下面 _preempt_request）
    #         num_tokens = min(req.num_prompt_tokens, token_budget)
    #         token_budget -= num_tokens                ← 扣减 #2

    return SchedulerOutput(num_scheduled_tokens=...)
```

定位方法：grep `token_budget -=` 或 `num_scheduled_tokens[`。两个扣减点分别对应 decode-first 和 prefill 部分。

---

**4. Recompute 比 swap 快的原因 + swap 反而更优的场景。**

**Recompute 快的原因**：

1. **PCIe 太慢**：H100 PCIe Gen5 ×16 单向 ~64 GB/s，70B 模型一个长上下文请求 KV 可达 GB 级，swap 一次几十到几百 ms
2. **GPU 算力富余**：现代 H100/B200 算力 dense，prefill 100 ms 内能干完几千 token
3. **Prefix cache 救场**：被踢请求重新调度时，前缀部分大概率命中 cache，实际重算只是 cache miss 段——往往很短
4. **代码实现简单**：recompute 路径就是"释放 block + 重新入 waiting"，无需管 host buffer / 拷贝调度

**Swap 反而更优的场景**：

- **KV 极大 + prefix cache 命中率低**（如全新 RAG，每次 prompt 都不一样）→ recompute 等于完全重 prefill，比 swap 慢
- **Prefill 算力极度紧张**（极大 batch 同时争 GPU）→ swap 释放算力给其他 running
- **特殊硬件**（如 Grace Hopper，CPU↔GPU 带宽 900 GB/s 接近 HBM）→ swap 成本接近免费

参数：`--preemption-mode swap`（V1 默认 `recompute`）。

---

**5. 长度方差极小的 workload（1024 in / 256 out），continuous 赢 static 多少？**

**预期**：continuous 仍赢，但优势缩小到约 1.5-2× 而不是 24×。原因：

- Static batching 的痛点是"长请求拖死短请求" → 当所有请求长度一样时，这个痛点消失
- Static 仍有的劣势：
  - 每批一起进出 → 批与批之间有 prefill / setup gap
  - decode 1 token 是 memory-bound，static batch 大小固定就是固定，不会动态填充
- Continuous 仍能赢的地方：
  - 新请求随到随进，**没有批切换的间隙**
  - prefill / decode 混跑，prefill 来时不卡死 decode
  - GPU 利用率仍更高（80%+ vs 70%）

**估算**：1024 in + 256 out workload，continuous ~1.5-1.8× static。论文里 24× 的来源是 ShareGPT 真实 workload，长度方差极大（10 token 到 4096 token），static 时长被最长请求决定。

**反直觉**：合成长度均匀的 benchmark 其实对 vLLM **不利**——掩盖了它最擅长解决的问题（长度方差）。生产 workload 长度方差越大，vLLM 越能拉开差距。

## 下一步

- 下一节：[`03-kv-cache-management.md`](03-kv-cache-management.md)（深入 BlockPool / LRU / 启动 profile run / preemption 实现细节）。
- 跟着读：[`05-chunked-prefill.md`](05-chunked-prefill.md)（continuous batching 的延伸优化，解决长 prompt 卡死 decode 的问题）。
- 想看源码：`vllm/v1/core/sched/scheduler.py`（schedule 主循环）、`vllm/v1/core/sched/async_scheduler.py`（async overlap）、`vllm/v1/worker/gpu_input_batch.py`（持久 InputBatch）。
- 想动手：[`07-hands-on/03-mini-experiments.md`](../07-hands-on/03-mini-experiments.md)（调 `max_num_batched_tokens` 观察 TTFT/吞吐曲线）。
- 想从生产视角理解：[`08-production-deployment/04-autoscaling-and-capacity.md`](../08-production-deployment/04-autoscaling-and-capacity.md)（token budget 与 HPA、并发上限的关系）。
