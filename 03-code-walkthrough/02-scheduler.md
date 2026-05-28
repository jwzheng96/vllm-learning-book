# 02. Scheduler 深读

> **谁该读这一篇？** 已经知道 vLLM 大致进程拓扑，准备死磕 V1 Scheduler 这 2300 行核心代码的工程师；面试想答清 token budget、preempt、chunked prefill 的同学。
>
> **前置阅读：** [`01-entry-points.md`](01-entry-points.md)、[`02-continuous-batching.md`](../02-core-concepts/02-continuous-batching.md)、[`05-chunked-prefill.md`](../02-core-concepts/05-chunked-prefill.md)
>
> **耗时：** 约 18 分钟
>
> **学完能：**
> 1. 在白板上画出 `schedule()` 的 5 个阶段（A-E）以及每步对 token budget 的扣减
> 2. 解释 `_schedule_running` 与 `_schedule_waiting` 各自承担的角色和优先级关系
> 3. 准确说出 preempt 的触发条件、被踢顺序，以及 recompute vs swap 两种模式
> 4. 解释 AsyncScheduler 把哪一段时间和 forward overlap，理论收益是多少

`vllm/v1/core/sched/scheduler.py` 是 vLLM 最重要的代码文件（2300+ 行）。看懂它就看懂了 vLLM 的灵魂。本节按 `schedule()` 的执行顺序拆解。

---

## 1. Scheduler 的状态

```python
class Scheduler:
    # 队列
    waiting: deque[Request]          # 等待进入 batch
    running: list[Request]           # 正在跑

    # 配置
    scheduler_config: SchedulerConfig
    cache_config: CacheConfig

    # 核心组件
    kv_cache_manager: KVCacheManager   # 管 block 分配
    encoder_cache_manager: ...          # 多模态用
    structured_output_manager: ...      # JSON / grammar 约束生成
    connector: KVConnector | None       # 跨节点 KV 传输

    # 统计
    num_lookahead_tokens: int           # 投机解码用
    finished_req_ids: set[str]          # 上一步刚完成的
```

---

## 2. `schedule()` 的整体框架

```python
def schedule(self) -> SchedulerOutput:
    # === Step A: 重置每步的临时状态 ===
    scheduled_new_reqs = []
    scheduled_cached_reqs = []
    num_scheduled_tokens = {}
    token_budget = self.max_num_scheduled_tokens

    # === Step B: 先调度 running 队列 ===
    # 保证已经在跑的请求至少 decode 1 token，剩余 budget 用于 chunked prefill
    req_to_new_block_ids = self._schedule_running(token_budget, num_scheduled_tokens)

    # === Step C: 再调度 waiting 队列 ===
    # 看 KV、token budget 是否允许新请求加入
    new_running = self._schedule_waiting(token_budget, num_scheduled_tokens)

    # === Step D: Encoder budget（多模态）===
    # vision encoder 也有独立的 budget

    # === Step E: 组装 SchedulerOutput ===
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=scheduled_cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        preempted_req_ids=self.preempted_req_ids,
        finished_req_ids=self.finished_req_ids,
        ...
    )
```

---

## 3. `_schedule_running`：保住老用户

伪代码（实际更复杂，含 spec decoding、KV connector 等）：

```python
def _schedule_running(self, token_budget, num_scheduled_tokens):
    new_running = []

    for req in self.running:
        # 决定这个请求本步算几个 token
        num_new_tokens = self._get_num_new_tokens_for_req(req, token_budget)

        if num_new_tokens == 0:
            # budget 用完
            break

        # 算需要多少新 block
        while not self.kv_cache_manager.allocate_slots(req, num_new_tokens):
            # KV 不够！要 preempt
            victim = self.running.pop()    # 通常从队尾找
            if victim is req:
                # 没人可以踢了，这个请求自己被踢
                self._preempt(req)
                num_new_tokens = 0
                break
            self._preempt(victim)
        else:
            num_scheduled_tokens[req.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            new_running.append(req)

    self.running = new_running
```

**核心逻辑**：

1. 给每个 running 请求分配本步要算几个 token
2. 如果 KV 不够，从队尾踢人（preempt）
3. 踢到自己头上就把自己设回 waiting

---

## 4. `_get_num_new_tokens_for_req`：本步算几个 token？

```python
def _get_num_new_tokens_for_req(self, req, token_budget):
    # 剩余要算的 prompt token 数
    remaining_prompt = req.num_prompt_tokens - req.num_computed_tokens
    
    if remaining_prompt > 0:
        # 还在 prefill 阶段 → chunk
        return min(remaining_prompt, token_budget, self.max_chunk_size)
    else:
        # 已经在 decode 阶段
        # 投机解码会一次提议多个，普通模式是 1
        return min(1 + self.num_lookahead_tokens, token_budget)
```

这就是 chunked prefill 的核心：把 "请求长度 - 已算" 切成 budget 大小。

---

## 5. `_schedule_waiting`：让新人进来

```python
def _schedule_waiting(self, token_budget, num_scheduled_tokens):
    while self.waiting and token_budget > 0:
        req = self.waiting[0]   # peek 不 pop

        # 检查多模态 encoder 是否有空间（不展开）
        if not self._can_schedule_encoder(req):
            break

        # 算 num_new_tokens
        num_new_tokens = min(req.num_prompt_tokens, token_budget, max_chunk_size)

        # 尝试分配 KV
        if not self.kv_cache_manager.allocate_slots(req, num_new_tokens):
            break   # 不够，停止往 batch 加新人

        # 加入 batch
        self.waiting.popleft()
        req.status = RUNNING
        self.running.append(req)
        num_scheduled_tokens[req.request_id] = num_new_tokens
        token_budget -= num_new_tokens
```

**注意**：

- waiting 队列是按 FCFS（或 priority）排序的
- 只要 KV 不够或 budget 用完，立刻停止往 batch 塞，**不会跳过等小请求**（避免饥饿）

---

## 6. `update_from_output`：处理上一步的结果

每个 step 结束后，把 Worker 返回的 sampled_token_ids 更新到 Request 上：

```python
def update_from_output(self, scheduler_output, model_output):
    self.finished_req_ids.clear()

    for req_id, sampled_tokens in model_output.sampled_token_ids.items():
        req = self.requests[req_id]

        # 1. 把新 token 加到 output
        req.output_token_ids.extend(sampled_tokens)
        req.num_computed_tokens = scheduler_output.num_scheduled_tokens[req_id]
                                  + req.num_computed_tokens

        # 2. 检查是否完成
        if (sampled_tokens[-1] == req.sampling_params.eos_token_id
            or len(req.output_token_ids) >= req.sampling_params.max_tokens
            or self._check_stop_strings(req)):
            self._finished(req)
            self.finished_req_ids.add(req_id)
            continue

        # 3. 检查 stop / structured output / spec decoding 等

    # 4. 收集本步的 EngineCoreOutput，返回给 EngineCore
    return [EngineCoreOutput(req_id, new_tokens) for ...]
```

---

## 7. Preemption 的实际策略

```python
def _preempt(self, req, preemption_mode="recompute"):
    # 释放 KV
    self.kv_cache_manager.free(req)
    
    if preemption_mode == "recompute":
        req.num_computed_tokens = 0
    else:  # swap
        self.kv_cache_manager.swap_out_to_cpu(req)
    
    req.status = RequestStatus.WAITING
    self.waiting.appendleft(req)  # 重启时优先级最高
```

V1 默认 recompute，前面解释过。

被 preempt 谁？默认从 running 队尾倒着踢（最晚到的最先牺牲）。priority 模式下选 priority 最低的。

---

## 8. 一些"高级"逻辑（速览，面试可不展开）

- **结构化输出（structured output）**：JSON schema / grammar 约束 → 每步 logits 加 mask。Scheduler 会维护 grammar state machine 每步推进。
- **投机解码（spec decoding）**：Scheduler 一次性给一个请求分配 `1 + num_lookahead_tokens` 个 token，Worker 返回的是 "提议 - 验证" 结果。
- **多模态 encoder budget**：vision encoder 算图像 embedding 是独立的，要单独计 budget。
- **KV connector（disaggregated prefill）**：跨节点传 KV，Scheduler 协调"传完了再 decode"的同步。
- **DP / EP**：data parallel 与 expert parallel 下的调度协调（多个 EngineCore 协同）。

---

## 9. Async Scheduler

`vllm/v1/core/sched/async_scheduler.py`：让 `schedule()` 跟上一步的 forward overlap。

实现思路：

- 普通 Scheduler 是 "schedule → forward → update → schedule → ..."
- AsyncScheduler 拆成两个 task：
  - schedule_task：每次给 worker 发任务时，**先发任务**，再 await 上次的输出
  - 也就是说 step N+1 的 schedule 在 step N 的 forward 还在跑时启动

CPU/GPU overlap 节省 5-10%。

---

## 10. 你应该精读的代码片段

打开 `vllm/v1/core/sched/scheduler.py`，重点看：

1. `__init__`：所有依赖看一遍
2. `schedule()`：主入口
3. `_schedule_running()`：含 preemption 逻辑
4. `_schedule_waiting()`：含 prefix caching 集成
5. `_get_num_new_tokens_for_req()`：chunked prefill 实现处
6. `_preempt()`：抢占细节
7. `update_from_output()`：状态更新与完成检测

预计 1-2 小时能扫完。

---

## 小结

- `schedule()` 一次的核心顺序是"先 running 再 waiting"，token budget 是贯穿所有分支的全局账本。
- `_schedule_running` 保 SLA，KV 不够就从队尾 preempt；`_schedule_waiting` 引入新人，KV/budget 不够立刻停手，避免饥饿。
- chunked prefill 的入口就一个函数 `_get_num_new_tokens_for_req`，由剩余 prompt 与 budget 取 min 决定本步算多少。
- AsyncScheduler 把"下一步的 schedule"塞到"本步 forward"的等待区，是低开销但很值的 CPU/GPU overlap。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. preempt victim 从队头还是队尾选？**

**FCFS 模式**：队尾（`self.running.pop()`，最近加入 batch 的）。代码（约 line 450）：

```python
if self.policy == SchedulingPolicy.FCFS:
    preempted_req = self.running.pop()         # 队尾，最后加入的
```

**PRIORITY 模式**：`max((priority, arrival_time))`——优先级最低（priority 数值最大）的，同优先级里最晚到的：

```python
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time),
    )                                          # 优先级最低、最晚到的
    self.running.remove(preempted_req)
```

**为什么 FCFS 选队尾？** 队头是已经跑了最久的请求，踢它损失最大；队尾是最新加入的，工作量小，踢它代价低。这是 SRPT（最短剩余处理时间优先）的近似。

详见 [`02b-scheduling-policies.md`](02b-scheduling-policies.md)。

---

**2. 32k prefill + `max_num_batched_tokens=2048`，要几个 step 跑完 prefill？**

每个 step 最多算 2048 token。但**实际每步要先扣 running decode**——假设 batch 里没有其他 running 请求，每步 prefill 满 2048：

```
step 1:  prefill chunk 0..2048    剩 30720 token
step 2:  prefill chunk 2048..4096 剩 28672 token
...
step 16: prefill chunk 30720..32768 剩 0
step 17: decode 第一个新 token
```

→ **16 个 step 跑完 prefill**，第 17 步开始 decode。

如果同时有 N 个 running decode 请求争 budget：

- 每步先扣 N（每 running 1 token）
- 剩余 2048-N 给 prefill chunk
- 总 step 数 = ⌈32768 / (2048-N)⌉

例：N=16 个 decode，每 step prefill 2032 token → 需要 ⌈32768/2032⌉ = **17 个 step** 跑完 prefill。

**时间轴**（gantt 风格）：

```
T=0  [decode×16 + prefill 2032 (chunk 0)]   ← 17 个用户在 decode 中, prefill 在并行
T=1  [decode×16 + prefill 2032 (chunk 1)]
T=2  [decode×16 + prefill 2032 (chunk 2)]
...
T=16 [decode×16 + prefill 2032 (chunk 16)]  ← prefill 完成
T=17 [decode×17] (新 prefill 请求开始 decode)
```

→ **这就是 chunked prefill 的核心好处**：长 prefill 不会阻塞其他人的 decode，每个 step 时长稳定（约 2048 token），TPOT 平稳。

---

**3. `update_from_output` 中检测"请求完成"的 3 种条件 + 分支位置。**

源码：`vllm/v1/core/sched/scheduler.py::_update_request_with_output`（行号会变）

```python
# ① EOS token
if new_token_id == request.sampling_params.eos_token_id:
    request.status = RequestStatus.FINISHED_STOPPED
    return True
# 或：if new_token_id in stop_token_ids:

# ② max_tokens 达到
if request.num_output_tokens >= request.sampling_params.max_tokens:
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return True

# ③ stop strings（detokenize 后字符串匹配）
if detokenized.contains_any(request.sampling_params.stop):
    request.status = RequestStatus.FINISHED_STOPPED
    return True
```

完整代码位置：

- EOS：通常在 `_check_stop()` 或 `_check_finish()` 内，对照 `eos_token_id` / `stop_token_ids`
- max_tokens：对照 `num_output_tokens >= max_tokens`
- stop strings：往往在 detokenizer 端做（`vllm/v1/engine/detokenizer.py`），因为需要解码完整字符串才能匹配

**`finish_reason`** 字段记录是哪种：`stop` / `length` / `abort`，对应 `vllm:request_success` 这个 Prometheus counter 的 label。

---

**4. AsyncScheduler 多了哪个 task？为什么能 overlap？正确性问题？**

**多了什么**：AsyncScheduler 在 `step()` 中**提前发起下一步的 schedule**，让它与本步的 GPU forward 重叠：

```python
# 普通 Scheduler
def step():
    so = schedule()                  # CPU
    out = execute_model(so)          # GPU (forward)
    update_from_output(so, out)

# AsyncScheduler
def step():
    out = self._pending_forward.result()    # 等上一步 forward
    update_from_output(self._last_so, out)
    self._last_so = schedule()              # 启动新 schedule
    self._pending_forward = execute_model_async(self._last_so)  # 异步发 forward
    # 函数返回前 forward 还在跑；下一个 step 进来时再 .result() 收
```

**为什么能 overlap**：CPU schedule 与 GPU forward 不依赖同一资源（CPU vs GPU stream），可以真正并行。

**正确性问题与解法**：

- **新 token 状态没更新就 schedule 下一步**？不会——schedule 不需要新 token 内容，它只需要"running 列表 + KV 占用"，这些上一步已经记好
- **新请求 add_request 时机**？AsyncScheduler 仍在 schedule 前同步 add，不会丢失
- **preempt 导致状态不一致**？这是真坑——AsyncScheduler 有专门处理"上一步触发 preempt 后下一步的 running 列表"的逻辑（`vllm/v1/core/sched/async_scheduler.py`），通过把 preempted 请求标记后再继续

整体收益 5-10%，详见 [`02-architecture.md`](../01-overview/02-architecture.md) §6 与本节自检题 5。

## 下一步

- 下一节：[`03-kv-cache-manager.md`](03-kv-cache-manager.md)（Scheduler 调用的 `allocate_slots` 内部到底做了什么）
- 想看源码：`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/sched/async_scheduler.py`
- 想动手：[`07-hands-on/03-mini-experiments.md`](../07-hands-on/03-mini-experiments.md)（造 preempt、观察调度行为）
- 想从生产视角理解：[`08-production-deployment/04-autoscaling-and-capacity.md`](../08-production-deployment/04-autoscaling-and-capacity.md)（max-num-batched-tokens、max-num-seqs 等调度参数如何影响容量规划）

