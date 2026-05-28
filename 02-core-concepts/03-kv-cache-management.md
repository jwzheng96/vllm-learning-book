# 03. KV Cache 管理深入

> **谁该读这一篇？** 想从"知道有 BlockPool"升级到"能 trace 出一个 block 的完整生命周期"的同学；面试要回答"vLLM 启动时怎么决定 num_blocks / 怎么 preempt / hash 怎么算"的候选人。
>
> **前置阅读：** [`01-paged-attention.md`](01-paged-attention.md)（理念层）、[`02-continuous-batching.md`](02-continuous-batching.md)（schedule 与 preempt 的上层逻辑）。
>
> **耗时：** 约 20 分钟。
>
> **学完能：**
> 1. 复述启动阶段 `profile run` 算 num_blocks 的 5 步公式，并解释为什么 `--gpu-memory-utilization` 默认 0.9。
> 2. 写出 KVCacheBlock / FreeKVCacheBlockQueue / cached_block_hash_to_block 三个数据结构的不变式。
> 3. 画出一个请求"WAITING → RUNNING → FINISHED"过程中 KV block 的分配 / 复用 / 释放序列。
> 4. 给面试官讲清 chain hash 为什么必须 chain、prefix caching 的命中是"按顺序"的。

PagedAttention 是"理念"，本节是"工程实现"。看完这节，你才能讲清"vLLM 启动时怎么决定要分多少 block"。

---

## 1. 启动阶段：profile run 决定 num_blocks

vLLM 启动时不会用 `torch.cuda.memory_allocated()` 估算 KV 可用空间，而是做一次**真实的 dummy forward** 测峰值显存。流程：

```
1. 加载模型权重 → 占去 W GB
2. 构造一个 max batch、max seq_len 的假输入
3. 跑一次 forward（关 grad）→ 记录峰值显存 P GB
4. KV 可用空间 = (gpu_memory_utilization × total_mem) - P
5. num_blocks = floor(KV_avail / (block_size × per_token_kv_bytes × num_layers))
```

代码：`vllm/v1/worker/gpu_worker.py : determine_available_memory()`

参数：

- `--gpu-memory-utilization`：默认 0.9（用 90% 显存）
- `--kv-cache-dtype`：默认与模型 dtype 一致；可设 fp8 减半

启动日志会打印类似 `# GPU blocks: 12345`，这就是 num_blocks。

---

## 2. BlockPool：物理 block 的中央仓库

`vllm/v1/core/block_pool.py : BlockPool` 维护：

```python
class BlockPool:
    # 所有物理 block 的元数据
    blocks: list[KVCacheBlock]      # 长度 = num_blocks

    # 空闲列表（双向链表，便于 LRU 复用）
    free_block_queue: FreeKVCacheBlockQueue

    # Hash → block 的索引（用于 prefix caching 命中）
    cached_block_hash_to_block: dict[BlockHash, list[KVCacheBlock]]
```

核心方法：

- `get_new_blocks(num_blocks)`：从 free queue 头部取
- `free_blocks(blocks)`：ref_cnt -= 1，归零则放回 free queue 尾部
- `touch(blocks)`：把命中的 block 从 free queue 拿出（被复用了）

注意 free queue 不是 LIFO 是 LRU：最近 free 的放尾巴，最早 free 的先被复用。这样**最旧的 block 才会被覆盖**，prefix cache 命中率最大。

---

## 3. KVCacheBlock 的内部状态

```python
@dataclass
class KVCacheBlock:
    block_id: int               # 物理 block 索引 (0 .. num_blocks-1)
    ref_cnt: int = 0            # 引用数
    block_hash: Optional[...]   # 内容哈希（用于 prefix caching）
    prev_free_block: ...        # free queue 前驱
    next_free_block: ...        # free queue 后继
```

关键不变式：

- `ref_cnt == 0` ⟺ block 在 free queue 里
- `ref_cnt > 0` ⟺ block 被至少一个请求引用
- `block_hash != None` ⟺ block 已写满且参与 prefix caching

---

## 4. 一个请求的 KV 分配生命周期

以一个 prompt=50 token、生成 100 token 的请求为例（block_size=16）：

```
T=0: 入队 (WAITING)
  - block_table = []
  - num_computed_tokens = 0

T=1: schedule 时调用 KVCacheManager.allocate_slots(req, 50)
  - 需要 ceil(50/16) = 4 个 block
  - 查 prefix cache: 假设没有命中
  - 从 BlockPool 取 4 个新 block: [42, 17, 99, 3]
  - ref_cnt[42,17,99,3] += 1
  - req.block_table = [42, 17, 99, 3]
  - status → RUNNING

T=2: prefill 完成，生成第 51 个 token
  - 第 51 token 落在 block 3 的位置 50%16=2（还有空间）
  - 不分新 block
  - block 99 现在"满了"（16 token），计算它的 hash，挂到 hash table
    （这样未来其他请求的 prefix 包含同样内容时能命中）
  - block 3 还没满，不计算 hash

T=3..N: decode 每步
  - 每步加 1 token
  - 落在当前末尾 block 内时不分配
  - 越过 block 边界时，allocate_slots(req, 1) 取 1 个新 block

T=N+1: 生成 EOS
  - status → FINISHED
  - KVCacheManager.free(req)
    - 对每个 block，ref_cnt -= 1
    - 归零的 block 放回 free queue 尾部
    - 如果开启 prefix caching，hash 保留（block 还能被命中复用）
```

---

## 5. Prefix Caching 的 hash 机制

这是面试高频考点。

### 5.1 Hash 的输入
对每个 block，hash 包含：

- block 内的所有 token_ids
- 前一个 block 的 hash（chain，保证位置敏感）
- 额外的"上下文"：LoRA 适配器、多模态 placeholder 等

代码：`vllm/v1/core/kv_cache_utils.py : hash_request_tokens / BlockHash`

### 5.2 为什么要 chain 前一个 hash？
因为 token "你好" 在不同前缀下产生的 KV 不同（取决于前面）。chain hash 保证：相同前缀且相同当前 block 才命中。

伪代码：

```python
def block_hash(token_ids, prev_hash, extra_keys):
    return hash((prev_hash, tuple(token_ids), extra_keys))
```

### 5.3 命中流程
```python
def allocate_slots(req):
    # 1. 算出 req 全部 prompt 的每个 block 的 hash
    hashes = compute_block_hashes(req.prompt_token_ids)

    # 2. 查 BlockPool.cached_block_hash_to_block
    for h in hashes:
        if h in pool.cached_hash:
            existing_block = pool.cached_hash[h]
            # 命中！直接挪用
            existing_block.ref_cnt += 1
            if existing_block.ref_cnt == 1:
                pool.touch(existing_block)  # 从 free queue 拿出
            req.block_table.append(existing_block.block_id)
        else:
            break  # 第一个 miss 之后就不再命中（前缀性质）

    # 3. 剩下的 block 用 get_new_blocks 分配
```

**重点**：命中是按顺序的，第一个 miss 后面全部 miss。因为前缀 hash 是链式的。

### 5.4 多请求并发的复用
```
T=0: req_A 完成 prefill，blocks=[42, 17, 99]，hash 都注册了
T=1: req_A FINISHED，3 个 block ref_cnt → 0，进 free queue
     但 hash 仍在 cached_hash_to_block 中
T=2: req_B 用同样 system prompt 进入
     - 计算 hash 命中 42、17、99
     - 把它们从 free queue 取出，ref_cnt = 1
     - req_B 跳过这 3 个 block 的 prefill 计算！直接从第 4 个 block 开始
```

这就是为什么 chatbot 重复用同一个 system prompt 能极大降 TTFT。

---

## 6. 怎么决定 num_scheduled_tokens？

KVCacheManager 的核心算法（简化）：

```python
def allocate_slots(req, num_new_tokens):
    # 当前已分配的 block 数
    cur_blocks = len(req.block_table)
    # 需要的总 block 数
    new_total_tokens = req.num_computed_tokens + num_new_tokens
    need_blocks = ceil(new_total_tokens / block_size)
    # 需要再分多少
    delta = need_blocks - cur_blocks

    if delta == 0:
        return True  # 现有 block 够用

    # 减去 prefix cache 命中节省的
    delta -= hits_for_uncomputed_part(req)

    if pool.free_count() < delta:
        return False  # 不够，需要 preempt 或拒绝

    new_blocks = pool.get_new_blocks(delta)
    req.block_table.extend(b.block_id for b in new_blocks)
    return True
```

---

## 7. 抢占（preemption）的实现细节

```python
def _preempt(self, req):
    # 1. 把 req 的所有 block ref_cnt -= 1
    self.kv_cache_manager.free(req)
    # 2. 改状态
    req.status = RequestStatus.WAITING_PREEMPTED
    req.num_computed_tokens = 0   # recompute 模式
    # 3. 从 running 队列移到 waiting 队列头部（保证它优先重启）
    self.running.remove(req)
    self.waiting.appendleft(req)
```

选谁来 preempt？默认 FCFS 倒序（最晚到的先牺牲）；priority 模式下选优先级最低的。

V1 几乎总是用 recompute 而非 swap，理由前面讲过（PCIe 慢 + prefix cache 兜底）。

---

## 8. KV Cache 的 dtype

- `--kv-cache-dtype auto`：跟模型 dtype（FP16/BF16）
- `--kv-cache-dtype fp8`（或 fp8_e4m3 / fp8_e5m2）：KV 存 FP8，每 token 占用减半，**等价于 num_blocks ×2**

FP8 量化 KV 在 attention 算之前 dequantize，引入精度损失。实测 perplexity 损失 < 1%。

---

## 9. 多 KV 类型：Mamba / Linear Attention

vLLM 的 KVCacheManager 已经被扩展为**多类型并存**：

- 普通 Transformer 层：上面讲的 paged KV
- Mamba SSM 层：固定大小的 state（不是 KV）
- Linear / sparse / sliding window：不同 block 结构

代码：`vllm/v1/core/single_type_kv_cache_manager.py` 和 `kv_cache_coordinator.py` 协调多种 manager。

面试不会深问，但你应该知道 vLLM 已经不只是"Transformer paged KV"了。

---

## 10. 面试追问汇总

**Q: vLLM 启动慢，为什么？**
A: profile run 阶段会跑 dummy forward + 可能 capture CUDA Graph 多个 batch_size + torch.compile，加起来几十秒到几分钟。可以用 `--enforce-eager` 关 CUDA Graph 加速启动（但跑慢）。

**Q: gpu_memory_utilization 设 0.95 行不行？**
A: 风险大。profile run 测的是单一 batch 的峰值，runtime 实际可能有 activation 临时高峰。0.9 给 10% 缓冲是经验值。

**Q: 一个 block 内 token 数量比 block_size 小时，attention 怎么处理？**
A: 通过 `seq_lens` / `block_table` 中的有效长度告诉 kernel "这个请求总共 N 个 token，超过 N 的位置 mask 掉"。kernel 内部按 block 加载，但 softmax 时排除无效位置。

**Q: prefix caching 会不会出错？比如 hash 冲突？**
A: 用 Python `hash()` 或更安全的 SHA-1（vLLM 用过 xxhash）。冲突概率极低，且即使冲突会被 token id 序列比对二次验证。

---

## 小结

- 启动时通过 **profile run** 实测峰值显存，再按 `(gpu_mem_util × total) - peak` 算 KV 可用空间，得到 `num_blocks`——这是 `--gpu-memory-utilization` 留 0.9 给 activation 余量的原因。
- **BlockPool** 是物理 block 的中央仓库：`blocks` 元数据数组 + LRU **FreeKVCacheBlockQueue** + **cached_block_hash_to_block** 索引。三个不变式：`ref_cnt==0 ⟺ 在 free queue`、`ref_cnt>0 ⟺ 被引用`、`block_hash!=None ⟺ 写满且参与 prefix cache`。
- 一个请求的 KV 一生：WAITING → 调度时 `allocate_slots` → RUNNING → 每越 block 边界再 allocate 1 → FINISHED 时 `free()` 把 ref_cnt-=1，归零的 block 进 free queue 尾（**但 hash 保留**供未来命中）。
- Prefix caching 用**链式 hash**（`hash((prev_hash, tokens, extra_keys))`）保证位置敏感；命中按顺序，第一个 miss 之后全部 miss。
- Preemption 默认走 **recompute**：释放所有 block + 状态回到 WAITING，下次 schedule 时重新 prefill（重算时还能命中 prefix cache）。FP8 KV 是免费翻倍 num_blocks 的常用手段。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. prompt=50, 生成 100 token, block_size=16, 画 KV block 一生。**

- prompt=50 token → 需要 ⌈50/16⌉ = **4 个 block**（最后一个只用 2 个 slot）
- 生成 100 token → 累计 150 token，需要 ⌈150/16⌉ = **10 个 block**
- 期间需要 alloc 6 个新 block（block 4..9）

```
T=0  (prefill 进 batch):   alloc block 0,1,2,3  → ref_cnt 全 = 1
T=1..N (decode 累积 token):
  当 token 数到达 64 时（block 3 已经满）→ alloc block 4 → ref_cnt[4]=1
  当 token 数到达 80 → alloc block 5
  ...
  当 token 数到达 144 → alloc block 9
T=END (请求 finish):       block 0..9 ref_cnt -= 1 → 全归 0
  → 全部回 free_queue 尾（如果有 prefix cache，被 hash 命中的 block 仍可被未来请求 ref_cnt++ 复用）
```

**如果有共享 prefix**（比如 system prompt 占 block 0-2，跟另一个请求一样）：

- 启动时：block 0-2 是 hash 命中 → ref_cnt++ 共享，**只新 alloc block 3 + 后续**
- 完成时：block 0-2 ref_cnt-- 回到原值（可能 > 0，仍在共享）；block 3+ ref_cnt → 0 回 free queue

---

**2. `hash_block_tokens(tokens, prev_hash, extra_keys)` 伪代码 + 不 chain 的 bug。**

```python
NONE_HASH = bytes(32)  # 哨兵

def hash_block_tokens(hash_function, prev_hash, tokens, extra_keys=None):
    if prev_hash is None:
        prev_hash = NONE_HASH
    # 关键：把 prev_hash 也进 hash → Merkle 链
    return hash_function((prev_hash, tuple(tokens), extra_keys))
```

**不 chain prev_hash 的 bug**：

假设两个请求：

- A: tokens [1,2,3,4,...,16, 99,100,...]  (block 0 = [1..16], block 1 = [99,100,...])
- B: tokens [5,6,7,8,...,20, 99,100,...]  (block 0 = [5..20], block 1 = [99,100,...])

不 chain 时：A 的 block 1 hash = hash([99,100,...])；B 的 block 1 hash = 同样 = hash([99,100,...])。

**结果**：B 错误命中 A 的 block 1 KV，但 attention 历史完全不同——B 的 attention 应该 attend 到 [5..20]，实际拿到的是 A 的 [1..16] 的 K/V。**输出乱码**。

chain prev_hash 后：A 的 block 1 hash = hash(hash([1..16]), [99,100,...])，B 的 block 1 hash = hash(hash([5..20]), [99,100,...])，**完全不同**，不会误命中。

→ Merkle chain 让"前缀完全相同"才命中，是 prefix caching 正确性的根基。

---

**3. `FreeKVCacheBlockQueue` 为什么 LRU 而不是 LIFO？**

源码：`vllm/v1/core/block_pool.py` 的 `FreeKVCacheBlockQueue`（双向链表）。

- 入队（block ref_cnt 归零时）：append 到链表**尾部**
- 出队（alloc 新 block 时）：从链表**头部**取

**为什么不 LIFO（从尾取）？**

- LIFO：最近释放的 block 先被覆盖 → **prefix cache 立刻丢失**。比如刚结束的 chatbot 对话，10 秒后用户接着问 → 同 prefix 但 KV 已经被覆盖
- LRU：最早释放的先被覆盖 → 给 prefix cache **更长存活时间**

**实测影响**：LRU 比 LIFO 在多轮对话 / RAG 等场景下 prefix cache 命中率高 2-5×，对应 TTFT 收益巨大。

代价：LRU 实现复杂一些（双向链表 + dict 维护"block → list_node"映射，O(1) 删除）。但相比命中率收益完全划得来。

---

**4. V1 总是用 recompute 的 3 条理由 + 什么硬件上 swap 合理。**

**Recompute 偏好的 3 条理由**：

1. **PCIe 太慢**：H100 PCIe Gen5 ×16 单向 ~64 GB/s。Llama-70B 长 2500 token KV ≈ 2.5 GB → swap out 40 ms + swap in 40 ms = 80 ms 通信。而 recompute 一个 2500-token 的 prefill 在 H100 上也就 100-150 ms，相差不大
2. **Prefix cache 救场**：被踢请求重新进 waiting 后，多数情况能命中之前 cached 的 block（同 hash），实际只重算 cache miss 段——通常远小于全部 prefill。综合 recompute 比 swap 快
3. **实现简单**：recompute = "free 所有 block + 回到 WAITING 状态"，无需管理 host buffer 池、PCIe DMA 调度、双端同步

**Swap 反而合理的场景**：

- **Grace Hopper / GH200**：CPU↔GPU 900 GB/s NVLink-C2C，swap 接近免费
- **超长上下文 + 极低 prefix cache 命中**：每次 prompt 都不一样，recompute 等于全部重 prefill 没意义
- **prefill 算力严重 contended**：大并发同时争 GPU，swap 释放 GPU 给其他 running 任务

参数：`--preemption-mode swap`（默认 `recompute`）。

---

**5. H100 80GB / Llama-3-8B / BF16，切到 fp8 KV num_blocks 多多少？**

**BF16 基线**：

- Llama-3-8B：32 层、num_kv_heads=8（GQA）、head_dim=128
- 单 token KV = 2 × 8 × 128 × 32 × 2 = 131072 字节 = **128 KB**
- 单 block = 16 token × 128 KB = **2 MB**
- 模型权重 8B × 2 = 16 GB
- 可用 KV 显存 ≈ 80 - 16 - 6（CUDA / 激活预算）= **58 GB**
- num_blocks = 58 GB / 2 MB = **29,696 个 block** ≈ 475,136 token

**FP8 KV**：

- 单 token KV 字节减半 = 64 KB
- 单 block = 1 MB
- 可用 KV 显存还是 58 GB（fp8 不影响模型权重）
- num_blocks = 58 GB / 1 MB = **59,392 个 block** ≈ 950,272 token

→ **block 数翻倍**，理论并发 / 并发上下文长度也接近翻倍。

**注意**：

- fp8 KV 是有损精度（per-tensor 或 per-channel scale + fp8 encoding）。多数模型 perplexity 影响 < 1%，但极敏感场景（math reasoning）要 ablation 验证
- 启用：`--kv-cache-dtype fp8`（前提硬件支持 fp8，H100/H200/B200 OK，A100 不行）
- 实际吞吐增益往往低于理论 2×，因为算力 / 调度也会成为新瓶颈

## 下一步

- 下一节：[`04-prefix-caching.md`](04-prefix-caching.md)（把本节 §5 hash 机制展开到完整的 prefix caching 专题：extra_keys / LoRA / 多模态 / 跨进程共享）。
- 跟着读：[`05-chunked-prefill.md`](05-chunked-prefill.md)（与 KV 管理强相关的另一项 V1 默认行为，剩余 prefill 是怎么切的）。
- 想看源码：`vllm/v1/core/block_pool.py`（BlockPool）、`vllm/v1/core/kv_cache_manager.py`（allocate_slots / free）、`vllm/v1/core/kv_cache_utils.py`（hash 计算）、`vllm/v1/worker/gpu_worker.py`（determine_available_memory）。
- 想动手：[`07-hands-on/03-mini-experiments.md`](../07-hands-on/03-mini-experiments.md)（用 `--gpu-memory-utilization` / `--kv-cache-dtype fp8` 对比 num_blocks 与并发上限）。
- 想从生产视角理解：[`08-production-deployment/05-slo-and-observability.md`](../08-production-deployment/05-slo-and-observability.md)（`vllm:kv_cache_usage`、`vllm:prefix_cache_hit_rate` 等 KV 相关指标怎么看、报警阈值怎么定）。
