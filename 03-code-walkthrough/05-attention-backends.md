# 05. Attention Backends 源码深读

> **谁该读这一篇？** 想搞懂 vLLM 怎么把 attention 抽象成可插拔后端、不同硬件/模型对应哪个 backend 的工程师；准备答清"FlashAttention vs FlashInfer vs MLA"等面试题的同学。
>
> **前置阅读：** [`01-paged-attention.md`](../02-core-concepts/01-paged-attention.md)、[`04-model-runner.md`](04-model-runner.md)、[`03-kv-cache-manager.md`](03-kv-cache-manager.md)
>
> **耗时：** 约 18 分钟
>
> **学完能：**
> 1. 列出 vLLM 主流 attention backend 和它们的适用硬件 / 模型
> 2. 解释 Backend / Impl / MetadataBuilder 三件套接口的职责划分
> 3. 默写 KV cache 张量的 5 维 shape，每一维含义和 FlashAttention 期望布局
> 4. 在白板上画出 query_len / context_len / seq_len 的关系
> 5. 描述 cascade attention 何时启用、解决什么问题

目录：`vllm/v1/attention/backends/`。vLLM 把"怎么算 attention"完全抽象成可插拔后端。本节看 backend 接口 + FlashAttention 实现。

---

## 1. 后端总表

`vllm/v1/attention/backends/registry.py` 定义了所有后端：

```python
class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    FLASH_ATTN          = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_ATTN_DIFFKV   = "...flash_attn_diffkv.FlashAttentionDiffKVBackend"
    TRITON_ATTN         = "...triton_attn.TritonAttentionBackend"
    ROCM_ATTN           = "...rocm_attn.RocmAttentionBackend"
    ROCM_AITER_MLA      = "...mla.rocm_aiter_mla.AiterMLABackend"
    ROCM_AITER_FA       = "...rocm_aiter_fa.AiterFlashAttentionBackend"
    FLASHINFER          = "...flashinfer.FlashInferBackend"
    FLASHINFER_MLA      = "...mla.flashinfer_mla.FlashInferMLABackend"
    TRITON_MLA          = "...mla.triton_mla.TritonMLABackend"
    CUTLASS_MLA         = "...mla.cutlass_mla.CutlassMLABackend"
    FLASHMLA            = "...mla.flashmla.FlashMLABackend"
    FLASH_ATTN_MLA      = "...mla.flashattn_mla.FlashAttnMLABackend"
    FLEX_ATTENTION      = "...flex_attention.FlexAttentionBackend"
    CPU_ATTN            = "...cpu_attn.CPUAttentionBackend"
    TURBOQUANT          = "...turboquant_attn.TurboQuantAttentionBackend"
    # ...
```

**直觉记法**：

- 普通 Transformer attention：FLASH_ATTN / FLASHINFER / TRITON_ATTN
- DeepSeek MLA：所有名字含 `MLA` 的（FLASHMLA、CUTLASS_MLA、TRITON_MLA…）
- AMD ROCm 平台：ROCM_* 系列
- 实验 / 长尾：FLEX_ATTENTION（动态 mask）、TURBOQUANT、CPU_ATTN

---

## 2. 抽象接口

`vllm/v1/attention/backend.py`（注意是单数 backend，定义基类）：

```python
class AttentionBackend(abc.ABC):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]: ...
    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]: ...
    @staticmethod
    def get_kv_cache_shape(...): ...
    # 各种 supports_xxx() 用于自动选择
```

三件套：

- **Backend**：静态信息（名字、KV shape、能力声明）
- **Impl**：实际跑 attention 的 forward
- **MetadataBuilder**：每步 build attention 需要的元数据

---

## 3. FlashAttentionBackend：默认后端

`vllm/v1/attention/backends/flash_attn.py:69`

```python
class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        # FlashAttention 期望 layout：
        # [2, num_blocks, block_size, num_kv_heads, head_size]
        #  ↑              ↑
        #  K, V 各一份      block 维度
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def supports_head_size(head_size: int) -> bool: ...
    @staticmethod
    def supports_kv_cache_dtype(kv_cache_dtype) -> bool: ...
    @staticmethod
    def supports_compute_capability(capability) -> bool:
        # SM80+ (A100)、SM89 (L40)、SM90 (H100)
        ...
```

---

## 4. FlashAttentionMetadata：每步的"地图"

`flash_attn.py:226`，注释画了精彩的一图：

```python
@dataclass
class FlashAttentionMetadata:
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int        # batch 内有效 token 数（不含 padding）
    max_query_len: int            # 最大 query 长度（决定 grid）
    query_start_loc: torch.Tensor # cumsum，定位每个请求在 packed 输入里
    max_seq_len: int              # 最大 seq 长度（含 KV cache 历史）
    seq_lens: torch.Tensor        # 每个请求当前 seq 长度
    block_table: torch.Tensor     # 读 KV 的间接索引
    slot_mapping: torch.Tensor    # 新 token 写到哪里

    # cascade attention：共享前缀的 attention 优化
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # 长上下文：context parallel decode
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None
```

**学懂这张图，你才能解释"query_len、context_len、seq_len 的区别"**：

- `context_len`：N-1 步结束时已有的 token（KV 已写好的部分）
- `query_len`：本步要新算的 token
- `seq_len`：context_len + query_len，总长度

---

## 5. FlashAttentionImpl.forward（line 677）

简化版（去掉 encoder/cascade/dcp 分支）：

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
    """
    query: [num_tokens, num_heads, head_size]
    key:   [num_tokens, num_kv_heads, head_size]   ← GQA 时 num_kv_heads < num_heads
    value: [num_tokens, num_kv_heads, head_size]
    kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
    """
    if attn_metadata is None:
        return output.fill_(0)   # profile run

    num_actual_tokens = attn_metadata.num_actual_tokens

    # 1. 拆 KV cache 为 K cache 和 V cache
    key_cache, value_cache = kv_cache.unbind(0)

    # 2. 必要时 view 成 FP8 dtype
    if is_quantized_kv_cache(self.kv_cache_dtype):
        dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
        key_cache = key_cache.view(dtype)
        value_cache = value_cache.view(dtype)

    # 3. 调用 FlashAttention varlen + paged
    flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k_cache=key_cache,
        v_cache=value_cache,
        out=output[:num_actual_tokens],
        cu_seqlens_q=attn_metadata.query_start_loc,
        seqused_k=attn_metadata.seq_lens,
        max_seqlen_q=attn_metadata.max_query_len,
        max_seqlen_k=attn_metadata.max_seq_len,
        block_table=attn_metadata.block_table,
        causal=attn_metadata.causal,
        # 还要传 new K/V，让 kernel 顺手写入 KV cache
        k=key[:num_actual_tokens],
        v=value[:num_actual_tokens],
        slot_mapping=attn_metadata.slot_mapping,
        ...
    )
    return output
```

注意几个点：

1. **K/V 写入 cache 是 attention kernel 顺手做的**——不是单独一次 op。`slot_mapping` 决定写哪。
2. **block_table 是 paged 寻址的核心**：kernel 内部按 `block_table[req][i]` 找到第 i 个 block 的物理位置。
3. **FlashAttention 的 varlen 模式**让我们传打平的 1D 输入 + `cu_seqlens_q`，无需 padding。

---

## 6. 后端怎么自动选择？

`vllm/v1/attention/selector.py` 根据：

- 当前硬件（compute capability）
- 模型 head_size / KV dtype / sliding window
- 用户参数 `--attention-backend`
- 各 backend 的 `supports_*()` 静态方法

选出最优后端。优先级大致：

```
NVIDIA H100+ : FLASH_ATTN (v3) > FLASHINFER
NVIDIA A100  : FLASH_ATTN (v2) > FLASHINFER
NVIDIA L40   : FLASH_ATTN (v2) > FLASHINFER
DeepSeek MLA : FLASHMLA > FLASH_ATTN_MLA > CUTLASS_MLA > TRITON_MLA
AMD MI300    : ROCM_AITER_FA > ROCM_ATTN
CPU          : CPU_ATTN
```

用户强制：`--attention-backend FLASHINFER`。

---

## 7. 几个值得了解的对比

### 7.1 FlashAttention vs FlashInfer
- 都支持 paged KV、varlen、causal、GQA
- **FlashAttention v3** 在 H100 上 prefill 大 batch 最快
- **FlashInfer** 在 decode 小 batch、长 seq 上略优（更激进的 split-KV）
- vLLM 默认 FlashAttention，benchmark 后可切 FlashInfer

### 7.2 普通 Attention vs MLA
- 普通：每层每头 K、V 单独存
- MLA：把 KV 压成 low-rank latent，存 latent + 升维投影矩阵。**KV 占用降到 ~1/10**
- DeepSeek-V2/V3 用 MLA。代码：`vllm/v1/attention/backends/mla/`

### 7.3 FlashAttention vs Triton attn
- FlashAttention：CUDA C++，性能最强
- Triton attn：Python 写的 Triton kernel，可读性好、易扩展（如加新 mask 类型）
- vLLM 提供 Triton 实现作为 fallback 和 prototype

---

## 8. Cascade Attention（高级优化）

`FlashAttentionMetadata.use_cascade`、`common_prefix_len` 等字段实现一个加速：

- 一组请求共享前缀时（prefix caching 多请求场景）
- 不重复对前缀做 attention，而是**先算 prefix 一次**，再各请求算 suffix
- 数学等价，但访存减少（共享前缀的 K/V 只读一次）

实际生效条件：`Scheduler.get_num_common_prefix_blocks(...)` 报告有公共前缀且足够长。

---

## 9. 推荐阅读顺序

1. `vllm/v1/attention/backend.py`：基类
2. `vllm/v1/attention/backends/registry.py`：后端注册表
3. `vllm/v1/attention/backends/flash_attn.py`（默认后端）：
   - `class FlashAttentionBackend`：静态方法
   - `class FlashAttentionMetadata`：那张精彩注释
   - `class FlashAttentionMetadataBuilder.build`：每步怎么造 metadata
   - `class FlashAttentionImpl.forward`：怎么调 FA kernel
4. `vllm/v1/attention/selector.py`：自动选择
5. （MLA 想了解就读）`vllm/v1/attention/backends/mla/flashmla.py`

---

## 10. 面试常见追问

**Q: vLLM 的 attention 是自己写的吗？**
A: 早期是（`csrc/attention/paged_attention_v1/v2.cu`），现在主用 FlashAttention v2/v3 和 FlashInfer 的 paged 版本——它们原生支持 `block_table`。vLLM 自己的 kernel 作为 fallback。

**Q: KV cache 在 GPU 上具体是什么布局？**
A: `[2, num_blocks, block_size, num_kv_heads, head_size]`。第 0 维是 K vs V，第 1 维是物理 block 索引。block_table 通过 indirection 找物理位置。

**Q: forward 时 K、V 是怎么写入 cache 的？**
A: 由 attention kernel 顺带完成（拿到 slot_mapping 后直接索引写入），不是单独一个 cache write op。这避免了一次 kernel launch。

**Q: query_start_loc 怎么用？**
A: 它是 cumulative sum，长度 num_reqs + 1。请求 i 的 query 在打平输入里的范围是 `[query_start_loc[i], query_start_loc[i+1])`。FlashAttention varlen 直接吃这个。

**Q: cascade attention 解决什么？**
A: 多个请求共享同一前缀时，避免重复对前缀算 attention，节省 K/V cache 访存。

---

## 小结

- attention 后端三件套：Backend（静态信息、能力声明）、Impl（forward 真正算）、MetadataBuilder（每步造元数据）。
- FlashAttention 的 KV cache 布局是 `[2, num_blocks, block_size, num_kv_heads, head_size]`，K/V 共享同一个 5D 张量。
- 关键三段长度：`context_len`（已缓存的）+ `query_len`（本步要算的）= `seq_len`；attention kernel 用 `query_start_loc` 在打平输入里定位每个请求。
- K/V 写入 KV cache 是 attention kernel 顺手做的（用 `slot_mapping`），不是单独 op，省一次 launch。
- selector 按硬件/模型自动选最优后端；MLA 走独立子目录，DeepSeek 这类模型必须用 MLA backend。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. N-1 与 N iteration 的关系（FlashAttentionMetadata 注释图复述）。**

vLLM 的 attention 设计把"上一步 forward 完成的状态"作为"本步的输入历史"：

```
iteration N-1：
  - 跑 forward，写新 K/V 到 cache 的 block_table[req] 末尾对应物理位置
  - sample 出新 token，append 到 request.token_ids
  - 更新 request.num_tokens 加 1

iteration N（本步）：
  - 本步 query = 上步 sample 出的 1 个新 token (decode) 或 prefill chunk
  - K/V cache 已含 N-1 步及之前全部 token 的 K/V
  - attention：Q @ K[0..N-1] @ V[0..N-1]
    - context_len = N-1（cached 历史）
    - query_len = 1 (decode) 或 chunk_size (prefill)
    - seq_len = context_len + query_len
  - 写新 K/V 到 cache 末尾
  - sample 出 next token
```

**关键不变量**：cache 里的 K/V 永远是"截至上一步末尾"的完整历史；本步 query 与之前 cache 拼起来做 attention，新 K/V 同步写入 cache 末尾。

---

**2. K/V 写到 cache 哪里？`slot_mapping[i]` 与 `block_table` 换算公式。**

KV cache 形状：`[2, num_blocks, block_size, num_kv_heads, head_dim]`（第 0 维：K vs V）。

每个本步 query token i 的 K/V 写入位置：

```
slot_id = slot_mapping[i]        # 由 metadata builder 提前算好
block_idx = slot_id // block_size
intra_block_offset = slot_id % block_size

K_cache[block_idx, intra_block_offset, :, :] = key[i, :, :]
V_cache[block_idx, intra_block_offset, :, :] = value[i, :, :]
```

`slot_mapping[i]` 怎么算：

```
req_id = sequence_id_of(i)
new_token_idx = num_cached_tokens_for(req_id) + (i - query_start_loc[req_id])
block_table_idx = new_token_idx // block_size
physical_block_id = block_table[req_id][block_table_idx]
slot_id = physical_block_id * block_size + (new_token_idx % block_size)
```

**核心 indirection**：物理 block id 来自 block_table（请求自己的逻辑→物理映射），slot id 是"扁平化的物理坐标"，让 kernel 一次 store 完事。

---

**3. H100 + Llama 默认后端选择路径需要哪些 `supports_*` 通过？**

源码：`vllm/v1/attention/selector.py` 的 `get_attn_backend()`。决策大致：

```
1. 检查 platform：CUDA / ROCm / TPU / ...      → 进入 CUDA 分支
2. 检查模型类型：MLA? GQA/MHA?                  → 普通 Llama 是 GQA，走标准 path
3. 检查 GPU 支持 FlashAttention v2/v3:
     - SM 8.0+ (A100/H100/Hopper)             ✓
     - dtype = bfloat16 / float16             ✓
     - head_dim ∈ {32,64,128,256}             ✓
     - 不需要 sliding window 特殊版本         ✓
4. 检查 use_v3:
     - SM 90 (H100) + FlashAttn v3 wheel 装了 → use v3
     - 否则降级 v2
5. 返回 FlashAttentionBackend
```

如果某一步 fail → 降级到 FlashInfer，再降到 Triton paged_attention，再到自家 paged_attention CUDA kernel。

**H100 + 普通 Llama-3 = FlashAttention v3**（最优路径）。

加分点：FlashInfer 在某些场景（特别是 GQA + 极大 batch）比 FlashAttention v3 还快；可通过 `VLLM_ATTENTION_BACKEND=FLASHINFER` 强制覆盖默认选择。

---

**4. cascade attention 触发条件 + `get_num_common_prefix_blocks` 调用时机？**

**cascade attention** 的核心 idea：当 batch 内多个请求共享一段长 prefix（如同一 system prompt），把 attention 拆成两段：

1. 共享段：所有 query 一起跟同一份 KV 做 attention（cache miss 一次，所有 query 重复利用）
2. 独立段：每个请求各自跟自己后段的 KV

**触发条件**（典型）：

- batch 内请求数 ≥ 阈值（一般 ≥ 2）
- 共享 prefix 长度 ≥ 阈值（一般 ≥ 256 token）
- attention backend 支持 cascade（FlashInfer 已支持，FlashAttn 在加）

**`get_num_common_prefix_blocks` 调用时机**：scheduler 每步 `schedule()` 决策完 batch 后、构造 SchedulerOutput 前。它扫描本步 running 的所有请求的 block_table 前缀，找最长公共前缀的 block 数，写入 SchedulerOutput.num_common_prefix_blocks。Attention metadata builder 据此决定要不要走 cascade 模式。

源码：`vllm/v1/core/sched/scheduler.py::Scheduler.schedule()` 末尾、构造 SchedulerOutput 时。

---

**5. DeepSeek-V3 KV cache shape 还是 `[2, ...]` 吗？**

**不是**。DeepSeek-V3 用 **MLA（Multi-head Latent Attention）**，KV 是通过低秩投影从 latent 维度算出来的，**只存一个 latent tensor 而不是 K/V 两个**。

MLA KV cache shape：`[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]`（注意没有 `[2, ...]` 的二元维度）

具体数字（DeepSeek-V3）：

- `kv_lora_rank = 512`（latent 维度）
- `qk_rope_head_dim = 64`（位置编码部分单独存）
- 单 token KV ≈ (512 + 64) × 2 byte = **1152 字节**

对比标准 GQA（Llama-3-70B 单 token 4 KB）：MLA 大约 **省 4×**。论文报 8× 来自 MLA 形状 + 配套的 W_dkv 拆 rope/nope 优化。

→ vLLM 用专门的 `MLA` 系列 attention backend（`vllm/v1/attention/backends/mla.py` 等），KVCacheSpec 也是单独类（`MLAKVCacheSpec`）。KV manager 通过 `KVCacheSpec` 抽象屏蔽形状差异——这就是 V1 解耦 KVCacheManager 的回报。

## 下一步

- 下一节：[`06-cuda-kernels.md`](06-cuda-kernels.md)（attention backend 之下，vLLM 自带的 paged attention CUDA kernel 是什么样的）
- 想看源码：`vllm/v1/attention/backends/flash_attn.py`、`vllm/v1/attention/backends/mla/flashmla.py`、`vllm/v1/attention/selector.py`
- 想动手：[`07-hands-on/04-profiling-and-debugging.md`](../07-hands-on/04-profiling-and-debugging.md)（切换 `--attention-backend` 看 forward 时间变化）
- 想从生产视角理解：[`08-production-deployment/04-autoscaling-and-capacity.md`](../08-production-deployment/04-autoscaling-and-capacity.md)（不同 backend 在不同 batch size 下吞吐差异，对容量规划的影响）

