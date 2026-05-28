# 04. vLLM 项目文件架构地图

> **谁该读这一篇？** 第一次 clone 完 vLLM 仓库被 1700+ Python 文件 + 237 个 CUDA/C++ 文件吓退的同学；后续每一篇源码走读前，需要先知道"我要找的东西在哪一级目录"的读者。
>
> **前置阅读：** [`03-v0-vs-v1.md`](03-v0-vs-v1.md)（知道为什么本节几乎只讲 `vllm/v1/` 而不讲 `vllm/engine/`）。
>
> **耗时：** 约 25 分钟（建议第一遍**通读 §1-3 + §21**，找东西时再回来查具体子目录）。
>
> **学完能：**
> 1. 看到任何一个 `vllm/...` 路径能立刻说出"它属于 V1 引擎 / 模型层 / 多模态 / 量化 / 量化算子 / 配置 / 入口"等大类。
> 2. 拿到"我要改 X"的需求，能直接在脑子里翻到 §21 的速查表，定位首要修改文件。
> 3. 区分 `vllm/v1/`、`vllm/model_executor/`、`vllm/csrc/`、`vllm/distributed/` 四大重量级目录各自的边界与上下游。
> 4. 不会再把 `csrc/attention/` 与 `vllm/v1/attention/backends/` 搞混（前者是 CUDA kernel，后者是 Python backend 接口）。

vLLM 仓库有 **1700+ 个 Python 文件 + 237 个 CUDA/C++ 文件 + 292 个模型实现**。第一次打开仓库容易迷路。本节给你一张"地图"——每个目录、每个关键文件干什么。找东西时回头查这一篇就行。

---

## 1. 仓库顶层（`/Users/zjw/Documents/github-project/vllm/`）

```
vllm/
├── vllm/              ← Python 主包（重点）
├── csrc/              ← C++/CUDA 内核
├── benchmarks/        ← 性能基准
├── tests/             ← 单元测试 + 集成测试
├── examples/          ← 使用示例（offline / online / 多模态 / tool use）
├── docs/              ← 官方文档源（mkdocs）
├── docker/            ← Dockerfile（CUDA / ROCm / TPU / CPU）
├── cmake/             ← C++ 编译配置
├── requirements/      ← 各平台 pip 依赖
├── scripts/           ← 开发辅助脚本
├── tools/             ← 内部工具
├── third_party/       ← 第三方代码
├── setup.py           ← Python 包安装（含 C++ 编译）
├── pyproject.toml     ← Python 项目配置
├── CMakeLists.txt     ← 整体 CMake 入口
├── README.md / AGENTS.md / CLAUDE.md / CONTRIBUTING.md ...
```

---

## 2. `vllm/` 包：顶层文件

打开 `vllm/__init__.py` 暴露的就是这些公共类。重要的顶层模块：

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 顶层导出 LLM、SamplingParams、PoolingParams 等 |
| `version.py` | 版本号 |
| `envs.py` | **所有环境变量**注册表（`VLLM_*`、NCCL、CUDA） |
| `env_override.py` | 平台特定环境覆盖 |
| `sampling_params.py` | `SamplingParams`、`StructuredOutputsParams`（OpenAI 协议入口） |
| `pooling_params.py` | `PoolingParams`（embedding/reranker 协议入口） |
| `outputs.py` | `RequestOutput`、`CompletionOutput` 等用户可见数据类 |
| `sequence.py` | 旧版 `Sequence`（V0 残留，V1 不用） |
| `tasks.py` | 任务类型枚举（generate / embed / classify / score / rerank） |
| `scalar_type.py` | 量化用的标量类型枚举（INT4 / FP8 / NF4 ...） |
| `_custom_ops.py` | 把 C++ kernel 暴露给 Python（`torch.ops.vllm.*`） |
| `_aiter_ops.py` / `_xpu_ops.py` / `_tilelang_ops.py` | 对应 ROCm AITER / XPU / TileLang 后端 |
| `forward_context.py` | **运行时上下文**（attention metadata 注入用） |
| `logits_process.py` | 用户自定义 LogitsProcessor 框架 |
| `logprobs.py` | logprobs 公共数据类型 |
| `beam_search.py` | Beam search（V1 已弃用） |
| `connections.py` | HTTP 会话池工具 |
| `logger.py` | 日志初始化 |
| `model_inspection.py` | 模型架构识别（HF config → vLLM model 类） |
| `exceptions.py` | 自定义异常 |
| `collect_env.py` | `vllm collect-env` 命令的实现 |
| `scripts.py` | CLI 入口（`vllm serve` / `vllm bench`） |

---

## 3. `vllm/v1/`：V1 引擎（重点中的重点）

```
v1/
├── __init__.py
├── cudagraph_dispatcher.py     ← runtime size → captured graph 路由
├── kv_cache_interface.py        ← KVCacheSpec（多 KV 类型抽象）
├── outputs.py                   ← EngineCoreOutput / ModelRunnerOutput
├── request.py                   ← Request 类（状态机：WAITING/RUNNING/...）
├── serial_utils.py              ← ZMQ msgpack 编码
├── utils.py
├── attention/
├── core/
├── engine/
├── executor/
├── kv_offload/
├── metrics/
├── pool/
├── sample/
├── simple_kv_offload/
├── spec_decode/
├── structured_output/
└── worker/
```

### 3.1 `v1/engine/`：引擎门面与异步入口

| 文件 | 作用 |
| --- | --- |
| `llm_engine.py` | 同步 `LLMEngine`（offline 入口，门面） |
| `async_llm.py` | `AsyncLLM`（在线服务用，asyncio queue） |
| `core.py` | **`EngineCore`** —— 真正的调度引擎进程主体 |
| `core_client.py` | `EngineCoreClient` —— ZMQ 客户端 |
| `coordinator.py` | DP / EP 跨实例协调 |
| `detokenizer.py` | token_id → 文本（streaming） |
| `input_processor.py` | prompt 预处理 + 多模态 input 准备 |
| `output_processor.py` | 把 EngineCoreOutput → 用户可见 RequestOutput |
| `parallel_sampling.py` | n>1 时的并行采样 |
| `tensor_ipc.py` | 跨进程传 tensor |
| `logprobs.py` | logprobs 公共逻辑 |
| `exceptions.py` |  |
| `utils.py` |  |

### 3.2 `v1/core/`：调度与 KV 管理

| 路径 | 作用 |
| --- | --- |
| `core/sched/scheduler.py` | **核心** Scheduler.schedule()（2300+ 行） |
| `core/sched/async_scheduler.py` | AsyncScheduler（schedule 与 forward overlap） |
| `core/sched/interface.py` | Scheduler 抽象基类 |
| `core/sched/output.py` | `SchedulerOutput` 数据类 |
| `core/sched/request_queue.py` | 优先级队列 / FCFS 队列 |
| `core/sched/utils.py` |  |
| `core/block_pool.py` | `BlockPool` —— 物理 KV block 中央仓库 |
| `core/kv_cache_manager.py` | `KVCacheManager` —— Scheduler 看到的接口 |
| `core/kv_cache_coordinator.py` | 多 KV 类型协调器（普通/MLA/Mamba 共存） |
| `core/kv_cache_metrics.py` | KV 指标采集 |
| `core/kv_cache_utils.py` | hash 计算、`KVCacheBlock` |
| `core/single_type_kv_cache_manager.py` | 单种 KV 类型的 manager 基类（Full/Sliding/Mamba） |
| `core/encoder_cache_manager.py` | 多模态 encoder 输出缓存 |

### 3.3 `v1/worker/`：Worker 进程与模型执行

| 文件 | 作用 |
| --- | --- |
| `worker_base.py` | Worker 抽象基类（所有平台共用接口） |
| `gpu_worker.py` | **GPU Worker**（init + load_model + execute_model） |
| `gpu_model_runner.py` | **`GPUModelRunner`**（3000+ 行，每步 forward 本体） |
| `gpu_input_batch.py` | 持久化 `InputBatch`（V1 性能关键） |
| `cpu_worker.py` / `cpu_model_runner.py` | CPU 后端 |
| `xpu_worker.py` / `xpu_model_runner.py` | Intel XPU 后端 |
| `gpu_ubatch_wrapper.py` | micro-batching 包装 |
| `block_table.py` | GPU 上的 block_table tensor 维护 |
| `kv_connector_model_runner_mixin.py` | KV connector 集成 mixin |
| `ec_connector_model_runner_mixin.py` | encoder cache connector mixin |
| `lora_model_runner_mixin.py` | LoRA 集成 mixin |
| `dp_utils.py` / `cp_utils.py` | DP / Context-Parallel 工具 |
| `ubatching.py` / `ubatch_utils.py` | micro-batching |
| `encoder_cudagraph.py` | encoder 路径的 CUDA Graph |
| `mamba_utils.py` | Mamba state 处理 |
| `workspace.py` | 显存 workspace 管理 |
| `gpu/` | GPU-specific 子模块（mm / model_states / pool / sample / spec_decode） |

### 3.4 `v1/executor/`：Worker 编排（每种部署一份）

| 文件 | 作用 |
| --- | --- |
| `abstract.py` | Executor 接口 |
| `uniproc_executor.py` | 单进程单卡 |
| `multiproc_executor.py` | 单机多卡（multiprocessing + shared memory） |
| `ray_executor.py` / `ray_executor_v2.py` | 多机多卡 Ray |
| `ray_utils.py` / `ray_env_utils.py` | Ray 辅助 |

### 3.5 `v1/attention/`：Attention 后端

| 路径 | 作用 |
| --- | --- |
| `backend.py` | 后端基类（AttentionBackend / Impl / MetadataBuilder） |
| `backends/registry.py` | 所有后端枚举与按需 import |
| `backends/flash_attn.py` | **FlashAttention v2/v3**（默认） |
| `backends/flashinfer.py` | FlashInfer（H100 decode 小 batch 优选） |
| `backends/triton_attn.py` | Triton fallback |
| `backends/rocm_attn.py` / `rocm_aiter_*.py` | AMD ROCm |
| `backends/cpu_attn.py` | CPU 后端 |
| `backends/flex_attention.py` | torch FlexAttention |
| `backends/turboquant_attn.py` | 量化 attention |
| `backends/{mamba1,mamba2,mamba}_attn.py` | Mamba SSM |
| `backends/gdn_attn.py` / `linear_attn.py` / `short_conv_attn.py` | 线性注意力变体 |
| `backends/mla/` | **MLA 子目录**（DeepSeek） |
|  &nbsp;&nbsp;`flashmla.py` / `flashattn_mla.py` / `cutlass_mla.py` | NVIDIA MLA kernel 套件 |
|  &nbsp;&nbsp;`flashinfer_mla.py` / `triton_mla.py` | 备选 |
|  &nbsp;&nbsp;`rocm_aiter_mla*.py` / `xpu_mla_sparse.py` | 其他硬件 |
| `selector.py` | 自动选择 backend |
| `ops/` | Triton attention ops |

### 3.6 `v1/sample/`：采样

| 文件 | 作用 |
| --- | --- |
| `sampler.py` | **`Sampler`**（forward 入口、温度、greedy/random 合并） |
| `rejection_sampler.py` | spec decode 接受/拒绝采样 |
| `metadata.py` | `SamplingMetadata`（打包整 batch） |
| `tpu_sampler.py` | TPU 版本 |
| `ops/topk_topp_sampler.py` | top-k / top-p 入口 |
| `ops/topk_topp_triton.py` | Triton kernel |
| `ops/penalties.py` | repetition / presence / frequency penalty |
| `ops/bad_words.py` | bad_words mask |
| `ops/logprobs.py` | logprobs 工具（fused log_softmax + gather） |
| `logits_processor/` | 自定义 logits processor 框架 |

### 3.7 `v1/structured_output/`：JSON / Grammar 约束

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | **`StructuredOutputManager`**（engine-level 调度） |
| `backend_types.py` | 基类（`StructuredOutputBackend`、`StructuredOutputGrammar`） |
| `backend_xgrammar.py` | xgrammar（默认） |
| `backend_guidance.py` | Microsoft llguidance |
| `backend_outlines.py` | outlines |
| `backend_lm_format_enforcer.py` | lm-format-enforcer |
| `request.py` | `StructuredOutputRequest`（per-request 状态） |
| `utils.py` |  |

### 3.8 `v1/spec_decode/`：投机解码

| 文件 | 作用 |
| --- | --- |
| `llm_base_proposer.py` | LLM-based draft proposer 基类 |
| `ngram_proposer_gpu.py` | n-gram 提议（GPU） |
| `eagle.py` | EAGLE proposer（vLLM 主力 spec method） |
| `medusa.py` | Medusa（经典方法，被 EAGLE 替代） |
| `metadata.py` | 共享数据结构 |
| `utils.py` | rejection sampler 数学辅助 |

### 3.9 `v1/pool/`：embedding/pooling 模型用

| 文件 | 作用 |
| --- | --- |
| `metadata.py` | PoolingMetadata |
| `late_interaction.py` | ColBERT-style late interaction |

### 3.10 `v1/kv_offload/`、`v1/simple_kv_offload/`：KV 卸载

KV 从 GPU HBM offload 到 CPU 内存 / NVMe 的实现：

- `kv_offload/cpu/` —— CPU memory offload
- `kv_offload/cpu/policies/` —— LRU / LFU 等
- `kv_offload/tiering/` —— 多层（GPU → CPU → 远端）
- `kv_offload/worker/` —— Worker 侧 hooks
- `simple_kv_offload/` —— 简单实现（学习用）

### 3.11 `v1/metrics/`：Prometheus / 日志

| 文件 | 作用 |
| --- | --- |
| `loggers.py` | stat logger（周期性打印 throughput / KV usage） |
| `prometheus.py` | `vllm:*` Prometheus 暴露 |
| `stats.py` | 统计数据类（PrefixCacheStats 等） |

---

## 4. `vllm/model_executor/`：模型与算子层

```
model_executor/
├── custom_op.py            ← @CustomOp 装饰器（注册到 torch.library）
├── parameter.py            ← 参数加载工具
├── utils.py
├── kernels/                ← 调度到不同 kernel 后端
├── layers/                 ← 各种 layer 实现
├── model_loader/           ← 模型权重加载
├── models/                 ← **292 个模型实现**
├── offloader/              ← 权重 offload 到 CPU
└── warmup/                 ← 启动 warmup
```

### 4.1 `model_executor/layers/`：算子层

| 路径 | 作用 |
| --- | --- |
| `linear.py` | `ColumnParallelLinear` / `RowParallelLinear` / `ReplicatedLinear` |
| `activation.py` | SiLU / GeLU / SwiGLU |
| `layernorm.py` | RMSNorm / fused versions |
| `rotary_embedding/` | RoPE 各种变种（YaRN / NTK / Linear / decoupled） |
| `vocab_parallel_embedding.py` | 词表并行 embedding |
| `logits_processor.py` | LM head + logits processor |
| `attention/` | Attention layer 上层包装（不是 backend） |
| `attention_layer_base.py` | Attention 基类 |
| `mla.py` | MLA 包装层（DeepSeek） |
| `mhc.py` / `kda.py` | Multi-head conv / Kernel decomposition |
| `mamba/` | Mamba mixer |
| `fla/` | Flash Linear Attention |
| `lightning_attn.py` / `linear_attn.py` | 线性注意力 |
| `sparse_attn_indexer.py` | sparse attention |
| `deepseek_compressor.py` / `deepseek_v4_attention.py` | DeepSeek 专属 |
| `conv.py` | 1D / 2D conv（Mamba 等用） |
| `resampler.py` | Perceiver-style resampler（多模态） |
| `batch_invariant.py` | Batch-invariant 模式 |
| `fused_moe/` | **MoE** 实现 |
| `pooler/` | Pooling（embedding 模型用） |
| `quantization/` | 量化集成 |

### 4.2 `model_executor/layers/fused_moe/`：MoE 层

| 文件 | 作用 |
| --- | --- |
| `layer.py` | `FusedMoE` 主层 |
| `fused_moe.py` | 核心 grouped GEMM 调度 |
| `fused_moe_method_base.py` | MoE method 基类 |
| `fused_moe_modular_method.py` | 模块化实现 |
| `moe_align_block_size.py` | token → expert 对齐 |
| `experts/` | 每种量化的 expert 实现（FP8 / AWQ / GPTQ） |
| `all2all_utils.py` | EP 用 AllToAll |
| `expert_map_manager.py` | EP 下 expert ↔ rank 映射 |
| `deep_gemm_utils.py` | H100 deep GEMM |
| `cpu_fused_moe.py` | CPU MoE |
| `activation.py` |  |
| `modular_kernel.py` |  |
| `config.py` / `configs/` |  |

### 4.3 `model_executor/layers/quantization/`：量化

| 文件 | 作用 |
| --- | --- |
| `base_config.py` | QuantizationConfig 基类 |
| `schema.py` |  |
| `kv_cache.py` | KV cache 量化（FP8 / INT8） |
| `fp8.py` | FP8 (E4M3 / E5M2) |
| `fp_quant.py` |  |
| `bitsandbytes.py` | NF4 / FP4（QLoRA 同款） |
| `awq.py` / `awq_marlin.py` / `awq_triton.py` | AWQ INT4 |
| `auto_gptq.py` | GPTQ |
| `gguf.py` | GGUF（llama.cpp 兼容） |
| `mxfp4.py` | Microscaling FP4 |
| `fbgemm_fp8.py` | Meta FBGEMM |
| `modelopt.py` | NVIDIA Model Optimizer |
| `torchao.py` | TorchAO |
| `experts_int8.py` / `moe_wna16.py` | MoE 专用 |
| `cpu_wna16.py` | CPU 量化 |
| `inc.py` | Intel Neural Compressor |
| `humming.py` |  |
| `compressed_tensors/` | Neural Magic 统一格式 |
| `online/` | Online quantization |
| `quark/` | Quark format |
| `turboquant/` | TurboQuant |
| `utils/` |  |

### 4.4 `model_executor/model_loader/`：权重加载

| 文件 | 作用 |
| --- | --- |
| `base_loader.py` | Loader 接口 |
| `default_loader.py` | HF safetensors（默认） |
| `gguf_loader.py` | GGUF |
| `bitsandbytes_loader.py` | bnb |
| `tensorizer.py` / `tensorizer_loader.py` | Tensorizer（CoreWeave 快速加载） |
| `runai_streamer_loader.py` | RunAI Model Streamer（S3 流式） |
| `sharded_state_loader.py` | 分片格式（训练 checkpoint） |
| `dummy_loader.py` | 不加载，仅 benchmark |
| `weight_utils.py` | 权重转换工具 |
| `ep_weight_filter.py` | EP 下按 expert 过滤权重 |
| `utils.py` |  |
| `reload/` | 在线重载权重 |

### 4.5 `model_executor/models/`：模型实现（292 个）

按家族分类（这里只列有代表性的）：

| 家族 | 文件举例 |
| --- | --- |
| Llama 系 | `llama.py` / `llama4.py` / `tinyllama.py` |
| Qwen 系 | `qwen2.py` / `qwen3.py` / `qwen2_vl.py` / `qwen2_audio.py` |
| DeepSeek 系 | `deepseek_v2.py` / `deepseek_v3.py` / `deepseek_v4.py`（含 MLA + MoE + MTP） |
| Mistral 系 | `mistral.py` / `mixtral.py` / `mistral3.py`（含 Mixtral MoE） |
| GLM / ChatGLM | `chatglm.py` |
| Phi 系 | `phi.py` / `phi3.py` / `phi3v.py` / `phi4.py` |
| Gemma | `gemma.py` / `gemma2.py` / `gemma3.py` / `gemma4.py` |
| BERT / embedding | `bert.py` / `bge_m3.py` / `jina_embeddings_v3.py` |
| Mamba 系 | `mamba.py` / `mamba2.py` / `jamba.py` / `zamba2.py` |
| 多模态 | `qwen2_vl.py` / `llava.py` / `phi3v.py` / `internvl.py` / `aria.py` / `chameleon.py` |
| Whisper | `whisper.py`（encoder-decoder） |
| 工具 | `adapters.py` / `__init__.py`（家族注册） |

每个模型文件结构相似：

```python
class XxxAttention(nn.Module): ...      # 用 vLLM Attention + 量化 Linear
class XxxMLP(nn.Module): ...
class XxxDecoderLayer(nn.Module): ...
class XxxModel(nn.Module): ...           # stack of DecoderLayer
class XxxForCausalLM(nn.Module): ...     # add LM head + forward 入口
```

---

## 5. `vllm/multimodal/`：多模态输入

| 文件 | 作用 |
| --- | --- |
| `inputs.py` | `MultiModalKwargs` / `PlaceholderRange` / `MultiModalFieldElem`（1015 行） |
| `registry.py` | `MultiModalRegistry`（模型 ↔ processor 绑定） |
| `image.py` | image 输入解析（PIL / np / bytes / URL） |
| `video.py` | video 帧抽样 / 时序重采样（1055 行） |
| `audio.py` | waveform 加载与切片 |
| `hasher.py` | mm 内容哈希（影响 prefix caching） |
| `cache.py` | 输入侧 cache |
| `encoder_budget.py` | encoder 算力 budget |
| `evs.py` |  |
| `parse.py` | OpenAI message → mm_data 解析 |
| `media/` | 子模块 |
| `processing/` | 每模型一份 Processor |
| `utils.py` |  |

---

## 6. `vllm/lora/`：LoRA 适配器

| 路径 | 作用 |
| --- | --- |
| `model_manager.py` | `LoRAModelManager` / `LRUCacheLoRAModelManager`（1057 行） |
| `worker_manager.py` | Worker 侧 LoRA 管理（add/remove/pin） |
| `lora_model.py` | LoRA 权重容器 |
| `lora_weights.py` | 权重 dataclass |
| `peft_helper.py` | PEFT 格式适配 |
| `request.py` | `LoRARequest` |
| `resolver.py` | 解析 lora_path（local / HF / S3） |
| `utils.py` |  |
| `layers/` | 各种 Linear 的 LoRA 包装（column/row/replicated/MoE） |
| `punica_wrapper/` | Punica kernel 入口（base/gpu/cpu/tpu） |
| `ops/` | Triton bgmv kernels |

---

## 7. `vllm/distributed/`：分布式

| 路径 | 作用 |
| --- | --- |
| `parallel_state.py` | **核心** TP / PP / EP 进程组初始化 |
| `communication_op.py` | AllReduce / AllGather wrapper |
| `device_communicators/` | NCCL / HPU / TPU / CPU 通信后端 |
| `kv_transfer/` | KV connector（NIXL / LMCache / Mooncake） |
| `kv_events.py` | KV cache 事件 |
| `weight_transfer/` | 跨进程 weight 传输（训练 + 推理同卡复用） |
| `ec_transfer/` | encoder cache 传输 |
| `eplb/` | Expert Parallel Load Balancer |
| `elastic_ep/` | Elastic EP（动态 expert 扩缩） |
| `nixl_utils.py` | NIXL RDMA 工具 |
| `stateless_coordinator.py` | 无状态 DP 协调器 |
| `utils.py` |  |

---

## 8. `vllm/entrypoints/`：API 入口

| 路径 | 作用 |
| --- | --- |
| `llm.py` | `LLM` 类（offline 入口） |
| `api_server.py` | 通用 API server |
| `launcher.py` | `vllm serve` 启动器 |
| `grpc_server.py` | gRPC 入口 |
| `cli/` | 命令行入口（vllm serve / bench / chat / ...） |
| `openai/` | **OpenAI 兼容**（绝大多数生产路径） |
|  &nbsp;&nbsp;`api_server.py` | FastAPI app |
|  &nbsp;&nbsp;`chat_completion/` / `completion/` / `responses/` | 各 endpoint |
|  &nbsp;&nbsp;`generative_scoring/` | 生成式打分 |
|  &nbsp;&nbsp;`models/` | /v1/models endpoint |
|  &nbsp;&nbsp;`engine/` | 与 EngineCore 交互 |
|  &nbsp;&nbsp;`cli_args.py` | 启动参数 |
|  &nbsp;&nbsp;`run_batch.py` | 批量推理 |
|  &nbsp;&nbsp;`server_utils.py` | 工具 |
|  &nbsp;&nbsp;`parser/` | request 解析 |
|  &nbsp;&nbsp;`orca_metrics.py` | ORCA 协议（负载报告给 LB） |
| `anthropic/` | Anthropic 协议兼容 |
| `mcp/` | MCP（Model Context Protocol）兼容 |
| `pooling/` | Embedding / rerank endpoints |
| `sagemaker/` | SageMaker 适配 |
| `speech_to_text/` | Whisper 风格 ASR |
| `serve/` | Service 框架 |
| `chat_utils.py` | chat template 应用 |
| `logger.py` | 请求日志 |
| `ssl.py` | TLS 设置 |
| `utils.py` / `constants.py` |  |

---

## 9. `vllm/compilation/`：torch.compile 集成

| 文件 | 作用 |
| --- | --- |
| `backends.py` | `VllmBackend` / `CompilerManager` / `split_graph`（1331 行） |
| `compiler_interface.py` | `CompilerInterface` / `InductorAdaptor`（782 行） |
| `cuda_graph.py` | CUDA Graph 包装 |
| `caching.py` | 编译缓存 |
| `codegen.py` | 代码生成辅助 |
| `decorators.py` | `@CustomOp` 注册自定义 op |
| `partition_rules.py` | 切图规则 |
| `piecewise_backend.py` | piecewise 编译 |
| `wrapper.py` | 顶层 wrapper |
| `monitor.py` | 编译监控 |
| `counter.py` |  |
| `base_static_graph.py` |  |
| `passes/` | 自定义 pass |
|  &nbsp;&nbsp;`vllm_inductor_pass.py` | 主 pass 集合 |
|  &nbsp;&nbsp;`pass_manager.py` | pass 串联 |
|  &nbsp;&nbsp;`fusion/` | AllReduce / activation / attention fusion |
|  &nbsp;&nbsp;`fx_utils.py` / `inductor_pass.py` |  |
|  &nbsp;&nbsp;`utility/` |  |

---

## 10. `vllm/config/`：所有配置类

每种 config 是一个 dataclass。用户通过 EngineArgs 设置后，组合成 `VllmConfig`：

| 文件 | 内容 |
| --- | --- |
| `vllm.py` | `VllmConfig`（聚合所有 sub-config） |
| `model.py` | `ModelConfig`（hf_config、dtype、max_model_len） |
| `model_arch.py` | 模型架构信息 |
| `cache.py` | `CacheConfig`（block_size、gpu_memory_utilization、KV dtype） |
| `parallel.py` | `ParallelConfig`（TP/PP/EP/DP） |
| `scheduler.py` | `SchedulerConfig`（max_num_seqs、max_num_batched_tokens、policy） |
| `device.py` | `DeviceConfig`（cuda/cpu/tpu/xpu） |
| `load.py` | `LoadConfig`（loader 类型、download_dir） |
| `lora.py` | `LoRAConfig`（max_loras、max_lora_rank） |
| `compilation.py` | `CompilationConfig`（level、CUDAGraph 配置） |
| `quantization.py` | `QuantizationConfig` 工厂 |
| `attention.py` | `AttentionConfig` |
| `kernel.py` | `KernelConfig` |
| `multimodal.py` | `MultimodalConfig` |
| `speculative.py` | `SpeculativeConfig` |
| `structured_outputs.py` | `StructuredOutputsConfig` |
| `mamba.py` | Mamba 专属 |
| `pooler.py` | `PoolerConfig`（embedding 模型） |
| `kv_transfer.py` / `kv_events.py` / `ec_transfer.py` | KV / EC connector 配置 |
| `weight_transfer.py` | 权重传输 |
| `observability.py` | Prom / OTel 配置 |
| `profiler.py` | Profile 配置 |
| `reasoning.py` | Reasoning parser 配置 |
| `speech_to_text.py` | STT 配置 |
| `offload.py` | KV offload 配置 |
| `utils.py` |  |

---

## 11. `vllm/profiler/`、`vllm/tool_parsers/`、`vllm/reasoning/`

### profiler/
- `layerwise_profile.py` —— 层级 profiler（每 transformer 层耗时）
- `wrapper.py` —— profile 包装
- `utils.py`

### tool_parsers/
- `abstract_tool_parser.py` —— 接口
- 每模型一个：`deepseekv3_tool_parser.py` / `gemma4_tool_parser.py` / `glm4_moe_tool_parser.py` / `granite4_tool_parser.py` / ...
- 把 LLM 输出的 function call 字符串解析成结构化对象

### reasoning/
- DeepSeek-R1 / o1 风格的 `<think>` 段解析
- 每模型一份 parser

---

## 12. `vllm/platforms/`：硬件抽象

| 文件 | 作用 |
| --- | --- |
| `interface.py` | `Platform` 基类 |
| `cuda.py` | NVIDIA GPU |
| `rocm.py` | AMD ROCm |
| `tpu.py` | Google TPU |
| `xpu.py` | Intel XPU |
| `cpu.py` | CPU |
| `zen_cpu.py` | AMD Zen CPU |

`Platform.get_attn_backend_cls()`、`get_communicator_cls()` 等方法让上层无关硬件。

---

## 13. `vllm/transformers_utils/`：HF 生态适配

| 路径 | 作用 |
| --- | --- |
| `tokenizer.py` | HF tokenizer 包装 |
| `processor.py` | HF processor |
| `config.py` | 从 HF config 推断模型 |
| `chat_templates/` | 各模型 chat template |
| `configs/` | 各模型 config 类 |
| `processors/` | 各模型 processor |
| `gguf_utils.py` | GGUF 元数据 |
| `runai_utils.py` / `s3_utils.py` | 远端权重加载 |
| `dynamic_module.py` | trust_remote_code 支持 |
| `repo_utils.py` |  |
| `model_arch_config_convertor.py` | 模型 config 转换 |

---

## 14. `vllm/inputs/`：请求输入处理

| 文件 | 作用 |
| --- | --- |
| `llm.py` | LLM offline 入口的输入处理 |
| `engine.py` | Engine 层输入打包 |
| `preprocess.py` | tokenize、chat template apply、mm preprocess |

---

## 15. `vllm/utils/`：工具集

最常用的：

| 文件 | 作用 |
| --- | --- |
| `cache.py` | LRUCache 等 |
| `hashing.py` | 哈希工具 |
| `mem_utils.py` / `mem_constants.py` | 显存工具 |
| `nccl.py` | NCCL 辅助 |
| `flashinfer.py` | FlashInfer 包装 |
| `deep_gemm.py` | DeepGEMM 包装 |
| `numa_utils.py` / `numa_wrapper.sh` | NUMA 亲和 |
| `nvtx_pytorch_hooks.py` | NVTX 标记 |
| `multi_stream_utils.py` | CUDA 多 stream |
| `gc_utils.py` | GC 控制 |
| `import_utils.py` / `argparse_utils.py` / `func_utils.py` | 通用 |
| `network_utils.py` | 端口探测等 |
| `cpu_resource_utils.py` / `cpu_triton_utils.py` | CPU 相关 |
| `mistral.py` | Mistral 专属 |
| `counter.py` / `async_utils.py` / `collection_utils.py` / `jsontree.py` | 杂项 |
| `math_utils.py` |  |

---

## 16. `csrc/`：C++ / CUDA 内核

```
csrc/
├── torch_bindings.cpp                     ← 主 pybind 入口
├── ops.h                                  ← 全部 op 声明
├── cuda_compat.h / cuda_utils.h / cuda_vec_utils.cuh / dispatch_utils.h ...
├── cumem_allocator.{cpp,h}                ← cumem 内存池
├── async_util.cuh / spinloop.cpp          ← 异步工具
├── launch_bounds_utils.h
│
├── activation_kernels.cu                  ← SiLU / GeLU / SwiGLU
├── layernorm_kernels.cu                   ← RMSNorm
├── layernorm_quant_kernels.cu             ← RMSNorm + 量化融合
├── pos_encoding_kernels.cu                ← RoPE
├── cache_kernels.cu / cache_kernels_fused.cu  ← KV reshape / copy
├── nvfp4_kv_cache_kernels.cu              ← NVFP4 KV
├── sampler.cu                             ← top-k / top-p sampling
├── topk.cu                                ← topk 辅助
├── custom_all_reduce.{cu,cuh}             ← 自定义 AllReduce
├── custom_all_reduce_test.cu
├── custom_quickreduce.cu                  ← quickreduce
├── cuda_utils_kernels.cu
├── cuda_view.cu
├── fused_qknorm_rope_kernel.cu            ← QK-norm + RoPE 融合
├── fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu  ← DSv4 专融合
├── minimax_reduce_rms_kernel.{cu,h}
│
├── attention/                             ← PagedAttention v1/v2 + 辅助
│   ├── paged_attention_v1.cu
│   ├── paged_attention_v2.cu
│   ├── merge_attn_states.cu               ← v2 的归并阶段
│   ├── attention_kernels.cuh              ← 核心 kernel 模板
│   ├── attention_generic.cuh / attention_utils.cuh
│   ├── attention_dtypes.h
│   ├── dtype_{bfloat16,float16,float32,fp8}.cuh
│   └── vertical_slash_index.cu             ← Sparse attention 索引
│
├── moe/                                   ← MoE 路由 + 分组 GEMM
│   ├── moe_align_sum_kernels.cu
│   ├── grouped_topk_kernels.cu
│   ├── moe_permute_unpermute_op.cu
│   ├── moe_wna16.cu
│   ├── dsv3_router_gemm_*.cu              ← DeepSeek-V3 路由 GEMM
│   ├── dsv4_norm_router_gemm_*.{h,cu}     ← V4
│   ├── dynamic_4bit_int_moe_cpu.cpp
│   ├── marlin_moe_wna16/                  ← Marlin × MoE
│   └── moeTopKFuncs.cuh / moe_ops.h
│
├── quantization/                          ← 量化 GEMM
│   ├── gptq/                              ← GPTQ
│   ├── awq/（不在这里）— AWQ 在上层 layers/quantization/awq*
│   ├── machete/                           ← H100+ 高速 mixed precision GEMM
│   ├── marlin/                            ← INT4 W × FP16 A 最快 GEMM
│   ├── w8a8/                              ← W8A8 量化
│   ├── fused_kernels/                     ← 融合量化 op
│   ├── gguf/                              ← GGUF kernel
│   ├── activation_kernels.cu
│   └── utils.cuh
│
├── core/                                  ← 公共 utility
├── cutlass_extensions/                    ← CUTLASS 自定义模板
├── libtorch_stable/                       ← 稳定 ABI 封装
│
├── concat_mla_q.cuh                       ← MLA Q 拼接
├── persistent_topk.cuh                    ← persistent kernel
├── type_convert.cuh
├── cub_helpers.h
│
├── mamba/                                 ← Mamba SSM kernel
├── rocm/                                  ← AMD ROCm 移植
├── quickreduce/                           ← QuickReduce 通信
└── cpu/                                   ← CPU 后端（多种向量化）
```

---

## 17. `benchmarks/`：性能基准

| 文件 | 用途 |
| --- | --- |
| `benchmark_throughput.py` | 离线吞吐 |
| `benchmark_latency.py` | 单请求延迟 |
| `benchmark_serving.py` | 在线服务（OpenAI 协议压测） |
| `benchmark_serving_structured_output.py` | 结构化输出 |
| `benchmark_prefix_caching.py` | prefix cache 效果 |
| `benchmark_prefix_block_hash.py` | block hash 性能 |
| `benchmark_block_pool.py` | block pool 性能 |
| `benchmark_hash.py` | hash 工具 |
| `benchmark_ngram_proposer.py` | ngram spec decode |
| `benchmark_topk_topp.py` | sampling kernel |
| `benchmark_long_document_qa_throughput.py` | 长上下文 |
| `benchmark_prioritization.py` | 优先级调度 |
| `benchmark_batch_invariance.py` | batch-invariant 模式 |
| `backend_request_func.py` | 各 backend client（vLLM / TGI / OpenAI / DeepSpeed） |
| `benchmark_utils.py` |  |
| `auto_tune/` | 自动调参 |
| `attention_benchmarks/` | attention kernel benchmark |
| `kernels/` / `fused_kernels/` / `cutlass_benchmarks/` | kernel 级 |
| `disagg_benchmarks/` | disaggregated prefill |
| `multi_turn/` | 多轮对话 |
| `overheads/` | overhead 测量 |

---

## 18. `examples/`：使用范例

| 子目录 | 内容 |
| --- | --- |
| `basic/` | 最简单 demo |
| `offline_inference/` | 离线推理脚本 |
| `applications/` | 完整应用 |
| `deployment/` | 部署示例（K8s YAML） |
| `disaggregated/` | Prefill / Decode 分离 |
| `features/` | 各功能示例（LoRA / spec decode / 量化 / structured output） |
| `generate/` | 生成参数示例 |
| `pooling/` | embedding / rerank |
| `reasoning/` | reasoning 模型 |
| `rl/` | RL 训练接入 |
| `tool_calling/` | function calling |
| `speech_to_text/` | Whisper |
| `ray_serving/` | Ray Serve 集成 |
| `observability/` | OTel 配置 |
| `template_*.jinja` / `tool_chat_template_*.jinja` | chat / tool 模板 |

---

## 19. `tests/`：测试

按 vllm 子模块对应：

- `tests/engine/` / `tests/spec_decode/` / `tests/samplers/` / `tests/lora/` / `tests/multimodal/` / `tests/models/` / `tests/kernels/` / `tests/distributed/` ...
- `basic_correctness/` —— 端到端正确性
- `evals/` —— 模型评估
- `compile/` —— torch.compile 测试
- `prompts/` —— 测试用 prompt
- `standalone_tests/` —— 不需要装 vLLM 的独立测试
- `conftest.py` —— pytest 共享 fixture
- `ci_envs.py` —— CI 环境变量

---

## 20. `docs/`：官方文档源

| 目录 | 内容 |
| --- | --- |
| `getting_started/` | 入门 |
| `serving/` | 服务部署 |
| `models/` | 模型支持矩阵 |
| `features/` | 各功能 |
| `deployment/` | K8s / Docker |
| `design/` | 架构设计文档（**含 V1 设计文档**） |
| `api/` | API 参考 |
| `cli/` | CLI 参考 |
| `usage/` | 使用模式 |
| `benchmarking/` | 性能 benchmark |
| `configuration/` | 配置参考 |
| `contributing/` | 贡献指南 |
| `community/` |  |
| `governance/` |  |
| `training/` |  |
| `mkdocs/` | mkdocs 配置 |
| `assets/` | 图片 |
| `examples/` | docs 引用的例子 |

---

## 21. "我要找 X，去哪里"速查表

| 想做什么                          | 第一站                                                              |
| ----------------------------- | ---------------------------------------------------------------- |
| 看一个请求生命周期                  | `v1/engine/llm_engine.py` → `v1/engine/core.py`                  |
| 看调度逻辑                       | `v1/core/sched/scheduler.py`                                     |
| 看 KV cache 怎么分               | `v1/core/{block_pool,kv_cache_manager}.py`                       |
| 看 forward 怎么跑                 | `v1/worker/gpu_model_runner.py`                                  |
| 看 attention kernel             | `v1/attention/backends/flash_attn.py` → `csrc/attention/`        |
| 看采样                         | `v1/sample/sampler.py`                                           |
| 看一个模型实现                    | `model_executor/models/<model>.py`                               |
| 看量化方法                       | `model_executor/layers/quantization/<method>.py`                 |
| 看 OpenAI API                  | `entrypoints/openai/api_server.py`                               |
| 看 TP / PP / EP 切             | `model_executor/layers/linear.py` + `distributed/parallel_state.py` |
| 看 multimodal                  | `multimodal/inputs.py` + 模型文件                                    |
| 看 LoRA                       | `lora/model_manager.py` + `lora/punica_wrapper/`                 |
| 看投机解码                       | `v1/spec_decode/` + `v1/sample/rejection_sampler.py`             |
| 看结构化输出                     | `v1/structured_output/__init__.py`                               |
| 看 CUDA Graph + compile         | `compilation/backends.py`                                        |
| 看环境变量                       | `envs.py`                                                        |
| 看一个配置项                     | `config/<sub>.py`                                                |
| 看 benchmark 怎么跑              | `benchmarks/benchmark_serving.py`（或 `vllm bench serve`）       |
| 看官方设计文档                    | `docs/design/`（含 V1 架构、KV cache 设计、scheduler）              |
| 看模型支持矩阵                    | `docs/models/`                                                   |

---

## 22. 这份地图怎么用

1. **写代码前先来这查**：要修改/扩展某个特性，先在地图里找到对应文件，再去打开看
2. **读源码迷路时回这查**：知道自己在哪一层
3. **面试官问"X 在哪实现的"时**：直接说出 file path

读完这节，配合 `03-code-walkthrough/`（重点文件深读）你就能拍着胸脯说"vLLM 我熟"。

---

## 小结

- vLLM 仓库的"重量级"目录就 4 个：**`vllm/v1/`**（V1 引擎、调度、KV 管理、采样、worker）+ **`vllm/model_executor/`**（模型实现、量化、loader、layers）+ **`vllm/csrc/`**（CUDA/C++ kernel）+ **`vllm/distributed/`**（TP/PP/EP/DP 通信）。其余目录围绕这四块展开。
- 入口在 `vllm/entrypoints/`，配置全集中在 `vllm/config/`，环境变量全在 `vllm/envs.py`——这是查任何"用户可见参数 / 默认值"的三大入口。
- Attention 路径**双层**：`vllm/v1/attention/backends/<backend>.py` 是 Python 后端（选 FlashAttn / FlashInfer / Triton / MLA / Mamba 等），真正的 CUDA kernel 在 `csrc/attention/` 与 `csrc/quantization/`、`csrc/moe/`。
- 模型实现都遵循 `XxxAttention / XxxMLP / XxxDecoderLayer / XxxModel / XxxForCausalLM` 五层结构——读懂 Llama 一份模型，其他 292 个模型基本都能照着扫。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. 三个任务的入口文件。**

| 任务 | 第一个打开的文件 |
| --- | --- |
| Prefix caching 怎么算 hash | `vllm/v1/core/kv_cache_utils.py`（`hash_block_tokens`，约 line 541）|
| Chunked prefill 怎么切 | `vllm/v1/core/sched/scheduler.py`（`_schedule_waiting` + token budget）|
| 加一个新模型 | `vllm/model_executor/models/registry.py`（注册入口），复制最像的现有模型改 |

---

**2. Python `flash_attn.py` vs CUDA `paged_attention_v2.cu` 怎么连？**

- **`vllm/v1/attention/backends/flash_attn.py`**：构造 attention metadata（block_table / seq_lens / query_start_loc）、决定调哪个 kernel。
- **`csrc/attention/paged_attention_v2.cu`**：实际 GPU kernel，通过 block_table 间接寻址访问物理 KV block。
- **桥**：`vllm/_custom_ops.py` 用 `torch.library` 注册 CUDA op，Python 端调 `torch.ops.vllm.paged_attention_v2(...)`，PyTorch dispatcher 自动派发。这样 torch.compile 不会 graph break。

---

**3. 量化拆 Python / CUDA 两层的原因。**

- `vllm/model_executor/layers/quantization/`（Python）：算法描述、权重加载、Linear 层 wrap、决定调哪个 kernel
- `csrc/quantization/`（CUDA）：纯 kernel（marlin / awq / gptq / fp8 / cutlass_w8a8）

**拆两层原因**：

1. 算法与硬件解耦——同一 AWQ 可适配 SM80/SM90/SM100，Python 不动
2. 多算法共享 kernel——fp8 kernel 同时给 fp8-static / fp8-dynamic / fp8-marlin 用
3. front-end / back-end 分离：Python 编排，CUDA 算

---

**4. 加 NVMe KV offload 后端要改哪？**

```
vllm/v1/kv_offload/
├── base.py              ← 已有，抽象接口
├── factory.py           ← 已有，注册要加 "nvme" → 你的 connector
├── nvme/                ← 新目录
│   ├── connector.py     ← 实现 base 接口（save/load/evict 必须异步）
│   ├── pool.py          ← NVMe block pool（mmap + io_uring）
│   └── config.py        ← 容量、路径、并发
```

外加 `vllm/config/cache.py` 或 `kv_transfer_config` 暴露 `--kv-offload-backend nvme`，`tests/v1/kv_offload/test_nvme.py` 新增单测。

**关键**：NVMe 比 DRAM 慢 100×，接口必须**异步**返回 `Future`，否则把 step 时长拖死。

---

**5. 30 秒速查 3 件事。**

| 任务 | 入口 | 主要文件 |
| --- | --- | --- |
| TP / PP / EP 切 | `vllm/distributed/` | `parallel_state.py`、`device_communicators/all2all.py`、`eplb/` |
| OpenAI API | `vllm/entrypoints/openai/` | `api_server.py`、`serving_chat.py` |
| LoRA 加载 | `vllm/lora/` | `model_manager.py`、`punica_wrapper/`、`layers/` |

口诀：分布式 → `distributed/`；HTTP 入口 → `entrypoints/`；特性（LoRA / 多模态 / spec_decode）各占顶层目录。

## 下一步

- 下一节：[`05-process-and-ipc-internals.md`](05-process-and-ipc-internals.md)（地图看完，正式深入 `vllm/v1/engine/` + `vllm/v1/executor/` + `vllm/distributed/device_communicators/` 的进程与 IPC 内部）。
- 源码深读起点：[`03-code-walkthrough/01-entry-points.md`](../03-code-walkthrough/01-entry-points.md)（从 `LLM(...)` 一路 trace 到 `step()`）。
- 想看核心创新：[`02-core-concepts/01-paged-attention.md`](../02-core-concepts/01-paged-attention.md)（带着地图去读 `vllm/v1/core/block_pool.py` 不会迷路）。
- 想动手：[`07-hands-on/01-setup.md`](../07-hands-on/01-setup.md)（把仓库 clone 下来照着这张地图浏览一遍）。
