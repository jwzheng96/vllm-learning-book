# 01. 量化（Quantization）

> **谁该读这一篇？** 准备给大模型做部署调优、需要在不同硬件上选 FP8 / AWQ / GPTQ / KV-cache 量化的工程师；以及面试想答清"量化收益、Marlin、KV FP8"等问题的同学。
>
> **前置阅读：** [`00-prerequisites.md`](../01-overview/00-prerequisites.md)、[`04-model-runner.md`](../03-code-walkthrough/04-model-runner.md)、[`05-attention-backends.md`](../03-code-walkthrough/05-attention-backends.md)
>
> **耗时：** 约 10 分钟
>
> **学完能：**
> 1. 解释 FP8 / INT4 weight-only / KV 量化各自的收益场景与代价
> 2. 说出 AWQ 与 GPTQ 的核心算法差异
> 3. 解释 Marlin 为什么能让 INT4 W × FP16 A 比反量化路径快 3-4×
> 4. 根据硬件（H100 / A100 / 4090 / MI300）选出合适的量化组合并写出 vLLM 启动命令

把权重 / 激活 / KV 从 FP16/BF16 压到 FP8/INT8/INT4，换显存、换带宽、换吞吐。代价是少量精度损失。

---

## 1. 为什么量化？

LLM 推理瓶颈：

- **Decode 阶段访存密集**（FLOPs/Bytes ~ 1），weight 从 HBM 读到 SRAM 是主要时间
- **显存有限**（H100 80GB 只能装 40B FP16 模型）
- **KV cache 占用**与 dtype 线性相关

量化的核心收益：

- 权重压缩：FP16 → INT4 = **4× 显存节省 + 4× 带宽节省**
- decode 阶段几乎线性提速
- prefill 阶段提速有限（compute-bound）

---

## 2. 量化的三个维度

| 量化对象       | 收益                  | 损失     | 典型方法               |
| ---------- | ------------------- | ------ | ------------------ |
| Weight only | 显存 + 带宽            | 小      | AWQ / GPTQ / Marlin |
| Weight + Activation | + 计算（INT8 matmul） | 中    | SmoothQuant / FP8  |
| KV cache    | KV 显存（间接提升并发数）       | 小      | FP8 / INT8 KV      |

---

## 3. 常见量化格式速览

### 3.1 FP8（H100+）
- 两种格式：E4M3（精度高）、E5M2（动态范围大）
- 权重通常用 E4M3，梯度/激活用 E5M2（训练）
- 推理时：weight + activation 都 FP8，matmul 直接 FP8 Tensor Core 跑
- 精度损失：约 1%（perplexity）
- **vLLM 支持**：`--quantization fp8`

### 3.2 INT8 / INT4 Weight-Only
- 只压权重，激活保 FP16
- 计算时反量化权重再 matmul
- 损失很小（INT8 < 0.5%，INT4 1-2%）
- 关键 kernel：**Marlin**（INT4 W × FP16 A 的极速 GEMM）

### 3.3 AWQ（Activation-aware Weight Quantization）
- 量化前观察激活，识别"重要" channel
- 给重要 channel 加 scale 保留精度
- INT4 W 下 perplexity 比 GPTQ 略好
- 文件：`vllm/model_executor/layers/quantization/awq.py`、`awq_marlin.py`

### 3.4 GPTQ（Generative Pre-trained Transformer Quantization）
- 逐层做 Hessian 二阶校正量化
- INT4 W 标准方法
- 文件：`vllm/model_executor/layers/quantization/gptq.py`、`gptq_marlin.py`

### 3.5 BitsAndBytes（NF4 / FP4）
- QLoRA 同款，主要服务训练但 vLLM 也支持
- 不如 AWQ/GPTQ 快，主要图易用

### 3.6 KV Cache 量化
- `--kv-cache-dtype fp8` / `fp8_e4m3` / `fp8_e5m2`
- KV 写入时量化、读出时反量化
- 每 token KV 占用减半 → **num_blocks ×2**

---

## 4. vLLM 的量化目录

```
vllm/model_executor/layers/quantization/
├── __init__.py             - 注册表
├── fp8.py                  - FP8 implementation
├── awq.py / awq_marlin.py
├── gptq.py / gptq_marlin.py
├── compressed_tensors/     - Neural Magic 的统一格式
├── kernels/                - Marlin / Machete / 自定义 GEMM
├── bitsandbytes.py
├── auto_round/
├── modelopt.py             - NVIDIA Model Optimizer
├── deepseek_v4_*           - DeepSeek 专用低比特
└── ...
```

vLLM 推荐路径：**通过 `LinearMethodBase` 接口让每种量化方法实现自己的 Linear 层替换**。

---

## 5. 量化的关键 trick

### 5.1 异常值（Outlier）处理
LLM activation 有少量极大值（outlier），naive 量化会让它们挤压剩余值。

- AWQ：激活感知，重要 channel 不量化或加大 scale
- SmoothQuant：把 activation 的难度迁移到 weight（scale + 反 scale）

### 5.2 Per-channel / Per-tensor / Per-group
- per-tensor：整个矩阵一个 scale（最简单，最不准）
- per-channel：每列/行一个 scale（中等）
- per-group：每 128 个元素一个 scale（INT4 标配）

### 5.3 Marlin Kernel 的魔法
INT4 W × FP16 A 直接做 GEMM 比"反量化再 FP16 GEMM"快 3-4×。原理：

- 用 tensor core 的 mixed precision 模式
- 反量化和乘法在寄存器层面融合
- 极致的 shared memory layout

---

## 6. 推荐配置（按硬件）

| 硬件         | 推荐量化                          | 备注                          |
| ---------- | ----------------------------- | --------------------------- |
| H100/H200  | FP8 (weight + activation)     | 原生支持，损失最小                   |
| A100       | AWQ INT4 或 GPTQ INT4 (Marlin)  | A100 没 FP8 硬件               |
| L40 / RTX 4090 | AWQ INT4                  | 消费卡显存小，量化收益最大              |
| MI300       | FP8（rocm 支持）                  | AMD 也有 FP8 了                |
| TPU / CPU  | INT8 / BF16                  | 走 OneDNN / XLA               |

---

## 7. 实操示例

```bash
# 在 vLLM 中使用预量化好的 AWQ 模型
vllm serve TheBloke/Llama-2-70B-AWQ \
    --quantization awq \
    --tensor-parallel-size 2

# 使用 FP8 模型（H100）
vllm serve neuralmagic/Meta-Llama-3-70B-FP8 \
    --tensor-parallel-size 2

# 在线 FP8 量化（不推荐，先用 llmcompressor 离线量化）
vllm serve meta-llama/Llama-3-70B \
    --quantization fp8 \
    --tensor-parallel-size 2

# KV cache FP8（与权重量化正交）
vllm serve meta-llama/Llama-3-70B \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 2
```

---

## 8. 面试常见追问

**Q: 量化哪里损失最大？**
A: outlier（异常激活值）和 LM head（生成 logits 的最后一层，对精度敏感）。所以 vLLM 默认 LM head 不量化。

**Q: FP8 比 INT8 好在哪？**
A: FP8 是浮点，保留小数和指数，处理大动态范围比 INT8 强；硬件支持后计算几乎免费（H100 FP8 TFLOPS = FP16 ×2）。INT8 需要 zero-point + scale，反量化更繁琐。

**Q: 为什么 KV 量化在 vLLM 里很简单？**
A: KV 写入时量化、读出时反量化（在 attention kernel 内部完成），不需要全局 scaler 调整。FlashAttention 和 vLLM 自己的 paged kernel 都支持 FP8 KV。

**Q: 量化后模型为什么有时反而变慢？**
A: 反量化开销 > 算力节省时（小 batch、prefill 阶段 compute-bound、没用 Marlin）。要看具体 workload。

---

## 小结

- 量化三层正交：weight-only（AWQ/GPTQ/Marlin）、weight+activation（FP8/SmoothQuant）、KV cache（FP8）。weight-only 主要省显存与带宽，KV 量化几乎翻倍 num_blocks。
- AWQ 是激活感知（保护重要 channel），GPTQ 是 Hessian 二阶校正；INT4 下 AWQ 略好，两者都用 Marlin 反量化加速。
- Marlin 把反量化和 GEMM 在寄存器层融合，配合 tensor core mixed precision，跑 INT4 W × FP16 A 比"反量化再 FP16 GEMM"快 3-4×。
- 硬件选型：H100/H200 优先 FP8；A100/L40/4090 用 AWQ-INT4 + Marlin；MI300 走 ROCm FP8。LM head 通常不量化以保精度。

## 自检

> 答案不必照搬，能讲到关键点即可。

**1. AWQ 与 GPTQ 入口 + 是否都调 Marlin？**

- **AWQ**：`vllm/model_executor/layers/quantization/awq.py`（基础 awq dequant + gemm）+ `awq_marlin.py`（Marlin 加速版）
- **GPTQ**：`vllm/model_executor/layers/quantization/gptq.py` + `gptq_marlin.py`

**`apply` 是否调 Marlin**：

- 默认场景（H100/H200 + INT4）：**两者都走 Marlin**——AWQ-Marlin 和 GPTQ-Marlin 共享底层 `csrc/quantization/marlin/` kernel
- 不能走 Marlin 的场景：硬件不支持（A100 之前）、checkpoint 格式不兼容、group_size 不匹配 → fallback 到原始 dequant + cublas gemm

**判断方法**：启动时 vLLM log 会打"Using Marlin kernel"或"Using AWQ kernel (fallback)"。生产部署要确认走了 Marlin，不然性能差 3×。

---

**2. Llama-70B FP16/FP8/AWQ-INT4 在 80GB H100 装多少？KV 留多少 block？**

| 配置 | 模型权重 | 可用 KV | num_blocks | 理论并发 |
| --- | --- | --- | --- | --- |
| FP16 | 70 × 2 = 140 GB | ❌ 装不下单 80GB H100 | — | 必须 TP-2+ |
| FP8 | 70 × 1 = 70 GB | 80-70-5(activation) = ~5 GB | ~5GB / 4KB-per-token-per-layer / 80 layer × 80 GB ≈ 1k block | 极少（10 个 short 请求左右）|
| AWQ-INT4 | 70 × 0.5 = 35 GB | 80-35-5 = 40 GB | ~40GB / 4KB / 80 = 130k token = ~8000 block | 大（>50 个并发）|

**实战推荐**：单机单 H100 跑 70B 必须 INT4（AWQ 或 GPTQ）。FP8 也能装但 KV 太小，几乎无并发能力。多机 / 多卡 TP=2 才能上 FP16/FP8 跑 70B。

加分点：实际还有 CUDA Graph buffer / Marlin workspace 等 1-2 GB 占用。生产部署用 `--gpu-memory-utilization 0.9` 留 buffer。

---

**3. `--kv-cache-dtype fp8` 启用后 attention kernel 怎么改？**

KV cache 物理存储改为 FP8（1 byte/element 而非 2）。Attention kernel 改动：

```
原 FP16 路径：
  K = load_fp16(K_cache[block, offset, ...])        # 直接 load
  attention_score = Q · K^T

FP8 路径：
  K_fp8 = load_fp8(K_cache[block, offset, ...])     # FP8 load
  K = K_fp8.to(fp16) * k_scale                       # 反量化回 FP16
  attention_score = Q · K^T
```

**读出来是 FP16**（或 BF16），attention 内部 compute 仍是 FP16 精度——只是 HBM 存储省了。

**实现细节**：

- `k_scale`、`v_scale`：每个 layer 一个（可选 per-tensor），从 calibration 来
- attention kernel 模板特化：FlashAttention v3 支持 fp8 KV
- 写入端 `slot_mapping`：写之前先 quantize（FP16 → FP8 + scale）

**精度损失**：典型 < 1% perplexity 退化。Math reasoning / code 等对小数值敏感的任务可能更明显，要 ablation。

---

**4. 量化反而比 FP16 慢的场景。**

| 场景 | 慢的原因 |
| --- | --- |
| **极小 batch（< 4）+ 短 seq**：算力本不富余 | 反量化的 dequant kernel 启动开销 ≥ 节省的 HBM 带宽收益 |
| **prefill compute-bound 阶段**（长 prompt + 大 batch）| FP16 直接 GEMM 算力是瓶颈；反量化加了 dequant 步骤，反而拖慢 |
| **AWQ/GPTQ 走非 Marlin fallback**（如 A100 之前的卡）| dequant + cublas gemm 比 Marlin fused 慢 2-3× |
| **激活仍 FP16**：W4A16 量化 | matmul 时实际算 dequant(W) · A，仍是 FP16 算力 |
| **量化粒度极细**（per-channel 甚至 per-token）| 反量化时 scale 表很大，影响 cache 效率 |
| **多 LoRA 切换频繁**：每个 LoRA 单独反量化 | LoRA 的 A/B 是 FP16，与量化主权重交互时频繁切 dtype |

→ **量化是 decode（memory-bound）的朋友、prefill（compute-bound）的敌人**。生产中如果 prefill 占比很高（短 prompt + 短输出），量化收益小甚至为负。

---

**5. 为什么默认不量化 LM head？换成 FP8 会怎样？**

**LM head**：`hidden_state [B, H] · W_lm_head [H, V] → logits [B, V]`。V 通常 128K（vocab size），矩阵很大但只跑**一次/请求**（decode 时每步 1×）。

**不量化原因**：

1. **输出敏感性**：LM head 直接产 logits，logits 的微小偏差经 softmax 放大后**采样分布显著偏移**——不同于 attention/MLP 输出还会被后续层"消化"
2. **精度需求**：top-k / top-p 采样依赖排序，FP8 的 1 位 mantissa 精度可能让相邻概率的 token 排序错位
3. **总收益小**：LM head 只占模型权重的 ~5%（70B 模型 LM head 才 ~4 GB），量化收益不大但风险大

**量化 LM head 会怎样**：

- Perplexity 通常退化 2-5%（attention/MLP 量化 < 1%）
- 容易出"小幅高频错误"——比如把 "the" 输出成 "The"、"is" → "Is"、token 边界附近偏移
- code generation / 结构化输出（JSON / function call）尤其敏感

**例外**：极致追求显存的场景（如 8B 模型上 LM head 占 25% 时），可以量化 LM head 但建议用 INT8 而非 FP8/INT4，平衡精度与显存。

参数：`--quantization-lm-head` 显式开启（默认 false）。

## 下一步

- 下一节：[`02-speculative-decoding.md`](02-speculative-decoding.md)（另一类正交优化：用小模型提议、大模型验证）
- 想看源码：`vllm/model_executor/layers/quantization/awq_marlin.py`、`vllm/model_executor/layers/quantization/fp8.py`、`csrc/quantization/gptq_marlin/`
- 想动手：[`07-hands-on/03-mini-experiments.md`](../07-hands-on/03-mini-experiments.md)（同模型跑 FP16 vs AWQ vs FP8，对比 throughput / latency / 显存占用）
- 想从生产视角理解：[`08-production-deployment/04-autoscaling-and-capacity.md`](../08-production-deployment/04-autoscaling-and-capacity.md)（量化对单卡容量与单位 token 成本的影响）

